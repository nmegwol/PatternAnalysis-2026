"""Load checkpoints and generate simplified reports."""

from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from modules import DEFAULT_MODEL_NAME
from utils import DEFAULT_PROMPT_TEMPLATE, read_json, write_json


GENERATION_METRICS_SCOPE = (
    "Returned output tokens after the decoder-start token, through the first EOS inclusive; "
    "padding after EOS, encoder input tokens and unreturned beam candidates are excluded. "
    "Without EOS, every returned position after decoder-start counts. Generation time is "
    "device-synchronized model.generate() wall time, including its encoder and decoder work; "
    "excludes model loading, input tokenization/transfers, loss, decoding, scoring and file writes. "
    "Tokens/second is total returned tokens divided by total generation seconds across batches."
)


@torch.inference_mode()
def generate_with_metrics(model, inputs, **generation_kwargs):
    """Generate seq2seq output and time generation independently of evaluation."""
    device = inputs["input_ids"].device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    synchronize()
    started = perf_counter()
    generated = model.generate(**inputs, **generation_kwargs)
    synchronize()
    seconds = perf_counter() - started

    # Generation starts with one decoder-start token. Finished sequences are
    # padded after EOS; counting through the first EOS also handles pad == EOS.
    tokens = generated[:, 1:]
    config = generation_kwargs.get("generation_config") or model.generation_config
    eos_ids = generation_kwargs.get("eos_token_id", config.eos_token_id)
    eos_mask = torch.zeros_like(tokens, dtype=torch.bool)
    if eos_ids is not None:
        for token_id in eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]:
            eos_mask |= tokens.eq(token_id)
    before_or_at_eos = eos_mask.cumsum(dim=1) - eos_mask.long() == 0
    counts = before_or_at_eos.sum(dim=1).tolist()
    total = sum(counts)
    return generated, {
        "generated_tokens": total,
        "generated_token_counts": counts,
        "generation_seconds": seconds,
        "generated_tokens_per_second": total / seconds,
    }


class RadiologyPredictor:
    """Generate layperson-friendly radiology reports with a trained model."""

    SOURCE_COLUMN = "radiology_report"
    PREDICTION_COLUMN = "layman_report"

    def __init__(self, model, tokenizer, *, device="cpu", batch_size=4,
                 max_input_length=512, max_new_tokens=256, num_beams=4,
                 prompt_template=DEFAULT_PROMPT_TEMPLATE):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.prompt_template = prompt_template
        self.last_metrics = None

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, *, model_name=DEFAULT_MODEL_NAME,
                        fine_tuning_mode="lora", model_revision=None, **kwargs):
        """Restore a full fine-tuned model or LoRA adapter for inference."""
        if fine_tuning_mode not in ("full", "lora"):
            raise ValueError("fine_tuning_mode must be 'full' or 'lora'.")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_dir if fine_tuning_mode == "full" else model_name,
            **({"revision": model_revision} if fine_tuning_mode == "lora" else {}),
        )
        if fine_tuning_mode == "lora":
            model = PeftModel.from_pretrained(model, checkpoint_dir)
        return cls(model, AutoTokenizer.from_pretrained(checkpoint_dir), **kwargs)

    def predict(self, report):
        """Simplify one report; do not require reference summaries."""
        return self.predict_batch([report])[0]

    @torch.inference_mode()
    def predict_batch(self, reports):
        """Generate ordered summaries and expose aggregate throughput in last_metrics."""
        self.model.eval()
        predictions = []
        generated_tokens = 0
        generation_seconds = 0.0
        for start in range(0, len(reports), self.batch_size):
            batch = reports[start:start + self.batch_size]
            inputs = self.tokenizer(
                [self.prompt_template.format(report=report) for report in batch],
                padding=True, truncation=True, max_length=self.max_input_length,
                return_tensors="pt",
            ).to(self.device)
            generated, metrics = generate_with_metrics(
                self.model, inputs,
                max_new_tokens=self.max_new_tokens, num_beams=self.num_beams, do_sample=False,
            )
            generated_tokens += metrics["generated_tokens"]
            generation_seconds += metrics["generation_seconds"]
            predictions.extend(text.strip() for text in self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            ))
        self.last_metrics = {
            "examples": len(reports), "generated_tokens": generated_tokens,
            "generation_seconds": generation_seconds,
            "generated_tokens_per_second": generated_tokens / generation_seconds if generation_seconds else 0.0,
            "generation_metrics_scope": GENERATION_METRICS_SCOPE,
            "device": str(self.device), "batch_size": self.batch_size,
            "max_input_length": self.max_input_length,
            "max_new_tokens": self.max_new_tokens, "num_beams": self.num_beams,
        }
        return predictions

    def predict_file(self, input_path, output_path):
        """Write predictions and a sibling <output-stem>_metrics.json file."""
        reports = [row[self.SOURCE_COLUMN] for row in read_json(input_path)]
        predictions = self.predict_batch(reports)
        write_json([{self.PREDICTION_COLUMN: text} for text in predictions], output_path)
        output_path = Path(output_path)
        write_json(self.last_metrics, output_path.with_name(f"{output_path.stem}_metrics.json"))
