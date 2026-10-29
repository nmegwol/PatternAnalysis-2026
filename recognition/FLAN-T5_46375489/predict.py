"""Load checkpoints and generate simplified reports."""

from pathlib import Path
from time import perf_counter

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from modules import DEFAULT_MODEL_NAME
from utils import DEFAULT_PROMPT_TEMPLATE, read_json, write_json


# Shared metadata saved with both prediction and training resource metrics.
GENERATION_METRICS_SCOPE = (
    "Returned output tokens after the decoder-start token, through the first "
    "EOS inclusive; padding after EOS, encoder input tokens and unreturned "
    "beam candidates are excluded. Without EOS, every returned position after "
    "decoder-start counts. Generation time is device-synchronized "
    "model.generate() wall time, including its encoder and decoder work; "
    "excludes model loading, input tokenization/transfers, loss, decoding, "
    "scoring and file writes. Tokens/second is total returned tokens divided "
    "by total generation seconds across batches."
)


@torch.inference_mode()
def generate_with_metrics(model, inputs, **generation_kwargs):
    """Generates token sequences and measures generation time and throughput.

    Inputs must be on the model's device. The caller sets evaluation mode;
    gradient tracking is disabled here.

    Args:
        model: Seq2seq model whose generate() method returns a token-ID tensor.
        inputs: Dict of input tensors, including input_ids and attention_mask.
        **generation_kwargs: Options passed to model.generate().

    Returns:
        A (token_ids, metrics) tuple. token_ids is the generated tensor; metrics
        records total and per-sequence token counts, seconds and tokens/second.
    """
    device = inputs["input_ids"].device

    def synchronize():
        """Waits for pending accelerator work; does nothing on CPU."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    # Accelerator calls are asynchronous, so wait on both sides of generation.
    synchronize()
    started = perf_counter()
    generated = model.generate(**inputs, **generation_kwargs)
    synchronize()
    generation_seconds = perf_counter() - started

    # The initial decoder-start token is supplied by the model, not generated.
    tokens = generated[:, 1:]
    generation_config = (
        generation_kwargs.get("generation_config") or model.generation_config
    )
    eos_ids = generation_kwargs.get(
        "eos_token_id", generation_config.eos_token_id
    )
    eos_mask = torch.zeros_like(tokens, dtype=torch.bool)
    if eos_ids is not None:
        # Generation supports either one EOS ID or several alternative EOS IDs.
        if not isinstance(eos_ids, (list, tuple)):
            eos_ids = [eos_ids]
        for token_id in eos_ids:
            eos_mask |= tokens.eq(token_id)

    # Count tokens with no earlier EOS, including the first EOS itself.
    # This excludes trailing padding even when PAD and EOS share an ID.
    # With no EOS, every returned position after decoder-start is counted.
    earlier_eos_count = eos_mask.cumsum(dim=1) - eos_mask.long()
    through_first_eos = earlier_eos_count == 0
    token_counts = through_first_eos.sum(dim=1).tolist()
    total_tokens = sum(token_counts)

    return generated, {
        "generated_tokens": total_tokens,
        "generated_token_counts": token_counts,
        "generation_seconds": generation_seconds,
        "generated_tokens_per_second": total_tokens / generation_seconds,
    }


class RadiologyPredictor:
    """Generates plain-language summaries from source reports alone.

    Attributes:
        last_metrics: Dict of metrics from the latest prediction call, or None
            before prediction. Single-report and file calls also update it.
    """

    SOURCE_COLUMN = "radiology_report"
    PREDICTION_COLUMN = "layman_report"

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device="cpu",
        batch_size=4,
        max_input_length=512,
        max_new_tokens=256,
        num_beams=4,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
    ):
        """Configures prediction and moves the model to the selected device.

        Args:
            model: Hugging Face seq2seq model, optionally with PEFT adapters.
            tokenizer: Hugging Face tokenizer matching the model.
            device: Device name (str) or torch.device for inference.
            batch_size: Integer number of reports per generation batch.
            max_input_length: Integer token limit for each prompted report.
            max_new_tokens: Integer limit on generated tokens per report.
            num_beams: Integer beam count; 1 uses greedy decoding.
            prompt_template: Prompt string containing a {report} placeholder.
        """
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
    def from_checkpoint(
        cls,
        checkpoint_dir,
        *,
        model_name=DEFAULT_MODEL_NAME,
        fine_tuning_mode="lora",
        model_revision=None,
        **kwargs,
    ):
        """Restores model weights and a tokenizer for inference.

        Optimizer state and training progress are not restored.

        Args:
            checkpoint_dir: Saved model or adapter directory (str or PathLike).
            model_name: Base-model Hub ID or directory for LoRA (str/PathLike).
            fine_tuning_mode: Checkpoint type (str): "full" or "lora".
            model_revision: Optional Hub revision string for the LoRA base.
            **kwargs: Predictor settings forwarded to the constructor.

        Returns:
            A RadiologyPredictor using the checkpoint's model and tokenizer.

        Raises:
            ValueError: If fine_tuning_mode is unsupported.
        """
        if fine_tuning_mode not in ("full", "lora"):
            raise ValueError("fine_tuning_mode must be 'full' or 'lora'.")

        if fine_tuning_mode == "full":
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        else:
            # An adapter checkpoint needs its matching base-model weights.
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                revision=model_revision,
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_dir)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        return cls(model, tokenizer, **kwargs)

    def predict(self, report):
        """Generates a summary for one report and updates last_metrics.

        Args:
            report: Source radiology report (str).

        Returns:
            The generated plain-language summary (str).
        """
        return self.predict_batch([report])[0]

    @torch.inference_mode()
    def predict_batch(self, reports):
        """Generates summaries in input order and updates last_metrics.

        Sets the model to evaluation mode. Empty input returns an empty list
        with zero token counts and timing.

        Args:
            reports: Sequence of source radiology report strings.

        Returns:
            A list of summary strings in the same order as reports.
        """
        self.model.eval()
        predictions = []
        generated_tokens = 0
        generation_seconds = 0.0

        for start in range(0, len(reports), self.batch_size):
            batch = reports[start:start + self.batch_size]
            prompts = [
                self.prompt_template.format(report=report) for report in batch
            ]
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
                return_tensors="pt",
            ).to(self.device)
            generated, metrics = generate_with_metrics(
                self.model,
                inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
            generated_tokens += metrics["generated_tokens"]
            generation_seconds += metrics["generation_seconds"]
            decoded = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            predictions.extend(text.strip() for text in decoded)

        # Use total tokens / total time so each batch is weighted by its time.
        # Empty input has no generation time and therefore zero throughput.
        tokens_per_second = (
            generated_tokens / generation_seconds if generation_seconds else 0.0
        )
        self.last_metrics = {
            "examples": len(reports),
            "generated_tokens": generated_tokens,
            "generation_seconds": generation_seconds,
            "generated_tokens_per_second": tokens_per_second,
            "generation_metrics_scope": GENERATION_METRICS_SCOPE,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "max_input_length": self.max_input_length,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
        }
        return predictions

    def predict_file(self, input_path, output_path):
        """Writes prediction JSON and a sibling <output-stem>_metrics.json file.

        Args:
            input_path: JSON path (str/PathLike) containing a list of objects
                with a radiology_report field.
            output_path: Destination JSON path (str/PathLike). Each output
                object contains a layman_report field, preserving input order.

        Raises:
            KeyError: If an input object lacks radiology_report.
        """
        reports = [row[self.SOURCE_COLUMN] for row in read_json(input_path)]
        predictions = self.predict_batch(reports)
        output_rows = [{self.PREDICTION_COLUMN: text} for text in predictions]
        write_json(output_rows, output_path)

        output_path = Path(output_path)
        metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
        write_json(self.last_metrics, metrics_path)
