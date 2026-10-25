"""Shared runtime, native checkpoint reload and generation; student 46375489."""

import json
import time
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, GenerationConfig, set_seed

from modules import build_model


def write_json(path, value):
    """Write local experiment settings or measurements, rejecting NaN values."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def setup_runtime(device="auto", seed=42, bf16=False):
    """Seed the run and select CPU/CUDA; reject unsupported mixed precision."""
    set_seed(seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if bf16:
        if device.type != "cuda":
            raise ValueError("BF16 requires supported CUDA hardware")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This CUDA device does not support BF16")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return device


def synchronize(device):
    """Wait for queued CUDA work before reading a timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_checkpoint(path, device):
    """Load standard full-model or PEFT adapter files; no project recipe needed."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    if (path / "adapter_config.json").exists():
        adapter = PeftConfig.from_pretrained(path)
        base = build_model(
            adapter.base_model_name_or_path, revision=adapter.revision
        )
        model = PeftModel.from_pretrained(base, path, is_trainable=False)
    else:
        model = build_model(str(path))
    if (path / "generation_config.json").exists():
        # PEFT seq2seq generation delegates to the underlying T5 config.
        target = (
            model.get_base_model() if isinstance(model, PeftModel) else model
        )
        target.generation_config = GenerationConfig.from_pretrained(path)
    if model.generation_config.max_new_tokens is None:
        model.generation_config.max_new_tokens = 256
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model.to(device).requires_grad_(False).eval(), tokenizer


@torch.inference_mode()
def generate_summaries(model, tokenizer, loader, device, bf16=False):
    """Generate in loader order from encoder inputs only, with cold-start timing.

    Returns predictions, output lengths (EOS included, padding/start excluded),
    and inference measurements. Time includes loading batches, transfer and
    decoding, but excludes model loading, reference loss and ROUGE calculation.
    """
    was_training = model.training
    model.eval()
    predictions, lengths = [], []
    capped = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    try:
        for batch in loader:
            inputs = {
                k: batch[k].to(device) for k in ("input_ids", "attention_mask")
            }
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=bf16
            ):
                ids = model.generate(**inputs, use_cache=True)
            emitted = ids.cpu()[:, 1:]
            predictions.extend(
                tokenizer.batch_decode(ids, skip_special_tokens=True)
            )
            lengths.extend((emitted != tokenizer.pad_token_id).sum(1).tolist())
            capped += sum(
                tokenizer.eos_token_id not in row.tolist() for row in emitted
            )
        synchronize(device)
        seconds = time.perf_counter() - started
        return (
            predictions,
            lengths,
            {
                "seconds": seconds,
                "examples": len(predictions),
                "tokens_per_second": sum(lengths) / seconds,
                "seconds_per_example": seconds / len(predictions)
                if predictions
                else None,
                "outputs_without_eos": capped,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else None,
            },
        )
    finally:
        model.train(was_training)
