"""FLAN-T5 model construction; COMP3710 Task 2.6, student 46375489.

Staff clarification #154 permits Transformers and PEFT. Only those libraries
and PyTorch are imported here; data, CLI and checkpoint handling live elsewhere.
"""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM


def build_model(
    model_name, mode="baseline", revision=None, rank=8, alpha=16, dropout=0.05
):
    """Load FP32 T5 weights; freeze baseline, train q/v LoRA, or train all.

    model_name may be a pinned Hub model or a local save_pretrained directory.
    Forward inputs are padded int64 input_ids/attention_mask and target labels
    [batch, length], with -100 for ignored target padding. Returns a CPU model.
    """
    if mode not in {"baseline", "lora", "full"}:
        raise ValueError("mode must be baseline, lora or full")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.float32,
        token=False,
    )
    if model.config.model_type != "t5":
        raise ValueError("This project expects a T5 encoder-decoder")
    model.requires_grad_(mode == "full")
    if mode == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=["q", "v"],
                bias="none",
                revision=revision,
            ),
        )
    return model


def parameter_counts(model):
    """Count distinct total and trainable parameter elements."""
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
