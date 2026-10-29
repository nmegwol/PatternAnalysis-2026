"""Load FLAN-T5 for full fine-tuning or LoRA."""

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM

DEFAULT_MODEL_NAME = "google/flan-t5-base"


def load_model(model_name=DEFAULT_MODEL_NAME, fine_tuning_mode="lora",
               lora_r=8, lora_alpha=16, lora_dropout=0.05,
               lora_target_modules=("q", "v"), model_revision=None):
    if fine_tuning_mode not in ("full", "lora"):
        raise ValueError("fine_tuning_mode must be 'full' or 'lora'.")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=model_revision)
    if fine_tuning_mode == "lora":
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules), bias="none",
        ))
    return model
