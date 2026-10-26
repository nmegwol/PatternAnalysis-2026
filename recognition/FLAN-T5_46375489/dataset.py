"""Official LaymanRRG splits, overlap checks and tokenisation; student 46375489."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import DataCollatorForSeq2Seq

DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
SOURCE_COLUMN = "radiology_report"
TARGET_COLUMN = "layman_report"
DEFAULT_PROMPT = (
    "Rewrite this radiology report in plain language for a patient, "
    "at approximately a sixth-grade reading level. Preserve the findings, "
    "negations, and uncertainty. Do not add findings or medical advice.\n\n"
    "Radiology report:\n{report}\n\nPlain-language summary:"
)


def load_splits():
    """Load official Hub splits unchanged, with row IDs for saved predictions."""
    splits = load_dataset(DATASET_ID)
    for name in ("train", "validation", "test"):
        if name not in splits or not len(splits[name]):
            raise ValueError(f"Missing or empty official {name} split")
        records = splits[name]
        required = {SOURCE_COLUMN}
        if name != "test":
            required.add(TARGET_COLUMN)
        if not required <= set(records.column_names):
            raise ValueError(f"Missing required columns in {name}: {required}")
        splits[name] = records.add_column("row_index", range(len(records)))
        splits[name] = splits[name].add_column("split", [name] * len(records))
    return splits


def validate_records(records, require_references=True):
    """Reject missing/blank text without silently discarding rows."""
    columns = [SOURCE_COLUMN] + ([TARGET_COLUMN] if require_references else [])
    if not len(records) or not set(columns) <= set(records.column_names):
        raise ValueError("Empty records or missing required text columns")
    for i, row in enumerate(records.select_columns(columns)):
        if any(
            not isinstance(row[c], str) or not row[c].strip() for c in columns
        ):
            raise ValueError(f"Invalid text at selected row {i}")


def check_overlap(train, validation):
    """Count normalised report overlap without filtering either split.

    Target agreement means matching any training target for that report.
    Unseen row IDs use source text only; this does not establish patient or
    semantic independence.
    """
    validate_records(train)
    validate_records(validation)

    def normalise(text):
        return " ".join(text.split()).casefold()

    train_refs = defaultdict(set)
    for row in train.select_columns([SOURCE_COLUMN, TARGET_COLUMN]):
        train_refs[normalise(row[SOURCE_COLUMN])].add(
            normalise(row[TARGET_COLUMN])
        )
    shared = set()
    matching = agreeing = 0
    unseen = []
    for i, row in enumerate(validation):
        source = normalise(row[SOURCE_COLUMN])
        if source in train_refs:
            shared.add(source)
            matching += 1
            agreeing += normalise(row[TARGET_COLUMN]) in train_refs[source]
        else:
            unseen.append(row.get("row_index", i))
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "shared_source_texts": len(shared),
        "matching_validation_rows": matching,
        "matching_targets": agreeing,
        "different_targets": matching - agreeing,
        "unseen_validation_row_indices": unseen,
    }


def tokenize_split(
    records,
    tokenizer,
    max_input_tokens=512,
    max_target_tokens=256,
    include_labels=True,
):
    """Return model features and counts of inputs/targets exceeding token limits.

    Original records/references remain unchanged. Output features contain only
    ragged input_ids, attention_mask and optional labels, ready for the collator.
    """
    validate_records(records, require_references=include_labels)
    if (
        min(max_input_tokens, max_target_tokens) < 2
        or tokenizer.eos_token_id is None
    ):
        raise ValueError("T5 needs EOS and token limits of at least 2")
    features = {"input_ids": [], "attention_mask": []}
    if include_labels:
        features["labels"] = []
    truncated = {"inputs": 0, "targets": 0 if include_labels else None}
    for start in range(0, len(records), 256):
        chunk = records[start : start + 256]
        texts = [
            DEFAULT_PROMPT.format(report=t.strip())
            for t in chunk[SOURCE_COLUMN]
        ]
        # Measure full lengths separately; native truncation retains T5 EOS.
        full_inputs = tokenizer(texts, truncation=False, verbose=False)[
            "input_ids"
        ]
        truncated["inputs"] += sum(
            len(ids) > max_input_tokens for ids in full_inputs
        )
        inputs = tokenizer(texts, truncation=True, max_length=max_input_tokens)
        features["input_ids"].extend(inputs["input_ids"])
        features["attention_mask"].extend(inputs["attention_mask"])
        if include_labels:
            summaries = [t.strip() for t in chunk[TARGET_COLUMN]]
            full_targets = tokenizer(
                text_target=summaries, truncation=False, verbose=False
            )["input_ids"]
            truncated["targets"] += sum(
                len(ids) > max_target_tokens for ids in full_targets
            )
            targets = tokenizer(
                text_target=summaries,
                truncation=True,
                max_length=max_target_tokens,
            )
            features["labels"].extend(targets["input_ids"])
    return Dataset.from_dict(features), truncated


def make_dataloader(features, tokenizer, batch_size=2, shuffle=False, seed=42):
    """Dynamically pad int64 tensors; target padding is -100, not the pad ID."""
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    def collate(rows):
        return {k: v for k, v in collator(rows).items() if v is not None}

    return torch.utils.data.DataLoader(
        features,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit official splits once; no model loading."
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    splits = load_splits()
    audit = check_overlap(splits["train"], splits["validation"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(f"Audit saved to {args.output}; no rows removed.")
