"""Official LaymanRRG splits, overlap checks and tokenisation; student 46375489."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
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
        records = splits[name]
        # Add a row index column for tracking original positions in saved predictions.
        splits[name] = records.add_column("row_index", range(len(records)))
    return splits


def checked_text(text):
    """Check that text is nonblank and strip exterior whitespace."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Reports and required references must be nonblank text"
        )
    return text.strip()


def check_overlap(train, validation):
    """
    Count how much report text is shared between the. training and valdiation datasets, and whether those shared reports have matching reference summaries.
    """

    def normalise(text):
        # Normalise by stripping, collapsing whitespace, and lowercasing.
        # Only normalise text for overlap checking; do not modify original records.
        # If two texts normalise to the same string, they are considered overlapping for the purpose of this check.
        return " ".join(checked_text(text).split()).casefold()

    # Store every distinct target for each normalised source text in the training set.
    # One report may have multiple distinct reference summaries,
    # so we use a set to collect all distinct targets for each source.

    train_refs = defaultdict(set)
    for row in train.select_columns([SOURCE_COLUMN, TARGET_COLUMN]):
        train_refs[normalise(row[SOURCE_COLUMN])].add(
            normalise(row[TARGET_COLUMN])
        )
    validation_refs = defaultdict(set)
    # shared is the unique set of source texts that appear in both the training and validation sets.
    shared = set()
    # matching counts how many validation rows have sources present in the training set.
    # agreeing counts how many of those matching rows have targets that match any of the training targets for that source.
    matching = agreeing = 0
    unseen = []

    for i, row in enumerate(validation):
        source = normalise(row[SOURCE_COLUMN])
        target = normalise(row[TARGET_COLUMN])
        validation_refs[source].add(target)
        # Check if this source text has been seen in the training set.
        if source in train_refs:
            shared.add(source)
            matching += 1
            agreeing += target in train_refs[source]
        else:
            # This source text is not present in the training set.
            unseen.append(row.get("row_index", i))
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "shared_source_texts": len(shared),
        "matching_validation_rows": matching,
        "matching_targets": agreeing,
        "different_targets": matching - agreeing,
        "train_reports_with_conflicting_targets": sum(
            len(refs) > 1 for refs in train_refs.values()
        ),
        "validation_reports_with_conflicting_targets": sum(
            len(refs) > 1 for refs in validation_refs.values()
        ),
        "shared_reports_with_conflicting_targets": sum(
            len(train_refs[s] | validation_refs[s]) > 1 for s in shared
        ),
        "unseen_validation_row_indices": unseen,
    }


def tokenize_split(
    records,
    tokenizer,
    max_input_tokens=512,
    max_target_tokens=256,
    include_labels=True,
):
    """Tokenize and split records into model inputs and labels."""
    required = (
        {SOURCE_COLUMN, TARGET_COLUMN} if include_labels else {SOURCE_COLUMN}
    )
    if not len(records) or not required <= set(records.column_names):
        raise ValueError("Empty records or missing required text columns")
    if (
        min(max_input_tokens, max_target_tokens) < 2
        or tokenizer.eos_token_id is None
    ):
        raise ValueError("T5 needs EOS and token limits of at least 2")

    def preprocess(batch):
        inputs = tokenizer(
            [
                DEFAULT_PROMPT.format(report=checked_text(t))
                for t in batch[SOURCE_COLUMN]
            ],
            truncation=True,
            max_length=max_input_tokens,
        )
        features = {key: inputs[key] for key in ("input_ids", "attention_mask")}
        if include_labels:
            features["labels"] = tokenizer(
                text_target=[checked_text(t) for t in batch[TARGET_COLUMN]],
                truncation=True,
                max_length=max_target_tokens,
            )["input_ids"]
        return features

    return records.map(
        preprocess, batched=True, remove_columns=records.column_names
    )


def make_dataloader(features, tokenizer, batch_size=2, shuffle=False, seed=42):
    """Pad each batch to its longest sequences; label padding defaults to -100."""
    return torch.utils.data.DataLoader(
        features,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorForSeq2Seq(tokenizer),
        generator=torch.Generator().manual_seed(seed),
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
