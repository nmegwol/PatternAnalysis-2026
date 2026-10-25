"""Official LaymanRRG splits, overlap checks and tokenisation; student 46375489."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import DataCollatorForSeq2Seq

DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
DATASET_REVISION = "6783ab38f9de28c4372799346f51d3927ae7eb9a"
SOURCE_COLUMN = "radiology_report"
TARGET_COLUMN = "layman_report"
OFFICIAL_SPLITS = ("train", "validation", "test")
DEFAULT_PROMPT = (
    "Rewrite this radiology report in plain language for a patient, "
    "at approximately a sixth-grade reading level. Preserve the findings, "
    "negations, and uncertainty. Do not add findings or medical advice.\n\n"
    "Radiology report:\n{report}\n\nPlain-language summary:"
)


def load_splits(
    data_dir=None,
    splits=("train", "validation"),
    revision=DATASET_REVISION,
    cache_dir=None,
):
    """Load requested official splits, retaining original row indices.

    Local files must be named <split>-*.parquet. Loading does not hash or scan
    every report: validate selected text during tokenisation, or audit explicitly.
    A local path is not a claim that files equal the pinned Hub revision.
    """
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(split not in OFFICIAL_SPLITS for split in splits)
    ):
        raise ValueError("Request unique train, validation or test split names")
    files = {}
    for split in splits:
        if data_dir is None:
            files[split] = [
                f"hf://datasets/{DATASET_ID}@{revision}/data/{split}-*.parquet"
            ]
        else:
            files[split] = sorted(
                str(p) for p in Path(data_dir).glob(f"{split}-*.parquet")
            )
            if not files[split]:
                raise FileNotFoundError(
                    f"No {split} Parquet files in {data_dir}"
                )
    loaded = load_dataset(
        "parquet", data_files=files, cache_dir=cache_dir, token=False
    )
    for split, records in loaded.items():
        if not len(records) or SOURCE_COLUMN not in records.column_names:
            raise ValueError(f"Empty {split} or missing {SOURCE_COLUMN}")
        if {"row_index", "split"} & set(records.column_names):
            raise ValueError("Input uses reserved row_index/split columns")
        loaded[split] = records.add_column(
            "row_index", list(range(len(records)))
        )
        loaded[split] = loaded[split].add_column(
            "split", [split] * len(records)
        )
    return loaded


def sample_records(records, limit=0, seed=42):
    """Select seeded rows in original order; zero means the complete split."""
    if limit < 0:
        raise ValueError("limit must be nonnegative (0 means all)")
    if not limit or limit >= len(records):
        return records
    indices = sorted(random.Random(seed).sample(range(len(records)), limit))
    return records.select(indices)


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


def normalise_text(text):
    """Casefold/collapse whitespace for auditing only; never change model text."""
    return " ".join(text.split()).casefold()


def audit_overlap(train, validation, id_columns=None):
    """Audit supplied splits without filtering them; return counts and unseen IDs.

    Use full official splits for a global audit. Patient/study columns are
    checked when present; pass id_columns for differently named identifiers.
    Image paths are not assumed to be patient/study IDs. No raw IDs are saved.
    """
    validate_records(train)
    validate_records(validation)
    result = {"train_rows": len(train), "validation_rows": len(validation)}
    for name, key in (("exact", str), ("normalised", normalise_text)):
        train_refs = defaultdict(set)
        train_counts = Counter()
        for row in train.select_columns([SOURCE_COLUMN, TARGET_COLUMN]):
            source = key(row[SOURCE_COLUMN])
            train_counts[source] += 1
            train_refs[source].add(key(row[TARGET_COLUMN]))
        val_counts = Counter()
        matching = agreeing = 0
        unseen = []
        for i, row in enumerate(validation):
            source = key(row[SOURCE_COLUMN])
            val_counts[source] += 1
            if source in train_refs:
                matching += 1
                agreeing += key(row[TARGET_COLUMN]) in train_refs[source]
            else:
                unseen.append(row.get("row_index", i))
        shared = train_counts.keys() & val_counts.keys()
        result[name] = {
            "shared_source_texts": len(shared),
            "matching_train_rows": sum(train_counts[s] for s in shared),
            "matching_validation_rows": matching,
            "matching_targets": agreeing,
            "different_targets": matching - agreeing,
            "train_duplicate_rows": len(train) - len(train_counts),
            "validation_duplicate_rows": len(validation) - len(val_counts),
        }
    result["unseen_validation_row_indices"] = unseen  # Normalised comparison.
    columns = id_columns or ("patient_id", "subject_id", "study_id")
    result["identifier_overlap"] = {}
    for column in columns:
        if (
            column not in train.column_names
            or column not in validation.column_names
        ):
            continue
        # Report coverage so missing identifiers never imply independence.
        a = [
            str(v).strip()
            for v in train[column]
            if v is not None and v == v and str(v).strip()
        ]
        b = [
            str(v).strip()
            for v in validation[column]
            if v is not None and v == v and str(v).strip()
        ]
        result["identifier_overlap"][column] = {
            "shared_ids": len(set(a) & set(b)),
            "train_rows_with_id": len(a),
            "validation_rows_with_id": len(b),
        }
    result["limitations"] = (
        "Only supplied identifier columns were checked; missing/absent IDs, "
        "semantic overlap and pretraining exposure remain unverified."
    )
    return result


def tokenize_split(
    records,
    tokenizer,
    max_input_tokens=512,
    max_target_tokens=256,
    include_labels=True,
):
    """Tokenise once, truncate with EOS retained, and return features + counts.

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
        inputs = tokenizer(texts, truncation=False, verbose=False)["input_ids"]
        targets = (
            tokenizer(
                text_target=[t.strip() for t in chunk[TARGET_COLUMN]],
                truncation=False,
                verbose=False,
            )["input_ids"]
            if include_labels
            else []
        )
        for ids in inputs:
            truncated["inputs"] += len(ids) > max_input_tokens
            ids = (
                ids[: max_input_tokens - 1] + [tokenizer.eos_token_id]
                if len(ids) > max_input_tokens
                else ids
            )
            features["input_ids"].append(ids)
            features["attention_mask"].append([1] * len(ids))
        for ids in targets:
            truncated["targets"] += len(ids) > max_target_tokens
            features["labels"].append(
                ids[: max_target_tokens - 1] + [tokenizer.eos_token_id]
                if len(ids) > max_target_tokens
                else ids
            )
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
    parser.add_argument("--data-dir")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--id-columns", nargs="+", help="Optional patient/study column names"
    )
    args = parser.parse_args()
    splits = load_splits(args.data_dir)
    audit = audit_overlap(
        splits["train"], splits["validation"], args.id_columns
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(f"Audit saved to {args.output}; no rows removed.")
