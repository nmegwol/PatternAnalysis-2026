"""Reload native full/LoRA checkpoints and print comparisons; student 46375489."""

import argparse
import json
from pathlib import Path

from datasets import Dataset

from dataset import (
    SOURCE_COLUMN,
    TARGET_COLUMN,
    load_splits,
    make_dataloader,
    sample_records,
    tokenize_split,
)
from utils import generate_summaries, load_checkpoint, setup_runtime


def main(argv=None):
    """Run source-only inference from one report or a seeded official split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument(
        "--input-file", type=Path, help="UTF-8 file containing one report"
    )
    source.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--data-dir")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Optional new JSONL output file"
    )
    args = parser.parse_args(argv)
    if args.limit < 1 or args.batch_size < 1:
        parser.error("limit and batch-size must be positive")
    if args.output and args.output.exists():
        parser.error("Output exists; choose a new file")
    if args.split:
        records = sample_records(
            load_splits(args.data_dir, splits=(args.split,))[args.split],
            args.limit,
            args.seed,
        )
    else:
        text = (
            args.input_file.read_text(encoding="utf-8")
            if args.input_file
            else args.text
        )
        if not text.strip():
            parser.error("Report must not be blank")
        records = Dataset.from_dict({SOURCE_COLUMN: [text]})
    device = setup_runtime(args.device, args.seed, args.bf16)
    model, tokenizer = load_checkpoint(args.checkpoint, device)
    # save_pretrained retains model_max_length; external tokenizers may use
    # an unlimited sentinel, for which the project's 512-token default applies.
    limit = (
        tokenizer.model_max_length
        if tokenizer.model_max_length < 1000000
        else 512
    )
    features, truncation = tokenize_split(
        records, tokenizer, max_input_tokens=limit, include_labels=False
    )
    predictions, lengths, profile = generate_summaries(
        model,
        tokenizer,
        make_dataloader(features, tokenizer, args.batch_size),
        device,
        args.bf16,
    )
    rows = []
    for i, (row, prediction, length) in enumerate(
        zip(records, predictions, lengths, strict=True)
    ):
        rows.append(
            {
                "split": row.get("split", "illustrative"),
                "row_index": row.get("row_index", i),
                "source": row[SOURCE_COLUMN],
                "reference": row.get(TARGET_COLUMN),
                "prediction": prediction,
                "generated_tokens": length,
            }
        )
        print(
            f"\n{rows[-1]['split']} row {rows[-1]['row_index']}\n"
            f"SOURCE: {row[SOURCE_COLUMN]}\nPREDICTION: {prediction}\n"
            f"REFERENCE: {row.get(TARGET_COLUMN) or '(unavailable)'}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps({"truncation": truncation, "inference": profile}, indent=2)
    )


if __name__ == "__main__":
    main()
