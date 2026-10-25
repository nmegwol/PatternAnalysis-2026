"""Manual FLAN-T5 training and ROUGE evaluation; COMP3710, student 46375489."""

import argparse
import gc
import json
import math
import re
import time
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import torch
from rouge_score.rouge_scorer import RougeScorer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import (
    DATASET_REVISION,
    DEFAULT_PROMPT,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    audit_overlap,
    load_splits,
    make_dataloader,
    sample_records,
    tokenize_split,
)
from modules import build_model, parameter_counts
from utils import (
    generate_summaries,
    load_checkpoint,
    setup_runtime,
    synchronize,
    write_json,
)

MODEL_REVISIONS = {
    "google/flan-t5-small": "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
    "google/flan-t5-base": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
}
ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    accumulation_steps=4,
    max_steps=None,
    bf16=False,
):
    """Run explicit token-weighted updates, including an incomplete last window."""
    if accumulation_steps < 1 or not len(loader):
        raise ValueError("Need a nonempty loader and positive accumulation")
    model.train()
    old_cache = model.config.use_cache
    model.config.use_cache = False
    parameters = [p for p in model.parameters() if p.requires_grad]
    history = []
    iterator = iter(loader)
    try:
        while max_steps is None or len(history) < max_steps:
            window = list(islice(iterator, accumulation_steps))
            if not window:
                break
            counts = [int((batch["labels"] != -100).sum()) for batch in window]
            if min(counts) == 0:
                raise ValueError("Training batch has no active target tokens")
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for batch, count in zip(window, counts, strict=True):
                inputs = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=bf16
                ):
                    loss = model(**inputs).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss * count / sum(counts)).backward()
                loss_sum += float(loss.detach()) * count
            torch.nn.utils.clip_grad_norm_(
                parameters, 1.0, error_if_nonfinite=True
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            history.append(
                {
                    "loss": loss_sum / sum(counts),
                    "target_tokens": sum(counts),
                    "examples": sum(len(b["input_ids"]) for b in window),
                    "microbatches": len(window),
                    "learning_rate": learning_rate,
                }
            )
            if len(history) == 1 or len(history) % 100 == 0:
                print(
                    f"Update {len(history)}: loss {history[-1]['loss']:.4f}",
                    flush=True,
                )
    finally:
        model.config.use_cache = old_cache
    return history


@torch.inference_mode()
def validation_loss(model, loader, device, bf16=False):
    """Mean loss per nonpadding target token, not the mean of batch means."""
    was_training = model.training
    model.eval()
    total = tokens = 0
    try:
        for batch in loader:
            count = int((batch["labels"] != -100).sum())
            if not count:
                raise ValueError("Validation batch has no active target tokens")
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=bf16
            ):
                loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss")
            total += float(loss) * count
            tokens += count
        if not tokens:
            raise ValueError("Validation loader is empty")
        return total / tokens
    finally:
        model.train(was_training)


def score_predictions(rows):
    """Score original references; Porter stemming and punctuation/newline Lsum.

    Missing references are unscored. Empty predictions score zero. Each metric
    is an arithmetic mean of per-example F1 on a 0--100 scale.
    """
    scorer = RougeScorer(ROUGE_KEYS, use_stemmer=True)
    scores = []
    for row in rows:
        reference = row.get("reference")
        row["rouge"] = None
        if not isinstance(reference, str) or not reference.strip():
            continue
        texts = [
            "\n".join(re.split(r"(?<=[.!?])\s+|\n+", t.strip()))
            for t in (reference, row["prediction"])
        ]
        values = scorer.score(*texts)
        row["rouge"] = {key: 100 * values[key].fmeasure for key in ROUGE_KEYS}
        scores.append(row["rouge"])
    return {
        "examples": len(rows),
        "scored_examples": len(scores),
        "rouge": {
            key: sum(s[key] for s in scores) / len(scores) for key in ROUGE_KEYS
        }
        if scores
        else None,
    }


def evaluate(
    model,
    tokenizer,
    records,
    features,
    device,
    batch_size=2,
    bf16=False,
    unseen_ids=None,
):
    """Generate in record order and retain full source/reference comparisons."""
    loader = make_dataloader(features, tokenizer, batch_size)
    predictions, lengths, profile = generate_summaries(
        model, tokenizer, loader, device, bf16
    )
    rows = [
        {
            "split": row["split"],
            "row_index": row["row_index"],
            "source": row[SOURCE_COLUMN],
            "reference": row.get(TARGET_COLUMN),
            "prediction": prediction,
            "generated_tokens": length,
        }
        for row, prediction, length in zip(
            records, predictions, lengths, strict=True
        )
    ]
    metrics = score_predictions(rows)
    metrics["inference"] = profile
    metrics["loss"] = (
        validation_loss(model, loader, device, bf16)
        if "labels" in features.column_names
        else None
    )
    metrics["unseen"] = (
        score_predictions([r for r in rows if r["row_index"] in unseen_ids])
        if unseen_ids is not None
        else None
    )
    return metrics, rows


def save_checkpoint(model, tokenizer, path):
    """Use native HF/PEFT files and generation configuration, with no manifest."""
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    model.generation_config.save_pretrained(path)


def save_predictions(path, rows):
    """Save only local JSONL evidence for the selected model."""
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_history(history, epochs, output):
    """Plot measured update loss, epoch validation loss and validation ROUGE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        range(1, len(history) + 1),
        [r["loss"] for r in history],
        ".-",
        label="Training",
    )
    steps = [r["step"] for r in epochs]
    axes[0].plot(steps, [r["loss"] for r in epochs], "o-", label="Validation")
    for key in ROUGE_KEYS:
        axes[1].plot(steps, [r["rouge"][key] for r in epochs], "o-", label=key)
    for ax in axes:
        ax.set_xlabel("Optimiser update")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Loss per target token")
    axes[1].set_ylabel("ROUGE F1 (%)")
    fig.tight_layout()
    fig.savefig(output / "training_metrics.png", dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    """One flat CLI; zero subset limits explicitly select all official rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("baseline", "lora", "full"), default="baseline"
    )
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument(
        "--revision",
        help="Immutable Hub commit; small/base defaults are pinned",
    )
    parser.add_argument(
        "--data-dir",
        help="Local official Parquet directory; omit for pinned Hub data",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-limit", type=int, default=16, help="0 = all")
    parser.add_argument(
        "--validation-limit", type=int, default=8, help="0 = all"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="CUDA BF16 autocast; master weights stay FP32",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit full train/validation once for this run, including unseen-view scoring",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate test only after model selection",
    )
    args = parser.parse_args(argv)
    for name in (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "accumulation_steps",
        "max_new_tokens",
        "num_beams",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if (
        min(args.train_limit, args.validation_limit, args.seed) < 0
        or args.seed >= 2**32
    ):
        parser.error("limits must be >=0 and seed in [0, 2**32)")
    if min(args.max_input_tokens, args.max_target_tokens) < 2:
        parser.error("Input/target limits must be at least 2 to retain EOS")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    args.learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else (0.0003 if args.mode == "lora" else 0.00005)
    )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if Path(args.model).is_dir():
        args.model, args.revision = str(Path(args.model).resolve()), None
    else:
        args.revision = args.revision or MODEL_REVISIONS.get(args.model)
        if not args.revision or not re.fullmatch(
            r"[0-9a-f]{40}", args.revision
        ):
            parser.error("Supply --revision COMMIT for other Hub models")
    return args


def main(argv=None):
    """Load data, run explicit train/evaluate epochs, and save selected weights."""
    args = parse_args(argv)
    device = setup_runtime(args.device, args.seed, args.bf16)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        **vars(args),
        "dataset_revision": DATASET_REVISION if args.data_dir is None else None,
        "actual_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "prompt": DEFAULT_PROMPT,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "versions": {
            p: version(p)
            for p in (
                "torch",
                "transformers",
                "peft",
                "datasets",
                "rouge_score",
            )
        },
    }
    write_json(output / "settings.json", settings)
    splits = load_splits(args.data_dir)
    unseen_ids = None
    if args.audit:
        audit = audit_overlap(splits["train"], splits["validation"])
        unseen_ids = set(audit["unseen_validation_row_indices"])
        write_json(output / "overlap.json", audit)
    train_rows = sample_records(splits["train"], args.train_limit, args.seed)
    validation = sample_records(
        splits["validation"], args.validation_limit, args.seed
    )
    # Cheap run-scoped checks never masquerade as a full-dataset audit.
    if not args.audit:
        audit = audit_overlap(train_rows, validation)
        write_json(
            output / "overlap.json",
            {"scope": "selected rows only; not a global unseen audit", **audit},
        )
    model = build_model(args.model, args.mode, args.revision).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, token=False
    )
    tokenizer.model_max_length = args.max_input_tokens
    model.generation_config.max_new_tokens = args.max_new_tokens
    model.generation_config.num_beams = args.num_beams
    model.generation_config.do_sample = False
    model.generation_config.length_penalty = 1.0
    model.generation_config.early_stopping = args.num_beams > 1
    counts = parameter_counts(model)
    val_features, val_truncation = tokenize_split(
        validation, tokenizer, args.max_input_tokens, args.max_target_tokens
    )
    history, epochs = [], []
    seconds = 0.0
    peak = None
    best_loss = math.inf
    best_epoch = None
    if args.mode == "baseline":
        metrics, rows = evaluate(
            model,
            tokenizer,
            validation,
            val_features,
            device,
            args.eval_batch_size,
            args.bf16,
            unseen_ids,
        )
        save_predictions(output / "validation.jsonl", rows)
    else:
        features, train_truncation = tokenize_split(
            train_rows, tokenizer, args.max_input_tokens, args.max_target_tokens
        )
        loader = make_dataloader(
            features, tokenizer, args.batch_size, shuffle=True, seed=args.seed
        )
        budget = math.ceil(len(loader) / args.accumulation_steps) * args.epochs
        budget = min(budget, args.max_steps) if args.max_steps else budget
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=0.01,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(0.1 * budget), budget
        )
        for epoch in range(1, args.epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            started = time.perf_counter()
            updates = train_epoch(
                model,
                loader,
                optimizer,
                scheduler,
                device,
                args.accumulation_steps,
                budget - len(history),
                args.bf16,
            )
            synchronize(device)
            seconds += time.perf_counter() - started
            if device.type == "cuda":
                peak = max(
                    peak or 0, torch.cuda.max_memory_allocated(device) / 2**30
                )
            history.extend(updates)
            write_json(output / "training_history.json", history)
            current, rows = evaluate(
                model,
                tokenizer,
                validation,
                val_features,
                device,
                args.eval_batch_size,
                args.bf16,
                unseen_ids,
            )
            epochs.append({"epoch": epoch, "step": len(history), **current})
            write_json(output / "validation_history.json", epochs)
            if current["loss"] < best_loss:
                best_loss, best_epoch, metrics = current["loss"], epoch, current
                save_checkpoint(model, tokenizer, output / "best")
                save_predictions(output / "validation.jsonl", rows)
            plot_history(history, epochs, output)
            if len(history) >= budget:
                break
        save_checkpoint(model, tokenizer, output / "last")
        del optimizer, scheduler
    summary = {
        "mode": args.mode,
        **counts,
        "training_seconds": seconds,
        "training_peak_gpu_gib": peak,
        "optimizer_steps": len(history),
        "examples_seen": sum(r["examples"] for r in history),
        "best_epoch": best_epoch,
        "validation": metrics,
        "validation_row_indices": list(validation["row_index"]),
        "train_row_indices": list(train_rows["row_index"])
        if args.mode != "baseline"
        else [],
        "validation_truncation": val_truncation,
        "training_truncation": train_truncation
        if args.mode != "baseline"
        else None,
        "test": None,
    }
    if args.evaluate_test:
        if args.mode != "baseline":
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model, tokenizer = load_checkpoint(output / "best", device)
        test = sample_records(
            load_splits(args.data_dir, splits=("test",))["test"],
            args.validation_limit,
            args.seed,
        )
        references = TARGET_COLUMN in test.column_names and all(
            isinstance(t, str) and t.strip() for t in test[TARGET_COLUMN]
        )
        features, truncation = tokenize_split(
            test,
            tokenizer,
            args.max_input_tokens,
            args.max_target_tokens,
            include_labels=references,
        )
        summary["test"], rows = evaluate(
            model,
            tokenizer,
            test,
            features,
            device,
            args.eval_batch_size,
            args.bf16,
        )
        summary["test"]["truncation"] = truncation
        save_predictions(output / "test.jsonl", rows)
    write_json(output / "metrics.json", summary)
    print(f"Saved metrics, predictions and any checkpoints to {output}")


if __name__ == "__main__":
    main()
