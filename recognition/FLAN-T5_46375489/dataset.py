"""Dataset loading, preprocessing and batching for radiology report simplification."""

from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding


class RadiologyData:
    """Manage the official LaymanRRG splits without reshuffling or deduplicating."""

    DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
    SOURCE_COLUMN = "radiology_report"
    TARGET_COLUMN = "layman_report"

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def load(self):
        """Load and tokenise official splits; smoke mode selects train/validation prefixes."""
        from datasets import load_dataset

        self._raw_splits = load_dataset(self.config["dataset_id"], revision=self.config.get("revision"))
        if self.config["smoke"]:
            # Select before tokenisation; never process test rows here.
            self._raw_splits = DatasetDict({
                split: self._raw_splits[split].select(range(count))
                for split, count in (("train", 16), ("validation", 8))
            })
        elif any(self.config.get(key) is not None for key in ("max_train_examples", "max_validation_examples")):
            self._raw_splits = DatasetDict({
                split: self._raw_splits[split].select(range(min(count, len(self._raw_splits[split]))))
                if count is not None else self._raw_splits[split]
                for split, count in (
                    ("train", self.config.get("max_train_examples")),
                    ("validation", self.config.get("max_validation_examples")),
                )
            })

        self._tokenised_splits = DatasetDict()
        for split, rows in self._raw_splits.items():
            # The fixed test split contains only withheld (empty) references.
            if split == "test":
                rows = rows.remove_columns(self.TARGET_COLUMN)
            self._tokenised_splits[split] = rows.map(
                self._tokenise_batch,
                batched=True,
                remove_columns=rows.column_names,
                desc=f"Tokenising {split}",
            )

    def _tokenise_batch(self, examples):
        """Tokenise inputs and targets, including truncation and label masking."""
        encoded = self.tokenizer(
            [self.config["prompt_template"].format(report=report) for report in examples[self.SOURCE_COLUMN]],
            max_length=self.config["max_input_length"],
            truncation=True,
            padding=False,
        )
        # Train/validation have targets; test input batches intentionally omit them.
        if self.TARGET_COLUMN in examples:
            targets = self.tokenizer(
                text_target=examples[self.TARGET_COLUMN],
                max_length=self.config["max_target_length"],
                truncation=True,
                padding=False,
            )
            encoded["labels"] = targets["input_ids"]
        # Label padding is introduced and masked with -100 by the collator.
        return dict(encoded)

    def dataloader(self, split, shuffle=None):
        """Pad batches and shuffle only training rows by default."""
        if split == "test":
            collator = DataCollatorWithPadding(self.tokenizer, return_tensors="pt")
        else:
            collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, padding=True, label_pad_token_id=-100,
                return_tensors="pt",
            )

        return DataLoader(
            self._tokenised_splits[split],
            batch_size=(self.config["batch_size"] if split == "train" else
                        self.config.get("eval_batch_size") or self.config["batch_size"]),
            shuffle=(split == "train") if shuffle is None else shuffle,
            collate_fn=collator,
        )
