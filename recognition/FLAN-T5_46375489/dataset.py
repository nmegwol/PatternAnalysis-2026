"""Load and batch LaymanRRG radiology reports for summarisation."""

from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding


class RadiologyData:
    """Prepares official LaymanRRG splits and creates padded batches.

    Call load() before requesting a dataloader.

    Attributes:
        tokenizer: Hugging Face tokenizer shared by all splits.
        config: Dict of dataset, tokenisation and batching settings.
    """

    DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
    SOURCE_COLUMN = "radiology_report"
    TARGET_COLUMN = "layman_report"

    def __init__(self, tokenizer, config):
        """Stores the tokenizer and configuration without loading data.

        Args:
            tokenizer: A Hugging Face PreTrainedTokenizerBase instance.
            config: Dict containing dataset_id, smoke, batch_size,
                max_input_length, max_target_length and prompt_template
                (with a {report} placeholder). Optional keys are revision,
                eval_batch_size, max_train_examples and max_validation_examples.
        """
        self.tokenizer = tokenizer
        self.config = config

    def load(self):
        """Loads dataset rows and stores their tokenised forms.

        Smoke mode selects the first 16 training and 8 validation rows.
        Optional example limits also select prefixes of those splits. Both
        subset modes omit test data; full loading excludes test references from
        tokenisation. Split membership and row order are preserved.

        Stores selected rows in _raw_splits and tokens in _tokenised_splits.
        """
        from datasets import load_dataset

        self._raw_splits = load_dataset(
            self.config["dataset_id"],
            revision=self.config.get("revision")
        )

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
        """Tokenises and truncates prompted reports and optional targets.

        Args:
            examples: Dict mapping column names to lists of strings. Requires
                radiology_report; layman_report is optional.

        Returns:
            A dict of unpadded input IDs and attention masks, plus labels when
            targets are present. Padding and -100 label masking happen later
            in the dataloader's collator.
        """
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
        """Creates a padded DataLoader for a previously loaded split.

        Uses batch_size for training. Other splits use eval_batch_size when
        provided, falling back to batch_size.
        Label padding uses -100; test batches contain no labels.

        Args:
            split: Split name (str): "train", "validation" or "test".
            shuffle: Whether to shuffle rows (bool). None shuffles only train.

        Returns:
            A torch.utils.data.DataLoader yielding batches of PyTorch tensors.

        Raises:
            KeyError: If the requested split was not loaded.
        """
        if split == "test":
            collator = DataCollatorWithPadding(
                self.tokenizer,
                return_tensors="pt"
            )
        else:
            collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True,
                label_pad_token_id=-100,
                return_tensors="pt",
            )

        return DataLoader(
            self._tokenised_splits[split],
            batch_size=(self.config["batch_size"] if split == "train" else
                        self.config.get("eval_batch_size") or self.config["batch_size"]),
            shuffle=(split == "train") if shuffle is None else shuffle,
            collate_fn=collator,
        )
