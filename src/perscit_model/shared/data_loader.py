import json
from pathlib import Path
from typing import Generator

import transformers
from transformers import AutoTokenizer

DEFAULT_CONFIG = (
    Path(__file__).parent.parent.parent.parent / "configs/extraction/baseline.yaml"
)


class SharedDataLoader:
    """Base data loader with shared utilities for tokenization and JSONL loading."""

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int | None = None,
        config_path: str | Path | None = None,
        special_tokens: str | list[str] | None = None,
        use_fast: bool = True,
    ):
        model_name_from_config, max_length_from_config = self.from_yaml(config_path)
        self.model_name = model_name if model_name else model_name_from_config
        self.max_length = max_length if max_length else max_length_from_config
        self._tokenizer = None
        if isinstance(special_tokens, str):
            special_tokens = [special_tokens]
        self.special_tokens = special_tokens if special_tokens is not None else []
        self._use_fast = use_fast

    @staticmethod
    def from_yaml(config_path: Path | str | None = None) -> tuple[str, int]:
        """
        Load model_name and max_length from YAML config file.

        Args:
            config_path: Path to YAML config file (uses default if None)

        Returns:
            Tuple of (model_name, max_length)
        """
        from perscit_model.shared.training_utils import TrainingConfig

        if config_path is None:
            config_path = DEFAULT_CONFIG
        config = TrainingConfig.from_yaml(config_path)
        return config.model_name, config.max_length

    def load_jsonl(self, filepath: Path | str) -> Generator[dict, None, None]:
        """
        Load JSONL file line by line (memory efficient generator).

        Args:
            filepath: Path to JSONL file

        Yields:
            Parsed JSON objects
        """
        with open(filepath) as f:
            for line in f:
                yield json.loads(line)

    def add_special_tokens(self, new_tokens: str | list):
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        self.special_tokens.extend(new_tokens)

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        """
        Load tokenizer for the specified model.

        Args:
            model_name: HuggingFace model name (e.g., "microsoft/deberta-v3-base")

        Returns:
            Pretrained tokenizer for the model
        """
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=self.use_fast
            )
            if self.special_tokens:
                self._tokenizer.add_special_tokens(
                    {"additional_special_tokens": self.special_tokens}
                )
        return self._tokenizer

    @property
    def use_fast(self) -> bool:
        return self._use_fast

    @use_fast.setter
    def use_fast(self, new_setting: bool) -> None:
        if self._tokenizer is None:
            print(
                f"tokenizer not yet created: when it is, parameter use_fast = {self._use_fast}"
            )
            self._use_fast = new_setting
        else:
            print(
                f"tokenizer already created with use_fast = {self._use_fast}, which has NOT been changed"
            )

    def tokenize_text(
        self,
        text: str,
        max_length: int | None = None,
        **kwargs,
    ) -> transformers.BatchEncoding:
        """
        Tokenize text with the given tokenizer.

        Args:
            text: Text to tokenize
            max_length: Maximum sequence length (uses config value if not specified)

        Returns:
            Tokenized output with tensors
        """
        if max_length is None:
            max_length = self.max_length

        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            **kwargs,
        )
