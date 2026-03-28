from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PuzzleExample:
    id: str
    prompt: str
    answer: str | None = None

    @property
    def has_answer(self) -> bool:
        return self.answer is not None


@dataclass(frozen=True)
class DatasetSplit:
    train_rows: list[PuzzleExample]
    valid_rows: list[PuzzleExample]


@dataclass(frozen=True)
class TrainConfig:
    train_csv: str
    run_dir: str
    model_ref: str
    valid_size: float
    seed: int
    subsample_size: int
    max_seq_len: int
    num_epochs: float
    learning_rate: float
    batch_size: int
    grad_accum: int
    warmup_ratio: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    max_steps: int

    @property
    def run_path(self) -> Path:
        return Path(self.run_dir)

    @property
    def adapter_dir(self) -> Path:
        return self.run_path / "adapter"

    @property
    def splits_dir(self) -> Path:
        return self.run_path / "splits"

    @property
    def run_config_path(self) -> Path:
        return self.run_path / "run_config.json"


@dataclass(frozen=True)
class EvalConfig:
    eval_csv: str
    adapter_dir: str
    model_ref: str
    output_csv: str
    max_new_tokens: int

    @property
    def adapter_path(self) -> Path:
        return Path(self.adapter_dir)

    @property
    def predictions_path(self) -> Path:
        if self.output_csv:
            return Path(self.output_csv)
        return self.adapter_path.parent / "predictions.csv"

    @property
    def metrics_path(self) -> Path:
        return self.adapter_path.parent / "eval_metrics.json"


@dataclass(frozen=True)
class PackageConfig:
    adapter_dir: str
    zip_path: str

    @property
    def adapter_path(self) -> Path:
        return Path(self.adapter_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.zip_path)


@dataclass(frozen=True)
class PredictionRecord:
    id: str
    prediction: str
    raw_generation: str
    answer: str | None = None
    raw_exact: bool | None = None
    normalized_exact: bool | None = None
