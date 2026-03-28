from __future__ import annotations

import argparse
import os

from baseline_domain import TrainConfig
from baseline_services import (
    BaselineTrainingPipeline,
    ChatTemplateRenderer,
    DatasetSampler,
    DatasetSplitter,
    JsonArtifactWriter,
    ModelPathResolver,
    NemotronFactory,
    NemotronRuntimePatcher,
    PuzzleCsvRepository,
    TorchRuntimePolicy,
    TrainingDatasetBuilder,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_MODEL_REF = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a Kaggle-ready Nemotron baseline LoRA adapter.")
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--run-dir", default="runs/baseline_v1")
    parser.add_argument("--model-ref", default=DEFAULT_MODEL_REF)
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample-size", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--num-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    return TrainConfig(
        train_csv=args.train_csv,
        run_dir=args.run_dir,
        model_ref=args.model_ref,
        valid_size=args.valid_size,
        seed=args.seed,
        subsample_size=args.subsample_size,
        max_seq_len=args.max_seq_len,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_steps=args.max_steps,
    )


def build_pipeline() -> BaselineTrainingPipeline:
    return BaselineTrainingPipeline(
        repository=PuzzleCsvRepository(),
        sampler=DatasetSampler(),
        splitter=DatasetSplitter(),
        factory=NemotronFactory(
            path_resolver=ModelPathResolver(),
            runtime_policy=TorchRuntimePolicy(),
            runtime_patcher=NemotronRuntimePatcher(),
        ),
        dataset_builder=TrainingDatasetBuilder(ChatTemplateRenderer()),
        artifacts=JsonArtifactWriter(),
    )


def main() -> None:
    config = parse_args()
    pipeline = build_pipeline()
    pipeline.run(config)
    print(f"Adapter saved to: {config.adapter_dir}")
    print(f"Train split: {config.splits_dir / 'train.csv'}")
    if (config.splits_dir / "valid.csv").exists():
        print(f"Valid split: {config.splits_dir / 'valid.csv'}")


if __name__ == "__main__":
    main()
