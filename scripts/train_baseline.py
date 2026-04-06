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

# RunPodでは HuggingFace のモデルIDを直接指定する
# Kaggle環境では "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default" を使う
DEFAULT_MODEL_REF = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a Kaggle-ready Nemotron baseline LoRA adapter.")
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--run-dir", default="runs/baseline_v1")
    parser.add_argument("--model-ref", default=DEFAULT_MODEL_REF)
    parser.add_argument("--valid-size", type=float, default=0.0)  # 全データを学習に使う
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample-size", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=2048)  # コンペは8192だが学習時はVRAM節約
    parser.add_argument("--num-epochs", type=float, default=1.0)  # 過学習防止のため1エポック
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)  # 実効バッチサイズ=8
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)  # alpha=2*rankが安定
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
