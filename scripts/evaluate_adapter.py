from __future__ import annotations

import argparse

from baseline_domain import EvalConfig
from baseline_services import (
    AdapterEvaluationPipeline,
    AnswerNormalizer,
    ChatTemplateRenderer,
    FinalAnswerExtractor,
    JsonArtifactWriter,
    ModelPathResolver,
    NemotronFactory,
    NemotronRuntimePatcher,
    PredictionCsvWriter,
    PuzzleCsvRepository,
    TorchRuntimePolicy,
)

DEFAULT_MODEL_REF = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate a trained Nemotron LoRA adapter.")
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model-ref", default=DEFAULT_MODEL_REF)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    return EvalConfig(
        eval_csv=args.eval_csv,
        adapter_dir=args.adapter_dir,
        model_ref=args.model_ref,
        output_csv=args.output_csv,
        max_new_tokens=args.max_new_tokens,
    )


def build_pipeline() -> AdapterEvaluationPipeline:
    return AdapterEvaluationPipeline(
        repository=PuzzleCsvRepository(),
        writer=PredictionCsvWriter(),
        factory=NemotronFactory(
            path_resolver=ModelPathResolver(),
            runtime_policy=TorchRuntimePolicy(),
            runtime_patcher=NemotronRuntimePatcher(),
        ),
        renderer=ChatTemplateRenderer(),
        extractor=FinalAnswerExtractor(),
        normalizer=AnswerNormalizer(),
        artifacts=JsonArtifactWriter(),
    )


def main() -> None:
    config = parse_args()
    pipeline = build_pipeline()
    pipeline.run(config)
    print(f"Predictions written to: {config.predictions_path}")
    if config.metrics_path.exists():
        print(f"Metrics written to: {config.metrics_path}")


if __name__ == "__main__":
    main()
