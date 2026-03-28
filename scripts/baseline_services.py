from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from baseline_domain import DatasetSplit, EvalConfig, PackageConfig, PredictionRecord, PuzzleExample, TrainConfig

if TYPE_CHECKING:
    import torch


class PuzzleCsvRepository:
    def load(self, csv_path: str | Path) -> list[PuzzleExample]:
        path = Path(csv_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [
                PuzzleExample(
                    id=row["id"],
                    prompt=row["prompt"],
                    answer=row.get("answer") if row.get("answer", "") != "" else None,
                )
                for row in reader
            ]

    def save(self, csv_path: str | Path, rows: list[PuzzleExample]) -> None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        has_answer = any(row.has_answer for row in rows)
        fieldnames = ["id", "prompt"] + (["answer"] if has_answer else [])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                payload = {"id": row.id, "prompt": row.prompt}
                if has_answer:
                    payload["answer"] = row.answer or ""
                writer.writerow(payload)


class PredictionCsvWriter:
    def save(self, csv_path: str | Path, predictions: list[PredictionRecord]) -> None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["id", "prediction", "raw_generation"]
        if any(record.answer is not None for record in predictions):
            fieldnames.extend(["answer", "raw_exact", "normalized_exact"])

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in predictions:
                payload = {
                    "id": record.id,
                    "prediction": record.prediction,
                    "raw_generation": record.raw_generation,
                }
                if record.answer is not None:
                    payload["answer"] = record.answer
                    payload["raw_exact"] = str(record.raw_exact)
                    payload["normalized_exact"] = str(record.normalized_exact)
                writer.writerow(payload)


class JsonArtifactWriter:
    def save(self, path: str | Path, payload: dict) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


class DatasetSampler:
    def sample(self, rows: list[PuzzleExample], sample_size: int, seed: int) -> list[PuzzleExample]:
        if sample_size <= 0 or sample_size >= len(rows):
            return list(rows)
        return random.Random(seed).sample(rows, sample_size)


class DatasetSplitter:
    def split(self, rows: list[PuzzleExample], valid_size: float, seed: int) -> DatasetSplit:
        if valid_size <= 0:
            return DatasetSplit(train_rows=list(rows), valid_rows=[])

        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        valid_count = int(round(len(shuffled) * valid_size))
        valid_count = max(1, min(valid_count, len(shuffled) - 1))
        return DatasetSplit(
            train_rows=shuffled[valid_count:],
            valid_rows=shuffled[:valid_count],
        )


class AnswerNormalizer:
    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text or "")
        normalized = normalized.strip()
        return re.sub(r"\s+", " ", normalized)

    def raw(self, text: str) -> str:
        return (text or "").strip()


class FinalAnswerExtractor:
    def extract(self, generated_text: str) -> str:
        value = generated_text or ""
        value = value.replace("<|im_end|>", "")
        value = value.replace("</s>", "")
        value = value.replace("<eos>", "")
        value = value.strip()

        boxed_match = re.search(r"\\boxed\{([^{}]+)\}", value)
        if boxed_match:
            return boxed_match.group(1).strip()

        lines = [line.strip() for line in value.splitlines() if line.strip()]
        return lines[0] if lines else ""


class ChatTemplateRenderer:
    def render_training_text(self, tokenizer, example: PuzzleExample) -> str:
        if example.answer is None:
            raise ValueError("Training examples must include answers.")
        return self._render(tokenizer, prompt=example.prompt, answer=example.answer, add_generation_prompt=False)

    def render_generation_prompt(self, tokenizer, example: PuzzleExample) -> str:
        return self._render(tokenizer, prompt=example.prompt, answer=None, add_generation_prompt=True)

    def _render(self, tokenizer, prompt: str, answer: str | None, add_generation_prompt: bool) -> str:
        messages = [{"role": "user", "content": prompt}]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            rendered = f"<|im_start|>user\n{prompt}<|im_end|>\n"
            if answer is None:
                return rendered + ("<|im_start|>assistant\n" if add_generation_prompt else "")
            return rendered + f"<|im_start|>assistant\n{answer}<|im_end|>"


class ModelPathResolver:
    def resolve(self, model_ref: str) -> str:
        candidate = Path(model_ref)
        if candidate.exists():
            return str(candidate)

        try:
            import kagglehub
        except ImportError:
            return model_ref

        return kagglehub.model_download(model_ref)


class TorchRuntimePolicy:
    def select_dtype(self) -> "torch.dtype":
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def select_device_map(self):
        import torch

        return "auto" if torch.cuda.is_available() else None


class NemotronRuntimePatcher:
    def apply(self) -> None:
        # The Kaggle reference notebook disables the Nemotron fast path for stability.
        for module in list(__import__("sys").modules.values()):
            if module is None:
                continue
            if "modeling_nemotron_h" not in getattr(module, "__name__", ""):
                continue
            if hasattr(module, "is_fast_path_available"):
                module.is_fast_path_available = False


class NemotronFactory:
    def __init__(
        self,
        path_resolver: ModelPathResolver,
        runtime_policy: TorchRuntimePolicy,
        runtime_patcher: NemotronRuntimePatcher,
    ) -> None:
        self._path_resolver = path_resolver
        self._runtime_policy = runtime_policy
        self._runtime_patcher = runtime_patcher

    def resolve_model_path(self, model_ref: str) -> str:
        return self._path_resolver.resolve(model_ref)

    def selected_dtype(self) -> "torch.dtype":
        return self._runtime_policy.select_dtype()

    def load_tokenizer(self, reference: str):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(reference, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_trainable_model(self, model_path: str):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.selected_dtype(),
            device_map=self._runtime_policy.select_device_map(),
        )
        import torch

        self._runtime_patcher.apply()
        if torch.cuda.is_available():
            model.gradient_checkpointing_enable()
        return model

    def load_inference_model(self, model_path: str, adapter_dir: str | Path):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.selected_dtype(),
            device_map=self._runtime_policy.select_device_map(),
        )
        self._runtime_patcher.apply()
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model.eval()
        return model

    def build_lora_config(self, train_config: TrainConfig):
        from peft import LoraConfig, TaskType

        return LoraConfig(
            r=train_config.lora_rank,
            lora_alpha=train_config.lora_alpha,
            target_modules="all-linear",
            lora_dropout=train_config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )


class TrainingDatasetBuilder:
    def __init__(self, renderer: ChatTemplateRenderer) -> None:
        self._renderer = renderer

    def build(self, rows: list[PuzzleExample], tokenizer):
        from datasets import Dataset

        return Dataset.from_list(
            [{"text": self._renderer.render_training_text(tokenizer, row)} for row in rows]
        )


class BaselineTrainingPipeline:
    def __init__(
        self,
        repository: PuzzleCsvRepository,
        sampler: DatasetSampler,
        splitter: DatasetSplitter,
        factory: NemotronFactory,
        dataset_builder: TrainingDatasetBuilder,
        artifacts: JsonArtifactWriter,
    ) -> None:
        self._repository = repository
        self._sampler = sampler
        self._splitter = splitter
        self._factory = factory
        self._dataset_builder = dataset_builder
        self._artifacts = artifacts

    def run(self, config: TrainConfig) -> None:
        config.run_path.mkdir(parents=True, exist_ok=True)
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        config.splits_dir.mkdir(parents=True, exist_ok=True)

        rows = self._repository.load(config.train_csv)
        sampled_rows = self._sampler.sample(rows, config.subsample_size, config.seed)
        split = self._splitter.split(sampled_rows, config.valid_size, config.seed)
        self._repository.save(config.splits_dir / "train.csv", split.train_rows)
        if split.valid_rows:
            self._repository.save(config.splits_dir / "valid.csv", split.valid_rows)

        model_path = self._factory.resolve_model_path(config.model_ref)
        tokenizer = self._factory.load_tokenizer(model_path)
        train_dataset = self._dataset_builder.build(split.train_rows, tokenizer)

        model = self._factory.load_trainable_model(model_path)
        from peft import get_peft_model

        model = get_peft_model(model, self._factory.build_lora_config(config))

        training_args = self._build_training_args(config)
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            args=training_args,
        )
        trainer.train()
        trainer.model.save_pretrained(config.adapter_dir)
        tokenizer.save_pretrained(config.adapter_dir)

        metadata = asdict(config)
        metadata.update(
            {
                "resolved_model_path": model_path,
                "num_total_rows": len(rows),
                "num_sampled_rows": len(sampled_rows),
                "num_train_rows": len(split.train_rows),
                "num_valid_rows": len(split.valid_rows),
                "torch_dtype": str(self._factory.selected_dtype()),
            }
        )
        self._artifacts.save(config.run_config_path, metadata)

    def _build_training_args(self, config: TrainConfig):
        from trl import SFTConfig
        import torch

        dtype = self._factory.selected_dtype()
        return SFTConfig(
            output_dir=str(config.adapter_dir),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            logging_steps=10,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            max_grad_norm=1.0,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=config.warmup_ratio,
            save_strategy="no",
            report_to="none",
            dataset_text_field="text",
            max_length=config.max_seq_len,
            packing=False,
            gradient_checkpointing=torch.cuda.is_available(),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_steps=config.max_steps,
        )


class AdapterEvaluationPipeline:
    def __init__(
        self,
        repository: PuzzleCsvRepository,
        writer: PredictionCsvWriter,
        factory: NemotronFactory,
        renderer: ChatTemplateRenderer,
        extractor: FinalAnswerExtractor,
        normalizer: AnswerNormalizer,
        artifacts: JsonArtifactWriter,
    ) -> None:
        self._repository = repository
        self._writer = writer
        self._factory = factory
        self._renderer = renderer
        self._extractor = extractor
        self._normalizer = normalizer
        self._artifacts = artifacts

    def run(self, config: EvalConfig) -> None:
        import torch

        rows = self._repository.load(config.eval_csv)
        model_path = self._factory.resolve_model_path(config.model_ref)
        tokenizer = self._factory.load_tokenizer(config.adapter_dir)
        model = self._factory.load_inference_model(model_path, config.adapter_dir)

        predictions: list[PredictionRecord] = []
        raw_matches = 0
        normalized_matches = 0
        answered_rows = [row for row in rows if row.answer is not None]

        for index, row in enumerate(rows, start=1):
            prompt_text = self._renderer.render_generation_prompt(tokenizer, row)
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()
            prediction = self._extractor.extract(decoded)
            record = PredictionRecord(
                id=row.id,
                prediction=prediction,
                raw_generation=decoded,
            )

            if row.answer is not None:
                raw_exact = self._normalizer.raw(prediction) == self._normalizer.raw(row.answer)
                normalized_exact = self._normalizer.normalize(prediction) == self._normalizer.normalize(row.answer)
                raw_matches += int(raw_exact)
                normalized_matches += int(normalized_exact)
                record = PredictionRecord(
                    id=row.id,
                    prediction=prediction,
                    raw_generation=decoded,
                    answer=row.answer,
                    raw_exact=raw_exact,
                    normalized_exact=normalized_exact,
                )

            predictions.append(record)
            if index % 50 == 0:
                print(f"Evaluated {index}/{len(rows)} rows")

        self._writer.save(config.predictions_path, predictions)
        if answered_rows:
            self._artifacts.save(
                config.metrics_path,
                {
                    "num_rows": len(answered_rows),
                    "raw_exact_accuracy": raw_matches / len(answered_rows),
                    "normalized_exact_accuracy": normalized_matches / len(answered_rows),
                    "eval_csv": config.eval_csv,
                    "adapter_dir": config.adapter_dir,
                },
            )


class AdapterPackager:
    REQUIRED_FILES = {"adapter_config.json", "adapter_model.safetensors"}

    def package(self, config: PackageConfig) -> list[str]:
        if not config.adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory does not exist: {config.adapter_path}")

        file_paths = [path for path in config.adapter_path.iterdir() if path.is_file()]
        file_names = {path.name for path in file_paths}
        missing = self.REQUIRED_FILES - file_names
        if missing:
            raise FileNotFoundError(
                f"Adapter directory is missing required files: {', '.join(sorted(missing))}"
            )

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(config.output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(file_paths):
                archive.write(file_path, arcname=file_path.name)

        return sorted(file_names)
