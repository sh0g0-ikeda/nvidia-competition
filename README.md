# Nemotron Baseline

[日本語版 README](./README.ja.md)

This repository now contains a minimal baseline for the Kaggle NVIDIA Nemotron reasoning competition:

- split `train.csv` into train/valid
- fine-tune a Nemotron base model with LoRA
- evaluate the adapter locally on exact match
- package the adapter as `submission.zip`

## Files

- `scripts/baseline_domain.py`: immutable config and data models
- `scripts/baseline_services.py`: repositories, runtime services, and pipelines
- `scripts/train_baseline.py`: thin CLI entry point for training
- `scripts/evaluate_adapter.py`: thin CLI entry point for evaluation
- `scripts/package_submission.py`: thin CLI entry point for packaging
- `requirements-baseline.txt`: minimal Python dependencies

## Quick Start

Install dependencies:

```powershell
& 'C:\Users\shogo\AppData\Local\Programs\Python\Python313\python.exe' -m pip install -r requirements-baseline.txt
```

Train a quick baseline on a subset:

```powershell
& 'C:\Users\shogo\AppData\Local\Programs\Python\Python313\python.exe' scripts/train_baseline.py `
  --train-csv train.csv `
  --run-dir runs\baseline_v1 `
  --model-ref metric/nemotron-3-nano-30b-a3b-bf16/transformers/default `
  --subsample-size 1200
```

Evaluate on the saved validation split:

```powershell
& 'C:\Users\shogo\AppData\Local\Programs\Python\Python313\python.exe' scripts/evaluate_adapter.py `
  --eval-csv runs\baseline_v1\splits\valid.csv `
  --adapter-dir runs\baseline_v1\adapter `
  --model-ref metric/nemotron-3-nano-30b-a3b-bf16/transformers/default
```

Create the submission archive:

```powershell
& 'C:\Users\shogo\AppData\Local\Programs\Python\Python313\python.exe' scripts/package_submission.py `
  --adapter-dir runs\baseline_v1\adapter `
  --zip-path runs\baseline_v1\submission.zip
```

## Notes

- The competition labels are short final answers, so this baseline trains `prompt -> answer` directly without synthetic reasoning traces.
- `model-ref` can be either a local path, a Hugging Face model id, or a Kaggle model ref if `kagglehub` is installed.
- The public Kaggle notebook pulled into `kaggle_artifacts/` was used as a reference for packaging and model choice.
- The code is split by responsibility so the CLI scripts stay small and the reusable logic lives in explicit services.
