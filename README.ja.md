# Nemotron ベースライン

[English README](./README.md)

このリポジトリには、Kaggle の NVIDIA Nemotron reasoning competition 向けの最小ベースラインが入っています。

- `train.csv` を train / valid に分割
- Nemotron ベースモデルに対して LoRA を学習
- exact match でローカル評価
- `submission.zip` を作成

## ファイル構成

- `scripts/baseline_domain.py`: 不変な設定モデルとデータモデル
- `scripts/baseline_services.py`: リポジトリ、ランタイムサービス、各種パイプライン
- `scripts/train_baseline.py`: 学習用の薄い CLI エントリポイント
- `scripts/evaluate_adapter.py`: 評価用の薄い CLI エントリポイント
- `scripts/package_submission.py`: `submission.zip` 作成用の薄い CLI エントリポイント
- `requirements-baseline.txt`: 最小依存

## クイックスタート

依存を入れる:

```powershell
& 'C:\Users\shogo\AppData\Local\Programs\Python\Python313\python.exe' -m pip install -r requirements-baseline.txt
```

まずはサブセットでベースライン学習:

```powershell
& 'C:\Users\shogo\AppData\Local\Programs\Python\Python313\python.exe' scripts/train_baseline.py `
  --train-csv train.csv `
  --run-dir runs\baseline_v1 `
  --model-ref metric/nemotron-3-nano-30b-a3b-bf16/transformers/default `
  --subsample-size 1200
```

保存された validation split で評価:

```powershell
& 'C:\Users\shogo\AppData\Local\Programs\Python\Python313\python.exe' scripts/evaluate_adapter.py `
  --eval-csv runs\baseline_v1\splits\valid.csv `
  --adapter-dir runs\baseline_v1\adapter `
  --model-ref metric/nemotron-3-nano-30b-a3b-bf16/transformers/default
```

提出用 zip を作成:

```powershell
& 'C:\Users\shogo\AppData\Local\Programs\Python\Python313\python.exe' scripts/package_submission.py `
  --adapter-dir runs\baseline_v1\adapter `
  --zip-path runs\baseline_v1\submission.zip
```

## 設計メモ

- このコンペの `answer` は短い最終回答なので、このベースラインは `prompt -> answer` をそのまま学習します。
- `model-ref` にはローカルパス、Hugging Face のモデル ID、または `kagglehub` 経由の Kaggle model ref を指定できます。
- `kaggle_artifacts/` に落としてある公開 Kaggle notebook を、モデル選択と提出形式の参考にしています。
- コードは責務ごとに分割してあり、CLI 側は薄く保っています。
