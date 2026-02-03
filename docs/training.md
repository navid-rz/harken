# Training Guide

This file summarizes the training pipeline implemented in `train/train.py` and the commonly used hyperparameters from `config/base.yaml`.

Pipeline summary
- Data loading: `data_loader.utils.make_datasets` creates train/val/test splits from `data/preprocessed` using `data.val_split` and `data.test_split`.
- Augmentation: controlled by `augmentation.enable`; when on a `FeatureAugment` transform applies random time shifts and noise during training.
- Model build: `DilatedTCN.from_config(cfg)` constructs `DilatedTCN` using fields from `config/base.yaml` `model` section (channels, kernel size, num_blocks, dropout, causal, etc.).
- Optimizer: Adam with `train.learning_rate` and `train.weight_decay` defaults (see `config/base.yaml`).
- Loss:
  - Binary tasks: `BCEWithLogitsLoss`, optionally with `pos_weight`.
  - Multiclass tasks: `CrossEntropyLoss` with optional `class_weights` and `label_smoothing`.
- Training loop: Trainer runs `num_epochs`, logs metrics to CSV (`plots/training/metrics.csv`) and saves final weights and a rich checkpoint under the `output.weights_dir`.

Metrics and logging
- Per-epoch: loss, accuracy, precision, recall, F1 are recorded for train/val and appended to CSV.
- For multiclass runs a confusion matrix is computed on test (`train.Trainer` uses `compute_confusion_matrix`).

QAT (quantization-aware training)
- Configured in `config/base.yaml` under `qat`:
  - `weight_bits`, `act_bits`, symmetry flags, warmup epochs, and separate learning rates for QAT.
  - QAT uses smaller default `num_epochs` and reduced `learning_rate` to fine-tune quantized-aware weights.

Default hyperparameters (from `config/base.yaml`)
- `train.batch_size`: 32
- `train.num_epochs`: 50
- `train.learning_rate`: 1e-4
- `train.weight_decay`: 1e-5
- `model.hidden_channels`: 26
- `model.kernel_size`: 3
- `model.num_blocks`: 4
- `model.dropout`: 0.2

Tips
- On Windows, ensure DataLoader `num_workers` is safe (default 2). The code already uses `train.device='auto'` to select CUDA if present.
- To reproduce results, pin `augmentation.seed` and ensure deterministic dataloaders where needed.
