# Architecture & Code Structure

This document maps the repository layout and the primary modules used by the project.

Top-level folders (selected):
- `model/` — model definitions. Key symbols:
  - `DilatedTCN` (residual TCN classifier)
  - `ResidualBlock1D`, `DepthwiseSeparableConv1d`, `CausalTrim`
- `feature_extraction/` — MFCC extraction and preprocessing (`extract_mfcc.py`).
- `data_loader/` — dataset classes and utilities for MFCC datasets (`mfcc_dataset.py`, `multiclass_dataset.py`, `binary_dataset.py`).
- `train/` — training and evaluation orchestration (`train.py`, `evaluate.py`, `qat.py`, `utils.py`).
- `quantization/` — quant helpers and weight-code exporters (`core.py`).
- `analysis/` — plotting and inspection utilities (metrics, weight visualizations).
- `checkpoints/` and `plots/` — outputs for weights and visualizations.

Simple diagram (data → model → output):

data/preprocessed (MFCCs)
    ↓ (DataLoader / augmentation)
    → `train.Trainer` builds model via `build_model_from_cfg(cfg)`
    → `model.DilatedTCN` (N, C, T) → global pool → `fc` → logits
    ↓
  optimizer / loss → training loop → `checkpoints/` and `plots/`

Where code hooks live:
- Model builder: `train.utils.build_model_from_cfg` reads `config/base.yaml` `model` block.
- Datasets: `data_loader.utils.make_datasets` sets up splits, transforms, and `MFCCAugment` when enabled.
- Quantization/export: `quantization.core.quantize_model_weights` and `quantize_state_dict_to_codes`.

Notes and suggestions for contributors:
- `model.py` is already modular — add new block types by creating a new nn.Module and adding an option in the model constructor.
- When changing MFCC parameters, update `config/base.yaml` and re-run preprocessing in `data/preprocessed`.
