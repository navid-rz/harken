# Configuration (YAML)

The canonical defaults live in `config/base.yaml`. Key sections:

- `data`:
  - `preprocessed_dir`: where MFCC npy files live (`data/preprocessed`).
  - `val_split`, `test_split`: fractions used to create splits.
  - `mfcc`: MFCC parameters (`sample_rate`, `n_mfcc`, `frame_length_s`, `hop_length_s`, `fixed_duration_s`).

- `augmentation`: enables simple MFCC augmentation used during training (`max_shift_ms`, `noise_prob`, `noise_std_factor`).

- `task`:
  - `type`: `binary` or `multiclass`.
  - `class_list`: list of keyword classes for multiclass runs; `include_unknown`/`include_background` control additional labels.

- `model`: TCN model parameters: `hidden_channels`, `kernel_size`, `num_blocks`, `dropout`, `causal`, `activation`, `norm`, `use_weight_norm`, and others.

- `train`: trainer settings: `batch_size`, `num_epochs`, `learning_rate`, `weight_decay`, `device`, and DataLoader knobs (`num_workers`, `persistent_workers`).

- `qat`: quantization-aware training options (bits, symmetry, warmup epochs, QAT hyperparameters).

- `output`: directories for `weights_dir` and `plots_dir`, and `tqdm` toggling.

How to override
- Pass a YAML file into the training entrypoint (the code uses `train.utils.load_config`) or modify `config/base.yaml` and commit. Checkpoints save the `cfg` inside the checkpoint metadata for reproducibility.
