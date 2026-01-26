**Quick Examples**
- **Activate venv (PowerShell):** . .\venv\Scripts\Activate.ps1
- **Activate venv (CMD):** .\venv\Scripts\activate.bat

**Feature Extraction**
- **Test MFCC vs log-mel features:** python -m feature_extraction.extract_features
  - MFCC: 16 coefficients, can be negative (use for signed int8 hardware)
  - Log-mel: 40 bands, all positive (use for unsigned uint8 hardware)

**Training**
- **Run trainer as module:** python -m train.train --config config/base.yaml

**Quantization-Aware Training (QAT)**
- **Train with 4-bit unsigned activations and 8-bit signed weights** (paper setup):
  ```bash
  # First, configure config/base.yaml with:
  # qat:
  #   weight_bits: 8
  #   act_bits: 4
  #   weight_symmetric: true    # signed weights
  #   act_symmetric: false      # unsigned activations [0, 15]
  #   fold_batchnorm: true      # fold BN before QAT
  #   warmup_epochs: 1
  #   num_epochs: 10
  #   batch_size: 32
  #   learning_rate: 0.0001
  
  python -m train.qat --config config/base.yaml --weights checkpoints/model_ckpt_fp.pt
  ```
  Automatically folds BatchNorm into Conv layers before QAT, then trains with fake-quantization nodes.

- **Fold BatchNorm manually (for inference deployment):**
  ```python
  from train.utils import fold_batchnorm
  from model.model import DilatedTCN
  from config import load_config
  import torch
  
  cfg = load_config('config/base.yaml')
  model = DilatedTCN.from_config(cfg)
  model.load_state_dict(torch.load('checkpoints/model_ckpt_fp.pt')['state_dict'])
  model = fold_batchnorm(model)  # Folds BN into Conv weights
  torch.save(model.state_dict(), 'checkpoints/model_bn_folded.pt')
  ```

**Weight visualization**
- **Overall + multi-quant (global):**
  python -m analysis.plot_weights --path checkpoints/model_ckpt_fp.pt --outdir plots/weights_viz --quant-bits 8,4,3 --quant-scheme global --global-percentile 90
- **Skip per-layer files:** remove the `--per-layer` flag (default is no per-layer output).

**Inference / analysis**
- **Run interactive GUI demo:**
  python -m demo.app --config config/base.yaml --weights checkpoints/model_ckpt_fp.pt --device auto

- **Evaluate saved checkpoint on test set:**
  (uses training `Trainer.save()` checkpoint format)
  python -c "from config import load_config; from model.model import DilatedTCN; import torch; cfg=load_config('config/base.yaml'); m=DilatedTCN.from_config(cfg); obj=torch.load('checkpoints/model_ckpt_fp.pt', map_location='cpu'); sd = obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj; m.load_state_dict(sd); m.eval(); print('Model loaded')"

- **Post-Training Quantization (PTQ) evaluation:**
  Evaluate model accuracy across different quantization bit-widths:
  ```bash
  # Float baseline + quantized weights and fp activations
  python -m analysis.eval_inf_ptq --config config/base.yaml --weights checkpoints/model_ckpt_fp.pt --scheme global --global-percentile 90 --bits float,8,7,6,5,4,3,2 --act-bits float --dataset val --symmetric

  # Custom weight/activation bit combinations
  python -m analysis.eval_inf_ptq --config config/base.yaml --weights checkpoints/model_ckpt_fp.pt --bits 8,4,2 --act-bits 8,8,4 --dataset test --symmetric --scheme per_tensor
  ```
  Generates plot at `plots/eval/ptq_metrics_vs_bits.png` showing accuracy/precision/recall/F1 vs bit-width.

- **Run the provided evaluation helpers:**
  - Use `train.evaluate.evaluate_model` through scripts, or run full evaluation by loading a checkpoint into `Trainer`-style code.

**Notes**
- If using `BatchNorm1d` during training and you need a simpler inference model, fold BatchNorm into Conv weights before export (BatchNorm can be folded; GroupNorm/LayerNorm cannot).
- Data is read from `data/preprocessed` by default; dataset loading may take time if many `.npy` files exist.

If you'd like, I can add runnable examples/scripts for folding BN, exporting a folded checkpoint, or a small wrapper to run inference on a single MFCC `.npy` file. Let me know which.