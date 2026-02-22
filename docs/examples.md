**Quick Examples**
- **Activate venv (PowerShell):** . .\venv\Scripts\Activate.ps1
- **Activate venv (CMD):** .\venv\Scripts\activate.bat

**Feature Extraction**
- **Test MFCC vs log-mel features:** python -m feature_extraction.extract_features
  - MFCC: 16 coefficients, can be negative (use for signed int8 hardware)
  - Log-mel: 40 bands, all positive (use for unsigned uint8 hardware)
- **Preprocess dataset:** python -m feature_extraction.preprocess_features --config config/base.yaml
  - Extracts features from all WAV files in data/speech_commands_v0.02/
  - Saves as .npy files in data/preprocessed/
  - Feature type (mfcc/log-mel), normalization controlled by config
- **Analyze range of values for preprocessed data:** python -m data.analyze_preprocessed_features --data-dir data/preprocessed_log_mel_40_norm --feature-type log-mel

**Training**
- **Run trainer as module:** python -m train.train --config config/base.yaml

**Inference Evaluation and Post-Training Quantization
- **Run inference evaluation with linearly quantized weights, same weight and activation limits are applied as specified in config.yaml train section** python -m analysis.eval_inf_ptq --config "config\experiments\logmel16_agc_no_batch_norm.yaml" --weights "checkpoints\logmel16_agc_no_batch_norm\model_weights_fp.pt" --quant-method linear --linear-bits "8,7,6,5,4,3" --verbose

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
  
  python -m train.qat --config "config\experiments\logmel16_agc_no_batch_norm.yaml" --weights "checkpoints\logmel16_agc_no_batch_norm\model_weights_fp.pt" --epochs 2
  ```
  Automatically folds BatchNorm into Conv layers before QAT, then trains with fake-quantization nodes.

- **Plot trained quantized weights**
  python -m analysis.plot_quant_npz --npz "checkpoints\logmel16_agc_no_batch_norm\qat\qat_weights_linear_4bit_act_10bit.npz" --outdir "plots\logmel16_agc_no_batch_norm\qat"

**Run the Demo App**
  python -m demo.app --config "config\experiments\logmel16_agc_no_batch_norm.yaml" --weights "checkpoints\logmel16_agc_no_batch_norm\qat\qat_weights_linear_4bit_act_10bit.npz"

**Post QAT Noise Injected Inference Evaluation**
  python -m analysis.eval_inf_weight_analog_error --config "config\experiments\logmel16_agc_no_batch_norm.yaml" --weights "checkpoints\logmel16_agc_no_batch_norm\qat\qat_weights_linear_4bit_act_10bit.npz" --analog-weight-error "0.0,0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0" --error-type gaussian --num-trials 3 --dataset val --verbose --save-results

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

**Weight Visualization and Analysis**
- **Basic weight histograms (all models):**
  ```bash
  # Generate histograms for original and BatchNorm-folded weights
  python -m analysis.plot_weights --path checkpoints/logmel16_agc/model_weights_fp.pt --outdir plots/logmel16_agc/weights
  python -m analysis.plot_weights --path checkpoints/logmel40_unnorm/model_weights_fp.pt --outdir plots/logmel40_unnorm/weights
  python -m analysis.plot_weights --path checkpoints/mfcc28_agc/model_weights_fp.pt --outdir plots/mfcc28_agc/weights
  ```

- **Multi-quantization analysis with different schemes:**
  ```bash
  # Per-tensor quantization (default) with 99.9th percentile scale
  python -m analysis.plot_weights --path checkpoints/model_weights_fp.pt --outdir plots/weights --quant-bits 8,4,3 --quant-scheme per_tensor --global-percentile 99.9

  # Global quantization (single scale for all weights) with max scale
  python -m analysis.plot_weights --path checkpoints/model_weights_fp.pt --outdir plots/weights --quant-bits 8,4,3 --quant-scheme global --global-percentile 100.0

  # Per-channel quantization (best accuracy) with 99th percentile
  python -m analysis.plot_weights --path checkpoints/model_weights_fp.pt --outdir plots/weights --quant-bits 8,4,3 --quant-scheme per_channel --global-percentile 99.0
  ```

- **Quantization scheme comparison:**
  - **`per_tensor`** (default): One scale per weight tensor - good balance of accuracy and simplicity
  - **`global`**: Single scale across ALL weight tensors - most hardware-friendly, potentially lower accuracy
  - **`per_channel`**: One scale per output channel - best accuracy, more complex hardware implementation

- **Percentile scale selection:**
  - **`100.0`** (default): Uses maximum absolute value for scaling
  - **`99.9`**: Ignores extreme outliers, often improves quantization quality
  - **`99.0`**: More aggressive outlier removal, may help with noisy weight distributions

- **Output files generated:**
  - `all_weights_hist.png` - Histogram of original floating-point weights
  - `all_weights_hist_folded.png` - Histogram after BatchNorm folding (more representative for deployment)
  - `weights_hist_multi_quant.png` - Multi-panel quantization analysis (uses folded weights)

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