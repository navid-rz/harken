# Inference & Evaluation

This page describes how inference is performed and which metrics are computed.

Inference flow
- Model expects inputs shaped `(N, C, T)` where `C` is MFCC channels and `T` is time steps (see `model.DilatedTCN.forward`).
- Typical steps:
  1. Load preprocessed MFCCs or run `feature_extraction.extract_mfcc` and format into `(C, T)`.
  2. Stack into batches and send to device (`Trainer` uses `pin_memory` and `DataLoader`).
  3. Obtain logits from the model: `logits = model(batch_x)`.
  4. For binary tasks apply `sigmoid` and threshold to get class predictions; for multiclass use `argmax` over logits.

Batch processing
- Use the same `batch_size` as during training or a suitable inference batch size. For large batches watch memory with `pin_memory` and `num_workers` options.

Metrics
- Binary: accuracy, precision, recall, F1 (the project computes TP/FP/FN in `train._loop`). It supports threshold search to select the best validation threshold (`Trainer.find_best_threshold`).
- Multiclass: accuracy and macro-precision/recall/F1. Confusion matrices are computed and can be plotted via `analysis.plot_metrics`.

Checkpoint & weight loading
- Use `torch.load` to load saved state dicts from `checkpoints/model_weights_fp.pt` or the richer checkpoint in `checkpoints/model_ckpt_fp.pt` (the trainer saves both in `Trainer.save`).

Quantized inference
- The repository provides functions to simulate quantized weights (`quantization/core.py`). For actual integer inference you will need a runtime that consumes quantized integer codes and scales — the helper `quantize_state_dict_to_codes` exports integer codes suitable for packing.
