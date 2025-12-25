# Math & Signal Processing

This page summarizes the key mathematical formulas and numeric choices used in Harken.

**MFCC processing**
- Sampling rate: $\mathrm{sr}=16000$ (from `config/base.yaml`).
- Window / FFT sizes (seconds → samples):
  - Frame length $L_s = \text{frame\_length\_s}$ → $N_{fft} = \lfloor L_s \cdot sr \rceil$ (example: $0.032\,s \times 16000 = 512$ samples).
  - Hop length $H_s = \text{hop\_length\_s}$ → $H = \lfloor H_s \cdot sr \rceil$ (example: $0.016\,s \times 16000 = 256$ samples).
- MFCC matrix shape: time-steps × `n_mfcc` (project defaults use `n_mfcc=28`). `extract_mfcc.py` returns `(y, mfcc.T)` so frames are rows.

**Temporal Convolutional Network (TCN)**
- Convolutions are 1D over the time dimension; input tensors are shaped `(N, C_in, T)` where `C_in` is MFCC channels.
- Exponential dilation schedule per block: $d_i = 2^i$ for block index $i=0..B-1$ (see `model.DilatedTCN`).
- Two convolutions per residual block; when causal padding is used the effective padding per conv is $(k-1)\cdot d$ and the implementation trims the right side to enforce causality.

- Receptive field (closed form used in code):
$$\mathrm{RF} = 1 + 2(k-1)\left(2^{B} - 1\right)$$
where $k$ is kernel size and $B$ is number of residual blocks (see `DilatedTCN.receptive_field`).

**Quantization math (weights)**
- Integer range for symmetric signed quant with `bits`:
$$q_{\max} = 2^{\text{bits}-1} - 1$$
- Symmetric per-tensor scale (example implementation in `quantization/core.py`):
  - Given floating weights $w$, compute $s = \max(|w|)$ (or a percentile-based value for global scheme).
  - Scale factor: $\alpha = s / q_{\max}$.
  - Integer codes: $q = \operatorname{clip}\big(\mathrm{round}(w / \alpha), -q_{\max}, q_{\max}\big)$.
  - Dequantized weight (used for simulation): $\hat w = q \cdot \alpha$.

- Asymmetric (zero-point) scheme used in code: compute $\min,\max$, then
$$\alpha = \frac{\max-\min}{q_{\max}-q_{\min}},\qquad z = \mathrm{round}(q_{\min} - \min/\alpha)$$
and quantize with $q=\mathrm{round}(w/\alpha + z)$.

- Per-channel quantization computes scales per output channel (or per-axis) and applies the same formula channel-wise.

These formulas are implemented across `quantization/core.py` and used for both code-book generation and simulated quantized weights.
