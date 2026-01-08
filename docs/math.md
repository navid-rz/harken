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

Below we make the TCN notation explicit and give per-layer forward-pass formulas so the reader can build the network bottom-up.

Notation
- Batch size: $N$
- Time length: $T$ (number of frames)
- Input channels (MFCC dim): $C^{(0)}$ (also called $C_{in}$)
- Hidden / block output channels: $C^{(i)}$ for block $i$ (in this project `C^{(i)}=C_{\mathrm{hidden}}` for all blocks by default)
- Kernel size: $k$
- Dilation of block $i$: $d_i = 2^{i}$

Input tensor:
$$X^{(0)}\in\mathbb{R}^{N\times C^{(0)}\times T}$$

Convolution operator (1D, dilation $d$, kernel $k$) from $C_{\mathrm{in}}$→$C_{\mathrm{out}}$:
$$\big(\mathrm{Conv}_{d,k}(W,b)[U]\big)_{n,c,t}=\sum_{c'=1}^{C_{\mathrm{in}}}\sum_{m=0}^{k-1} W_{c,c',m}\;U_{n,c',\,t - m\,d}+b_c$$
where left-padding makes indices valid for causal convs; the implementation trims the last $(k-1)\cdot d$ time samples after convolution to enforce causality.

Residual block (block index $i$, input $X^{(i)}$ with $C^{(i)}_{\mathrm{in}}$ channels and output channels $C^{(i)}_{\mathrm{out}}$):

1. First convolution, trim, norm, activation, dropout:
$$Y^{(i)}_1 = \mathrm{Conv}_{d_i,k}\big(W^{(i)}_1,b^{(i)}_1\big)[X^{(i)}]$$
$$\tilde Y^{(i)}_1 = \mathrm{Trim}_{(k-1)d_i}(Y^{(i)}_1)$$
$$Z^{(i)}_1 = \mathrm{Norm}^{(i)}_1\big(\tilde Y^{(i)}_1\big)$$
$$A^{(i)}_1 = \sigma^{(i)}_1\big(Z^{(i)}_1\big)\quad(\text{ReLU / GELU / PReLU})$$
$$D^{(i)}_1 = \mathrm{Dropout}^{(i)}_1\big(A^{(i)}_1\big)$$

2. Second convolution, trim, norm, (optional) dropout:
$$Y^{(i)}_2 = \mathrm{Conv}_{d_i,k}\big(W^{(i)}_2,b^{(i)}_2\big)[D^{(i)}_1]$$
$$\tilde Y^{(i)}_2 = \mathrm{Trim}_{(k-1)d_i}(Y^{(i)}_2)$$
$$Z^{(i)}_2 = \mathrm{Norm}^{(i)}_2\big(\tilde Y^{(i)}_2\big)$$
$$D^{(i)}_2 = \mathrm{Dropout}^{(i)}_2\big(Z^{(i)}_2\big)$$

3. Residual / projection and final activation:
$$R^{(i)} = P^{(i)}\big(X^{(i)}\big) = \begin{cases} X^{(i)}, & C^{(i)}_{\mathrm{in}} = C^{(i)}_{\mathrm{out}},\\[4pt]
\mathrm{Conv}_{1,1}(W^{(i)}_p,0)[X^{(i)}], & \text{otherwise (1x1 projection)}\end{cases}$$
$$U^{(i)} = D^{(i)}_2 + R^{(i)}$$
$$X^{(i+1)} = \sigma^{(i)}_2\big(U^{(i)}\big)\in\mathbb{R}^{N\times C^{(i)}_{\mathrm{out}}\times T}\,.$$ 

Notes on shapes and variants
- If the implementation uses a single `hidden_channels` value then $C^{(i)}_{\mathrm{out}}=C^{(i+1)}_{\mathrm{in}}=C_{\mathrm{hidden}}$ for all blocks; the very first block uses $C^{(0)}$ (input MFCC dim) as $C^{(0)}_{\mathrm{in}}$.
- Depthwise-separable conv variant: each $\mathrm{Conv}_{d,k}$ is replaced by
  $$\mathrm{PW}\circ\mathrm{DW},$$
  where the depthwise (DW) conv applies per-channel filters (groups=$C_{\mathrm{in}}$) and the pointwise (PW) conv is a $1\times1$ mixing across channels. Formally,
  $$\mathrm{DW}_{d,k}[U]_{n,c,t}=\sum_{m=0}^{k-1} w^{\mathrm{dw}}_{c,m}\;U_{n,c,t-m\,d}$$
  $$\mathrm{PW}[V]_{n,c,t}=\sum_{c'=1}^{C_{\mathrm{in}}} w^{\mathrm{pw}}_{c,c'}\;V_{n,c',t}+b^{\mathrm{pw}}_c\,.$$

Full model readout
- After $B$ residual blocks the TCN returns $X^{(B)}\in\mathbb{R}^{N\times C^{(B)}\times T}$.
- Global pooling (adaptive avg or max) collapses time:
$$h = \mathrm{Pool}\big(X^{(B)}\big)\in\mathbb{R}^{N\times C^{(B)}}\,.$$ 
- Final linear classifier:
$$\mathrm{logits} = W_{\mathrm{fc}}\,h + b_{\mathrm{fc}}\in\mathbb{R}^{N\times N_{\mathrm{classes}}}\,.$$ 

This bottom-up view shows the local operations (per-block convolutions, trims, norms, activations, dropouts) and how residual connections and optional 1x1 projections preserve a stable temporal dimension $T$ while changing the channel dimensionality $C^{(i)}$ across the network.

Trim, normalization and dropout (formulas)

- Trim (causal right-trim): if a convolution with dilation $d$ and kernel $k$ is applied with left-padding then the conv output has $T + r$ timesteps where $r=(k-1)d$. The implemented trim removes the last $r$ samples:
$$\mathrm{Trim}_r:\;Y\in\mathbb{R}^{N\times C\times (T+r)}\mapsto \tilde Y\in\mathbb{R}^{N\times C\times T},\qquad \tilde Y_{n,c,t}=Y_{n,c,t}\; (t=1\dots T).$$

- Normalization (per-block): the code supports BatchNorm1d, GroupNorm, LayerNorm (wrapped to operate on channels), or Identity. The formulas are:

  - BatchNorm (per-channel, statistics over batch and time): for channel $c$ compute
  $$\mu_c=\frac{1}{NT}\sum_{n=1}^N\sum_{t=1}^T Y_{n,c,t},\qquad
  \sigma_c^2=\frac{1}{NT}\sum_{n=1}^N\sum_{t=1}^T\big(Y_{n,c,t}-\mu_c\big)^2.$$ 
  Normalized output:
  $$\mathrm{BN}(Y)_{n,c,t}=\gamma_c\frac{Y_{n,c,t}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}+\beta_c.$$ 

  - GroupNorm (G groups of channels): partition channels into groups; for group $g$ with channel indices $\mathcal{C}_g$ compute mean/var over $(n,t,c\in\mathcal{C}_g)$ and apply learnable scale/shift per-channel within the group (formally analogous to BN but with group-scoped moments).

  - LayerNorm (channel-wise for each time step): for each sample $n$ and time $t$ compute moments across channels:
  $$\mu_{n,t}=\frac{1}{C}\sum_{c=1}^C Y_{n,c,t},\qquad \sigma_{n,t}^2=\frac{1}{C}\sum_{c=1}^C\big(Y_{n,c,t}-\mu_{n,t}\big)^2,$$
  $$\mathrm{LN}(Y)_{n,c,t}=\gamma_c\frac{Y_{n,c,t}-\mu_{n,t}}{\sqrt{\sigma_{n,t}^2+\epsilon}}+\beta_c.$$ 

  - Identity: $\mathrm{Id}(Y)=Y$.

- Dropout (in training): given dropout probability $p$, Dropout samples a mask $M_{n,c,t}\sim\mathrm{Bernoulli}(1-p)$ and rescales to preserve expectation:
$$\mathrm{Dropout}_p(Y)_{n,c,t}=\frac{M_{n,c,t}}{1-p}\;Y_{n,c,t}.$$ In evaluation mode Dropout is the identity.

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
