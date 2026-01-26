import torch
import torch.nn as nn
import torch.nn.functional as F
# Helper to apply weight normalization re-parameterization to layers.
# See: https://pytorch.org/docs/stable/nn.html#weight-norm
# It replaces a parameter `weight` with `weight_v` and `weight_g` such that
# `weight = weight_g * weight_v / ||weight_v||`. This often improves
# optimization stability for conv layers by decoupling magnitude from direction.
from torch.nn.utils import weight_norm
from typing import Optional, Literal, List
import math


class CausalTrim(nn.Module):
    """Trim right-side padding to enforce causality for a given kernel and dilation.
    Assumes padding=(kernel_size-1)*dilation was applied before conv.
    """
    def __init__(self, trim: int):
        super().__init__()
        self.trim = int(trim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.trim == 0 else x[:, :, :-self.trim]


def act_norm(norm: Literal["batch", "group", "layer", "none"], num_channels: int, *, groups: int = 8):
    if norm == "batch":
        return nn.BatchNorm1d(num_channels)
    elif norm == "group":
        # clamp groups to divisors of num_channels
        g = max(1, min(groups, num_channels))
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    elif norm == "layer":
        # LayerNorm over channel dimension by swapping, applying, then swapping back via a wrapper
        return ChannelLayerNorm(num_channels)
    else:
        return nn.Identity()


class ChannelLayerNorm(nn.Module):
    """LayerNorm across channels for (N, C, T) tensors.
    PyTorch's LayerNorm expects last dims; we transpose to (N, T, C), normalize, transpose back.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        x = x.transpose(1, 2)  # (N, T, C)
        x = self.ln(x)
        x = x.transpose(1, 2)  # (N, C, T)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1D conv with optional weight norm and causal trim.

    Args:
        in_ch, out_ch: channels
        kernel_size: int
        dilation: int
        causal: if True, uses left padding and trims right side
        use_weight_norm: apply weight normalization to conv layers
        bias: include bias terms
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1,
                 causal: bool = True, use_weight_norm: bool = False, bias: bool = True):
        super().__init__()
        pad = (kernel_size - 1) * dilation if causal else (kernel_size // 2) * dilation

        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, dilation=dilation,
                                   padding=pad, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)
        if use_weight_norm:
            self.depthwise = weight_norm(self.depthwise)
            self.pointwise = weight_norm(self.pointwise)
        self.trim = CausalTrim(pad) if causal else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.trim(x)
        x = self.pointwise(x)
        return x


class ResidualBlock1D(nn.Module):
    """TCN residual block with two convolutions (standard or depthwise-separable).

    Structure per block:
        x → Conv1 → Norm → Act → Drop → Conv2 → Norm → (optional Drop) → +Skip → Act

    Notes:
        * Both convs use the same dilation; dilation typically doubles each block.
        * If in/out channels mismatch, a 1x1 conv is used for the residual path.
        * Causality is enforced by left padding + right trim.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        *,
        causal: bool = True,
        dropout: float = 0.0,
        activation: Literal["relu", "gelu", "prelu"] = "relu",
        norm: Literal["batch", "group", "layer", "none"] = "batch",
        groups_for_groupnorm: int = 8,
        use_weight_norm: bool = False,
        depthwise_separable: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        # Build the residual stack: `num_blocks` ResidualBlock1D modules are
        # appended to `self.tcn`. Each block uses exponentially increasing
        # dilation (2**i) so the network's receptive field grows quickly.
        # `in_ch` tracks the current number of channels as blocks are stacked.

        pad = (kernel_size - 1) * dilation if causal else (kernel_size // 2) * dilation

        # Build the two convolutional layers used by the residual block.
        # Optionally use depthwise-separable convs (depthwise per-channel then 1x1 pointwise).
        # `pad` ensures left-padding for causal convs; we trim right-side outputs after conv.
        if depthwise_separable:
            conv1 = DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size, dilation,
                                             causal=causal, use_weight_norm=use_weight_norm, bias=bias)
            conv2 = DepthwiseSeparableConv1d(out_ch, out_ch, kernel_size, dilation,
                                             causal=causal, use_weight_norm=use_weight_norm, bias=bias)
        else:
            conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                              padding=pad, bias=bias)
            conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation,
                              padding=pad, bias=bias)
            # Optionally apply weight normalization to the conv parameters. This
            # is a re-parameterization of `weight` into `weight_v` and `weight_g`.
            # It affects the optimizer trajectory but not the functional form
            # of the convolution at inference once `weight` is materialized.
            if use_weight_norm:
                conv1 = weight_norm(conv1)
                conv2 = weight_norm(conv2)

        self.conv1 = conv1
        self.conv2 = conv2
        self.trim = CausalTrim(pad) if causal else nn.Identity()

        # Normalization layers applied to conv outputs. These normalize
        # activation statistics (per-batch, per-group, or per-sample depending
        # on `norm`) and include learnable affine parameters (scale/shift)
        # when enabled. For deployment on hardware without activation
        # normalization we fold BatchNorm (if used) into the preceding conv
        # weights after training; Group/LayerNorm cannot be exactly folded.
        self.norm1 = act_norm(norm, out_ch, groups=groups_for_groupnorm)
        self.norm2 = act_norm(norm, out_ch, groups=groups_for_groupnorm)

        if activation == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif activation == "gelu":
            self.act1 = nn.GELU()
            self.act2 = nn.GELU()
        elif activation == "prelu":
            self.act1 = nn.PReLU(num_parameters=out_ch)
            self.act2 = nn.PReLU(num_parameters=out_ch)

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual path
        self.need_proj = in_ch != out_ch
        self.proj = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
            if self.need_proj else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # First conv block: conv -> trim (causal) -> norm -> activation -> dropout
        out = self.conv1(x)
        out = self.trim(out)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        # Second conv block: conv -> trim -> norm -> dropout
        # The output of this stage will be added to the residual/skip path.
        out = self.conv2(out)
        out = self.trim(out)
        out = self.norm2(out)
        out = self.dropout2(out)

        # Residual connection: possibly project input channels to match output
        # channels with a 1x1 conv (no bias). Then add and apply final activation.
        res = self.proj(residual)
        out = out + res
        out = self.act2(out)
        return out


class DilatedTCN(nn.Module):
    """Residual TCN for sequence classification (e.g., KWS with MFCCs).

    Args:
        input_channels: number of MFCC features per frame
        num_blocks: number of residual blocks
        hidden_channels: channels in hidden blocks
        kernel_size: conv kernel size
        num_classes: classifier output dim
        dropout: dropout prob inside blocks
        causal: enforce causality (left pad + right trim)
        activation: relu | gelu | prelu
        norm: batch | group | layer | none
        use_weight_norm: apply weight norm to convs
        depthwise_separable: use depthwise-separable convs in blocks
        pool: 'avg' | 'max' global pooling before FC
    """
    def __init__(
        self,
        input_channels: int,
        num_blocks: int,
        hidden_channels: int,
        *,
        kernel_size: int = 3,
        num_classes: int = 12,
        dropout: float = 0.1,
        causal: bool = True,
        activation: Literal["relu", "gelu", "prelu"] = "relu",
        norm: Literal["batch", "group", "layer", "none"] = "batch",
        groups_for_groupnorm: int = 8,
        use_weight_norm: bool = False,
        depthwise_separable: bool = False,
        pool: Literal["avg", "max"] = "avg",
        bias: bool = True,
    ):
        super().__init__()

        blocks: List[nn.Module] = []
        in_ch = input_channels
        for i in range(num_blocks):
            d = 2 ** i  # exponential dilation per block
            out_ch = hidden_channels
            blocks.append(
                ResidualBlock1D(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation=d,
                    causal=causal,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    groups_for_groupnorm=groups_for_groupnorm,
                    use_weight_norm=use_weight_norm,
                    depthwise_separable=depthwise_separable,
                    bias=bias,
                )
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        # `self.tcn` outputs tensor with shape (N, hidden_channels, T').
        # Global pooling below reduces T' -> 1 producing (N, hidden_channels).

        self.pool = nn.AdaptiveAvgPool1d(1) if pool == "avg" else nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(
                        in_features=hidden_channels,
                        out_features=num_classes,
                        bias=bias
                    )

        self.apply(self._init_weights)
        # Note: if using `BatchNorm1d` layers during training, fold them into
        # preceding `Conv1d` layers before exporting the model for inference.
        # GroupNorm/LayerNorm cannot be exactly folded and must remain active.

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C_in, T) where N is batch size, C_in is input channels, T is sequence length → logits: (N, num_classes)"""
        # Expect input MFCC/feature tensors shaped (N, C_in, T).
        # Processing overview:
        #  - `self.tcn` processes temporal features and returns (N, hidden_channels, T').
        #  - `self.pool` collapses the temporal dimension producing (N, hidden_channels).
        #  - `self.fc` maps to logits (N, num_classes).
        out = self.tcn(x)
        out = self.pool(out).squeeze(-1)  # (N, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

    @staticmethod
    def receptive_field(kernel_size: int, num_blocks: int) -> int:
        """Receptive field for this TCN (two convs per block, equal dilation per block).
        RF = 1 + (k-1) * 2 * sum_{i=0}^{B-1} 2^i = 1 + 2*(k-1)*(2^B - 1)
        """
        return 1 + 2 * (kernel_size - 1) * (2 ** num_blocks - 1)

    @classmethod
    def from_config(cls, cfg: dict) -> "DilatedTCN":
        """Construct a DilatedTCN from a configuration dictionary.

        This mirrors the previous `build_model_from_cfg` logic but lives on the
        model class as a clear alternate constructor.
        """
        m = cfg["model"]
        t = cfg["task"]
        mfcc = cfg["data"]["mfcc"]

        input_channels = int(mfcc["n_mfcc"])
        kernel_size = int(m["kernel_size"])
        hidden_channels = int(m["hidden_channels"])
        dropout = float(m["dropout"])
        num_blocks = int(m["num_blocks"])
        causal = bool(m.get("causal", True))
        activation = str(m.get("activation", "relu"))
        norm = str(m.get("norm", "batch"))
        groups_for_groupnorm = int(m.get("groups_for_groupnorm", 8))
        use_weight_norm = bool(m.get("use_weight_norm", False))
        depthwise_separable = bool(m.get("depthwise_separable", False))
        pool = str(m.get("pool", "avg"))
        bias = bool(m.get("bias", True))

        # Determine number of classes from task block
        class_list = t.get("class_list", [])
        include_unknown = bool(t.get("include_unknown", False))
        include_background = bool(t.get("include_background", False))
        num_classes = len(class_list)
        if include_unknown:
            num_classes += 1
        if include_background:
            num_classes += 1

        model = cls(
            input_channels=input_channels,
            num_blocks=num_blocks,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_classes=num_classes,
            dropout=dropout,
            causal=causal,
            activation=activation,
            norm=norm,
            groups_for_groupnorm=groups_for_groupnorm,
            use_weight_norm=use_weight_norm,
            depthwise_separable=depthwise_separable,
            pool=pool,
            bias=bias,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            "[MODEL] DilatedTCN built: "
            f"in_ch={input_channels}, hidden={hidden_channels}, "
            f"kernel={kernel_size}, blocks={num_blocks}, dropout={dropout}, classes={num_classes}, "
            f"causal={causal}, act={activation}, norm={norm}, groups={groups_for_groupnorm}, "
            f"dwise_sep={depthwise_separable}, w_norm={use_weight_norm}, pool={pool}, bias={bias}"
        )
        print(f"[PARAMS] total={total_params:,} trainable={trainable_params:,}")
        return model

