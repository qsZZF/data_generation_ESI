"""G4 NeuroDyn-EEG inverse model and supervised fitting.

Modified from Gnosis-Neurodynamics/NeuroDyn-EEG, commit
c9dd9074818e80bfe8e8ba7b2787cf5d8a2be38c, TrainWorks/src/model.py.
Apache-2.0; see NEURODYN_LICENSE and NEURODYN_NOTICE in this directory.
Changes: physical-unit targets, prior-consistent bounds, dynamic atlas/montage,
relative leadfield regularization, compact profile and leak-checked training.
The model retains scalp-temporal, leadfield-inverse and spectral branches,
regional self-attention, and cross-attention to scalp temporal tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.fft as _torch_fft
import torch.nn as nn

from .simulator import PARAMETER_BOUNDS, PARAMETER_NAMES

TARGET_BOUNDS = {**PARAMETER_BOUNDS, "delay_ms": (0.0, 10.0)}
TARGET_NAMES = tuple(TARGET_BOUNDS)
# Paper Table S3, plus unit weight for the global delay term in Eq. 5.
PAPER_WEIGHTS = dict(zip(TARGET_NAMES, (1.2, 1., 1.5, 1.5, 1., 1.5, 1., 5., 5., 20., 20., 1.)))

class BoundSigmoid(nn.Module):
    def __init__(self, low: float, high: float):
        super().__init__()
        self.low, self.high = float(low), float(high)

    def forward(self, x):
        return self.low + (self.high - self.low) * torch.sigmoid(x)


class ConstrainedLayer(nn.Module):
    """Match the joint training prior, including ordering and absolute bounds."""

    def __init__(self, low: float, high: float, k: float = 12 / 5):
        super().__init__()
        self.low, self.high = float(low), float(high)
        self.k = float(k)

    def forward(self, tau_e: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        low = tau_e.clamp_min(self.low)
        high = (self.k * tau_e).clamp_max(self.high)
        return low + (high - low) * torch.sigmoid(raw)


class Conv1dPad(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation=1, bias=True):
        super().__init__()
        total_p = kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        left_p = total_p // 2
        right_p = total_p - left_p
        self.net = nn.Sequential(
            nn.ConstantPad1d((left_p, right_p), 0),
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
            ),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadConv(nn.Module):
    def __init__(self, in_ch, hidden, stride=2):
        super().__init__()
        unit = hidden // 4

        def make(k):
            return nn.Sequential(
                Conv1dPad(in_ch, unit, kernel_size=k, stride=stride, bias=False),
                nn.BatchNorm1d(unit),
                nn.LeakyReLU(0.2),
            )

        self.h1 = make(3)
        self.h2 = make(5)
        self.h3 = make(7)
        self.h4 = make(9)

    def forward(self, x):
        return torch.cat([self.h1(x), self.h2(x), self.h3(x), self.h4(x)], dim=1)


# ---------------------------------------------------------------------------
# Pseudo-inverse helper
# ---------------------------------------------------------------------------
def compute_lead_field_pinv(lead_field: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Relative-ridge inverse; solve in channel space, including for 994 nodes.

    Normalize the forward's RMS first: the input EEG is standardized, so the
    feature extractor must not depend on the arbitrary leadfield unit scale.
    """
    L = np.asarray(lead_field, dtype=np.float64)
    if L.ndim != 2 or not np.isfinite(L).all() or not np.any(L):
        raise ValueError("leadfield must be a finite, nonzero channels x regions array")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("pinv ridge must be positive")
    L = L / np.max(np.abs(L))
    L = L / np.sqrt(np.mean(L * L))
    gram = L @ L.T
    penalty = ridge * np.trace(gram) / L.shape[0]
    return np.linalg.solve(gram + penalty * np.eye(L.shape[0]), L).T.astype(np.float32)



# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class DilatedResBlock(nn.Module):
    """Pre-activation residual block with dilated 1D convs.

    Used to enlarge the temporal receptive field of the scalp branch without
    further down-sampling. Takes (B, C, T) and returns (B, C, T).
    """

    def __init__(
        self, channels: int, kernel: int = 5, dilation: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel, dilation=dilation, padding=pad
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel, dilation=dilation, padding=pad
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv1(self.act1(self.bn1(x)))
        y = self.drop(y)
        y = self.conv2(self.act2(self.bn2(y)))
        return x + y


class SpectralBranch(nn.Module):
    """Per-channel log-magnitude FFT followed by a small CNN over (channel, freq).

    Input  : (B, C, T)
    Output : (B, d_spec)
    """

    def __init__(
        self, in_ch: int = 19, n_freq_bins: int = 64, hidden: int = 32, d_out: int = 64
    ):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        # treat channels as separate input feature dim, freq as the temporal axis
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden * 2, d_out)

    def forward(self, x):
        # x: (B, C, T) — assumed already z-scored upstream
        spec = _torch_fft.rfft(x, dim=-1)  # (B, C, T//2+1) complex
        mag = spec.abs()
        log_mag = torch.log1p(mag)
        log_mag = log_mag[..., : self.n_freq_bins]  # keep low-freq bins
        z = self.net(log_mag).flatten(1)  # (B, hidden*2)
        return self.proj(z)  # (B, d_out)


class SelfAttnBlock(nn.Module):
    """Pre-LN self-attention + FFN. PyTorch-1.7-safe (manual (L,B,d) transpose)."""

    def __init__(
        self, d_model: int, n_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, L, d)
        h = self.norm1(x).transpose(0, 1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a.transpose(0, 1)
        x = x + self.ff(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):

    def __init__(
        self, d_model: int, n_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm_f = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        q_n = self.norm_q(q).transpose(0, 1)
        k_n = self.norm_k(kv).transpose(0, 1)
        a, _ = self.attn(q_n, k_n, k_n, need_weights=False)
        q = q + a.transpose(0, 1)
        q = q + self.ff(self.norm_f(q))
        return q


class PerNodeTimeBackboneV6(nn.Module):

    def __init__(
        self, n_nodes: int = 90, hidden_per_node: int = 12, out_per_node: int = 24
    ):
        super().__init__()
        N, h, o = n_nodes, hidden_per_node, out_per_node
        self.net = nn.Sequential(
            nn.Conv1d(N, N * h, kernel_size=7, stride=4, padding=3, groups=N),
            nn.BatchNorm1d(N * h),
            nn.LeakyReLU(0.2),
            nn.Conv1d(N * h, N * h, kernel_size=5, stride=4, padding=2, groups=N),
            nn.BatchNorm1d(N * h),
            nn.LeakyReLU(0.2),
            nn.Conv1d(N * h, N * o, kernel_size=3, stride=2, padding=1, groups=N),
            nn.BatchNorm1d(N * o),
            nn.LeakyReLU(0.2),
        )
        self.N = N
        self.out_per_node = o

    def forward(self, src):  # (B, N, T)
        return self.net(src)  # (B, N*o, T_short)


# ---------------------------------------------------------------------------
# v6 main model
# ---------------------------------------------------------------------------
class NeuroDynEEG(nn.Module):
    def __init__(
        self,
        pick_params,
        in_nodes: int = 19,
        T: int = 1280,
        hidden_scalp: int = 128,
        d_model: int = 192,
        n_heads: int = 6,
        n_self_attn: int = 2,
        n_cross_attn: int = 2,
        ff_dim: int = 384,
        dropout: float = 0.1,
        pinv_out_per_node: int = 24,
        pinv_hidden_per_node: int = 12,
        node_emb_dim: int = 32,
        spec_freq_bins: int = 201,
        spec_hidden: int = 32,
        d_spec: int = 64,
        time_pool: int = 8,
        head_hidden: int = 128,
        lead_field: Optional[np.ndarray] = None,
        lead_field_pinv: Optional[np.ndarray] = None,
        pinv_ridge: float = 1e-3,
        source_nodes: int = 90,
        param_config: Optional[dict] = None,
    ):
        super().__init__()
        if param_config is None:
            param_config = {k: {"bounds": v, "shape": (1,) if k == "delay_ms" else (source_nodes,)}
                            for k, v in TARGET_BOUNDS.items()}
        self.param_config = param_config
        self.pick_params = list(pick_params)
        self.N = int(source_nodes)
        self.d_model = int(d_model)
        self.time_pool = int(time_pool)

        # ---------- 1. lead-field pseudo-inverse buffer ----------
        if lead_field_pinv is None:
            if lead_field is None:
                raise ValueError(
                    "model requires lead_field (channels, regions) or lead_field_pinv (regions, channels)."
                )
            lead_field_pinv = compute_lead_field_pinv(lead_field, ridge=pinv_ridge)
        L_pinv = torch.from_numpy(np.asarray(lead_field_pinv, dtype=np.float32))
        if L_pinv.shape != (self.N, in_nodes) or not torch.isfinite(L_pinv).all():
            raise ValueError(f"L_pinv must be finite with shape ({self.N}, {in_nodes})")
        self.register_buffer("L_pinv", L_pinv)

        # ---------- 2. scalp time branch ----------
        self.scalp_backbone = MultiHeadConv(in_nodes, hidden=hidden_scalp, stride=2)
        self.scalp_res1 = DilatedResBlock(
            hidden_scalp, kernel=5, dilation=2, dropout=dropout
        )
        self.scalp_res2 = DilatedResBlock(
            hidden_scalp, kernel=5, dilation=4, dropout=dropout
        )
        # project scalp time tokens to d_model for cross-attention KV
        self.scalp_proj = nn.Conv1d(hidden_scalp, d_model, kernel_size=1)

        # ---------- 3. source pseudo-inverse branch ----------
        self.source_backbone = PerNodeTimeBackboneV6(
            n_nodes=self.N,
            hidden_per_node=pinv_hidden_per_node,
            out_per_node=pinv_out_per_node,
        )
        self.pool_source = nn.AdaptiveAvgPool1d(1)

        # ---------- 4. spectral branch ----------
        self.spec_branch = SpectralBranch(
            in_ch=in_nodes, n_freq_bins=spec_freq_bins, hidden=spec_hidden, d_out=d_spec
        )

        # ---------- 5. node token construction & node self-attention ----------
        self.node_emb = nn.Parameter(torch.randn(self.N, node_emb_dim) * 0.02)
        node_in_dim = pinv_out_per_node + node_emb_dim
        self.node_proj = nn.Linear(node_in_dim, d_model)
        self.sc_blocks = nn.ModuleList(
            [
                SelfAttnBlock(d_model, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
                for _ in range(n_self_attn)
            ]
        )

        # ---------- 6. scalp <-> source cross-attention ----------
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttnBlock(d_model, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
                for _ in range(n_cross_attn)
            ]
        )
        # extra learnable global query (delay_scale 等 sample-level scalar 用它)
        self.global_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # ---------- 7. dynamic time-token sequence length (for sinusoidal PE) ----------
        self.T_kv = self.time_pool
        self.register_buffer("time_pe", sinusoidal_pe(self.T_kv, d_model))
        self.kv_pool = nn.AdaptiveAvgPool1d(self.time_pool)

        # ---------- 8. per-parameter heads ----------
        self.heads = nn.ModuleDict()
        self.constrains = nn.ModuleDict()
        # node head sees (d_model from cross-attn) + (d_spec broadcast)
        node_feat_total = d_model + d_spec
        scalar_feat_total = d_model + d_spec
        for k in self.pick_params:
            cfg = param_config[k]
            if k == "delay_ms":
                self.heads[k] = nn.Sequential(
                    nn.Linear(scalar_feat_total, head_hidden),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, 1),
                )
            else:
                self.heads[k] = nn.Sequential(
                    nn.Linear(node_feat_total, head_hidden),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, head_hidden),
                    nn.LeakyReLU(0.2),
                    nn.Linear(head_hidden, 1),
                )
            low, high = cfg["bounds"]
            if k.startswith("tau_i"):
                self.constrains[k] = ConstrainedLayer(low, high, k=12 / 5)
            else:
                self.constrains[k] = BoundSigmoid(low, high)

    # ------------------------------------------------------------------
    def _scalp_features(self, x):
        """x: (B, C, T) -> kv_tokens (B, T_kv, d_model)."""
        f = self.scalp_backbone(x)  # (B, hidden_scalp, T/2)
        f = self.scalp_res1(f)
        f = self.scalp_res2(f)  # large RF, still T/2
        f = self.kv_pool(f)  # (B, hidden_scalp, T_kv)
        f = self.scalp_proj(f)  # (B, d_model, T_kv)
        return f.transpose(1, 2)  # (B, T_kv, d_model)

    def _source_features(self, x):
        """x: (B, C, T) -> per-node pinv features (B, N, pinv_out_per_node)."""
        src_pinv = torch.einsum("ij,bjt->bit", self.L_pinv, x)  # (B, N, T)
        f = self.source_backbone(src_pinv)  # (B, N*o, T_short)
        f = self.pool_source(f).squeeze(-1)  # (B, N*o)
        B = x.shape[0]
        return f.view(B, self.N, -1)  # (B, N, o)

    def _build_node_tokens(self, z_pinv):
        """concat per-node pinv feature with learnable node embedding -> project to d_model."""
        B = z_pinv.shape[0]
        emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, e)
        z = torch.cat([z_pinv, emb], dim=-1)  # (B, N, o+e)
        return self.node_proj(z)  # (B, N, d_model)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """x: (B, C, T) -> physical regional (B, N) and delay (B, 1) heads."""
        if x.ndim != 3 or x.shape[1] != self.L_pinv.shape[1] or x.shape[-1] < 32:
            raise ValueError("EEG must have shape batch x configured channels x time (>=32 samples)")
        B = x.shape[0]

        # --- 1) scalp time tokens (KV memory) ---
        kv = self._scalp_features(x)  # (B, T_kv, d_model)
        kv = kv + self.time_pe.unsqueeze(0)  # broadcast (1,T_kv,d)

        # --- 2) source per-node features (Tikhonov inversion) ---
        z_pinv = self._source_features(x)  # (B, N, o)

        # --- 3) spectral global feature ---
        z_spec = self.spec_branch(x)  # (B, d_spec)

        # --- 4) node tokens + node self-attention ---
        node_tok = self._build_node_tokens(z_pinv)  # (B, N, d_model)
        # append global query for sample-level scalars
        glob_tok = self.global_query.unsqueeze(0).expand(B, -1, -1)  # (B,1,d)
        all_tok = torch.cat([node_tok, glob_tok], dim=1)  # (B, N+1, d)
        for blk in self.sc_blocks:
            all_tok = blk(all_tok)

        # --- 5) cross-attention: node tokens look at scalp time memory ---
        for blk in self.cross_blocks:
            all_tok = blk(all_tok, kv)

        node_final = all_tok[:, : self.N, :]  # (B, N, d_model)
        glob_final = all_tok[:, self.N, :]  # (B, d_model)

        # --- 6) per-param heads ---
        z_spec_n = z_spec.unsqueeze(1).expand(-1, self.N, -1)  # (B, N, d_spec)
        node_feat = torch.cat([node_final, z_spec_n], dim=-1)  # (B, N, d+d_spec)
        scalar_feat = torch.cat([glob_final, z_spec], dim=-1)  # (B, d+d_spec)

        raw = {}
        for k in self.pick_params:
            if k == "delay_ms":
                raw[k] = self.heads[k](scalar_feat).view(B, 1)
            else:
                raw[k] = self.heads[k](node_feat).squeeze(-1)  # (B, N)

        out = {}
        for k in self.pick_params:
            if k.startswith("tau_i"):
                ek = k.replace("tau_i", "tau_e")
                out[k] = self.constrains[k](out[ek], raw[k])
            else:
                out[k] = self.constrains[k](raw[k])
        return out


def create_model(n_regions, n_channels, n_times, leadfield, *, seed=0, profile="compact",
                 sampling_rate_hz=256, spectral_max_hz=40):
    """Build the released MB-SSI topology with configurable atlas and montage.

    ``paper`` uses the released model defaults; ``compact`` reduces widths and
    attention depth for a single 8 GiB GPU. Neither includes trained weights.
    The spectral branch retains frequencies through spectral_max_hz using the
    supplied sampling rate and training window length. Changing these at
    inference is a domain shift and requires rebuilding/retraining the model.
    """
    if min(n_regions, n_channels) < 1 or n_times < 32:
        raise ValueError("positive region/channel counts and >=32 samples required")
    if np.shape(leadfield) != (n_channels, n_regions):
        raise ValueError("leadfield shape must match channels x regions")
    if profile not in ("compact", "paper"):
        raise ValueError("profile must be 'compact' or 'paper'")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if not np.isfinite(spectral_max_hz) or not 0 < spectral_max_hz <= sampling_rate_hz / 2:
        raise ValueError("spectral_max_hz must be positive and no greater than Nyquist")
    frequency_bins = min(n_times // 2 + 1, math.floor(spectral_max_hz * n_times / sampling_rate_hz) + 1)
    torch.manual_seed(seed)
    options = {} if profile == "paper" else dict(
        hidden_scalp=32, d_model=48, n_heads=4, n_self_attn=1,
        n_cross_attn=1, ff_dim=96, pinv_out_per_node=8,
        pinv_hidden_per_node=4, node_emb_dim=8, spec_hidden=16,
        d_spec=16, head_hidden=32,
    )
    model = NeuroDynEEG(TARGET_NAMES, in_nodes=n_channels, T=n_times,
                        source_nodes=n_regions, lead_field=leadfield,
                        spec_freq_bins=frequency_bins, **options)
    model.profile = profile
    model.sampling_rate_hz = float(sampling_rate_hz)
    model.spectral_max_hz = float(spectral_max_hz)
    model.config = dict(n_regions=int(n_regions), n_channels=int(n_channels),
                        n_times=int(n_times), seed=int(seed), profile=profile,
                        sampling_rate_hz=float(sampling_rate_hz),
                        spectral_max_hz=float(spectral_max_hz))
    return model


def pack_parameters(parameters):
    """Physical parameter dict -> (..., 11*N+1), family-major order.

    Regional arrays must share shape (..., N); delay accepts shape (...) or
    (..., 1). A single simulator parameter draw therefore returns a 1D vector.
    """
    arrays = [np.asarray(parameters[k], dtype=np.float32) for k in PARAMETER_NAMES]
    if not arrays[0].ndim or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("regional parameter arrays must have identical (..., N) shapes")
    shape = arrays[0].shape[:-1] + (1,)
    delay = np.asarray(parameters["delay_ms"], dtype=np.float32)
    if delay.shape == shape[:-1]:
        delay = delay[..., None]
    if delay.shape != shape:
        raise ValueError("delay_ms must have the batch shape, optionally followed by 1")
    packed = np.concatenate([*arrays, delay], axis=-1)
    if not np.isfinite(packed).all():
        raise ValueError("parameters must be finite")
    return packed


def unpack_parameters(packed, n_regions=None):
    """Inverse of pack_parameters; delay retains a final singleton axis."""
    packed = np.asarray(packed, dtype=np.float32)
    if not packed.ndim:
        raise ValueError("packed parameters need a final parameter dimension")
    inferred, remainder = divmod(packed.shape[-1] - 1, len(PARAMETER_NAMES))
    if remainder or inferred < 1 or (n_regions is not None and n_regions != inferred):
        raise ValueError("packed length must be 11*n_regions+1")
    result = {k: packed[..., i * inferred:(i + 1) * inferred]
              for i, k in enumerate(PARAMETER_NAMES)}
    result["delay_ms"] = packed[..., -1:]
    return result


def prepare_eeg(eeg):
    """Per-window common-average reference, demean and global z-score.

    This is shared between synthetic training and measured inference. It
    intentionally removes absolute amplitude; parameters are inferred from
    temporal and spatial/spectral structure, not EEG amplitude calibration.
    A single scale per window preserves relative channel power and reference.
    """
    values = np.asarray(eeg, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] < 32 or not np.isfinite(values).all():
        raise ValueError("EEG must be finite batch x channels x time, time >=32")
    values = values - values.mean(axis=-1, keepdims=True)
    if values.shape[1] > 1:
        values = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=(1, 2), keepdims=True)
    return values / np.maximum(scale, 1e-12)


def weighted_mae(prediction, target, weights=None):
    """Paper Table S3 weights applied to family-wise span-normalized MAE.

    Normalization is an explicit G4 adaptation: physical milliseconds must not
    silently receive ten times the weight of the paper's stored ms/10 targets.
    No out-of-bound target masking is applied: invalid banks fail validation.
    """
    weights = PAPER_WEIGHTS if weights is None else weights
    return sum(float(weights[k]) * (prediction[k] - target[k]).abs().mean() / (hi - lo)
               for k, (lo, hi) in TARGET_BOUNDS.items()) / sum(weights[k] for k in TARGET_NAMES)


def parameter_metrics(prediction, target):
    """Per-family physical MAE, normalized MAE, and pooled recovery R-squared."""
    metrics = {}
    for k, (low, high) in TARGET_BOUNDS.items():
        p, t = np.asarray(prediction[k]), np.asarray(target[k])
        if p.shape != t.shape:
            raise ValueError(f"prediction/target shape mismatch for {k}")
        error = p.astype(np.float64) - t
        variance = np.sum((t - t.mean()) ** 2)
        mae = float(np.mean(np.abs(error)))
        metrics[k] = {"mae": mae, "normalized_mae": mae / (high - low),
                      "r2": float(1 - np.sum(error ** 2) / variance) if variance > 0 else None}
    return metrics


def _device(device):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


def predict(model, eeg, *, batch_size=8, device="auto"):
    """Return physical-unit NumPy parameter arrays for raw EEG windows."""
    if batch_size < 1 or len(eeg) < 1:
        raise ValueError("prediction requires positive batch size and nonempty EEG")
    device = _device(device)
    model.to(device).eval()
    chunks = {k: [] for k in TARGET_NAMES}
    with torch.inference_mode():
        for start in range(0, len(eeg), batch_size):
            batch = torch.from_numpy(prepare_eeg(eeg[start:start + batch_size])).to(device)
            result = model(batch)
            for k in TARGET_NAMES:
                chunks[k].append(result[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in chunks.items()}


def train_from_arrays(model, eeg, targets, *, train_indices, validation_indices,
                      groups=None, epochs=150, batch_size=8, lr=1e-3,
                      weight_decay=1e-5, patience=20, device="auto", seed=0,
                      progress=None):
    """Adam + weighted normalized MAE; restore the lowest validation-loss state.

    Explicit indices prevent hidden splitting; groups optionally guard subject
    or simulation-family leakage. Final-test indices must be held out by the
    caller. EEG may be a memory-mapped array; only minibatches move to the GPU.
    Returns JSON-ready loss histories and metrics; mutates the supplied model.
    Optional progress(record) receives epoch losses and is_best after each
    epoch, allowing the caller to log or atomically checkpoint the model.
    """
    if epochs < 1 or batch_size < 1 or patience < 1 or lr <= 0:
        raise ValueError("epochs, batch_size, patience and lr must be positive")
    train = np.asarray(train_indices, dtype=int)
    valid = np.asarray(validation_indices, dtype=int)
    for indices in (train, valid):
        if (indices.ndim != 1 or not len(indices) or indices.min() < 0
                or indices.max() >= len(eeg) or len(np.unique(indices)) != len(indices)):
            raise ValueError("splits must contain unique, valid, nonempty 1D indices")
    if np.intersect1d(train, valid).size:
        raise ValueError("training and validation indices overlap")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(eeg),):
            raise ValueError("groups must contain one group per EEG window")
        if np.intersect1d(groups[train], groups[valid]).size:
            raise ValueError("training and validation groups overlap")
    if isinstance(targets, dict):
        targets = pack_parameters(targets)
    target = unpack_parameters(targets, model.N)
    if np.shape(targets) != (len(eeg), len(PARAMETER_NAMES) * model.N + 1):
        raise ValueError("one packed parameter vector is required per EEG window")
    for k, (low, high) in TARGET_BOUNDS.items():
        values = target[k]
        if not np.isfinite(values).all() or np.any(values < low - 1e-6) or np.any(values > high + 1e-6):
            raise ValueError(f"invalid target bounds for {k}")
    for suffix in ("1_ms", "2_ms"):
        e, i = target["tau_e" + suffix], target["tau_i" + suffix]
        if np.any(i < e - 1e-5) or np.any(i > 2.4 * e + 1e-5):
            raise ValueError("inhibitory target violates the joint time-constant prior")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "validation_loss": [], "best_epoch": None,
               "objective": "paper_weights_span_normalized_mae", "seed": seed,
               "preprocessing": "temporal_demean_common_average_global_window_std",
               "device": str(device), "train_count": len(train), "validation_count": len(valid)}
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        for training, indices in ((True, rng.permutation(train)), (False, valid)):
            model.train(training)
            total = 0.0
            for start in range(0, len(indices), batch_size):
                subset = indices[start:start + batch_size]
                # BatchNorm cannot estimate variance from a singleton final
                # time token. Keep gradients but use running stats for B=1.
                model.train(training and len(subset) > 1)
                x = torch.from_numpy(prepare_eeg(eeg[subset])).to(device)
                y = {k: torch.from_numpy(v[subset]).to(device) for k, v in target.items()}
                with torch.set_grad_enabled(training):
                    loss = weighted_mae(model(x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite inverse-model loss")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                total += float(loss.detach()) * len(subset)
            history["train_loss" if training else "validation_loss"].append(total / len(indices))
        validation_loss = history["validation_loss"][-1]
        if validation_loss < best_loss:
            best_loss, stale = validation_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch + 1
        else:
            stale += 1
        if progress is not None:
            progress({"epoch": epoch + 1, "train_loss": history["train_loss"][-1],
                      "validation_loss": validation_loss, "best_epoch": history["best_epoch"],
                      "is_best": history["best_epoch"] == epoch + 1})
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    prediction = predict(model, eeg[valid], batch_size=batch_size, device=str(device))
    history["validation_metrics"] = parameter_metrics(prediction, {k: v[valid] for k, v in target.items()})
    return history
