"""PD-SSM layers: multi-head SSMs whose transition is a dictionary entry (hardened to a
permutation-like push) times a diagonal, x_t = P_t (D_t * x_{t-1}) + b_t.

FlashPDSSM               flame's PDSSM_Layer_Clean: same functional form and parameter names
                         (fla's ShortConvolution and FusedRMSNormGated re-implemented in PyTorch
                         with the same parameters). Full-sequence forward only.
ConstitutiveFlashPDSSM   the same layer driven by a step size nu_t >= 0 per position (e.g. the
                         strain-increment magnitude), so that nu_t = 0 leaves the state unchanged:
                             D_t -> D_t ** nu_t,   b_t -> nu_t * b_t,   P_t -> identity as nu_t -> 0
                         (the identity is an extra dictionary entry with weight 1 - g_t,
                         g_t = 1 - exp(-nu_t / tau)). No short convolution by default: it mixes a
                         fixed number of positions regardless of their size.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import scan


class ShortConvolution(nn.Conv1d):
    """Depthwise causal convolution + SiLU (fla's ShortConvolution, same parameters)."""

    def __init__(self, hidden_size, kernel_size=4, bias=True):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias,
                         padding=kernel_size - 1)

    def forward(self, x):                                   # (B, L, C)
        L = x.shape[1]
        return F.silu(super().forward(x.transpose(1, 2))[..., :L].transpose(1, 2))


class GatedRMSNorm(nn.Module):
    """RMSNorm(x) * weight * silu(gate) (fla's FusedRMSNormGated, swish activation)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, g):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight * F.silu(g.float())).to(x.dtype)


class FlashPDSSM(nn.Module):
    """
    Args:
      embed_size: model width D (input and output size).
      state_size: state size N per head.
      num_heads: H.
      dict_size: K, number of dictionary matrices.
      dict_init: 'glorot', 'diag' or 'gauss_diag_union_<alpha>'.
      conv_size: short-convolution kernel size.
      min_bias, max_bias: the diagonal magnitudes start uniform in (min_bias, max_bias).
      min_min_bias: if >= 0, per-head lower bounds spaced from min_min_bias to min_bias.
      mamba_bias: initialise the magnitudes like Mamba instead.
      short_conv: use the short convolution (flame: True).
    forward(x): x (batch, length, embed_size) -> (batch, length, embed_size).
    """

    def __init__(self, embed_size, state_size=16, num_heads=4, dict_size=6, dict_init="glorot",
                 conv_size=4, min_bias=0.99, min_min_bias=-1.0, max_bias=0.9999, norm_eps=1e-6,
                 mamba_bias=False, short_conv=True):
        super().__init__()
        D, N, H, K = embed_size, state_size, num_heads, dict_size
        self.embed_size, self.state_size, self.num_heads, self.dict_size = D, N, H, K
        self.short_conv = short_conv

        self.X_linear = nn.Linear(D, D + 2 * N * H)
        if short_conv:
            self.x_conv = ShortConvolution(D + N * H, kernel_size=conv_size, bias=True)
        self.B = nn.Linear(D, N * H)

        if mamba_bias:
            A_bar = torch.exp(-torch.arange(1, N + 1, dtype=torch.float32))
            log_dt = torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            y = torch.einsum("h,n->hn", torch.exp(torch.exp(log_dt)), A_bar)
        elif min_min_bias < 0:
            y = torch.rand(H, N) * (max_bias - min_bias) + min_bias
        else:  # as in flame
            lo = torch.linspace(min_min_bias, min_bias, H)
            lo = lo.unsqueeze(-1)
            y = torch.rand(H, N) * (max_bias - lo) + lo      # uniform in [lo_h, max_bias)
        self.D_mag_b = nn.Parameter(torch.log(y / (1 - y)))                       # (H, N)
        self.D_mag_W = nn.Parameter(torch.randn(H, N, D) / math.sqrt(N))           # (H, N, D)
        self.S_weight = nn.Parameter(torch.randn(H, K, D) / math.sqrt(D))          # (H, K, D)

        if dict_init == "glorot":
            A = torch.randn(H, K, N, N) / math.sqrt(N)
        elif dict_init == "diag":
            A = torch.eye(N).repeat(H, K, 1, 1)
        elif dict_init.startswith("gauss_diag_union"):
            alpha = float(dict_init.split("gauss_diag_union_")[1])
            q = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1 - (1 - alpha) / (N - 1))).item()
            A = torch.eye(N).repeat(H, K, 1, 1) + torch.randn(H, K, N, N) / (math.sqrt(2) * q)
        else:
            raise ValueError(f"unknown dict_init {dict_init!r}")
        self.A_dict = nn.Parameter(A)                                               # (H, K, N, N)

        self.norm = GatedRMSNorm(N * H, eps=norm_eps)
        self.o_proj = nn.Linear(N * H, D, bias=False)

    def _inputs(self, x):
        """Selection weights, dictionary, magnitude logits, input term and output gate."""
        Bsz, L, D = x.shape
        H, N = self.num_heads, self.state_size
        xBt, Ct = torch.split(self.X_linear(x), [D + N * H, N * H], dim=-1)
        xBt = self.x_conv(xBt) if self.short_conv else F.silu(xBt)
        x_conv, Bt = torch.split(xBt, [D, N * H], dim=-1)
        sel = F.softmax(torch.einsum("bld,hkd->bhlk", x_conv, self.S_weight), dim=-1)      # (B, H, L, K)
        A_soft = F.softmax(self.A_dict, dim=-2)                                             # (H, K, N, N)
        z = torch.einsum("bld,hnd->bhln", x_conv, self.D_mag_W) + self.D_mag_b.view(1, H, 1, N)
        b = (self.B(x_conv) * Bt).view(Bsz, L, N, H).permute(0, 3, 1, 2)                   # (B, H, L, N)
        return sel, A_soft, z, b, Ct

    def _output(self, A_soft, sel, mag, b, Ct):
        Bsz, H, L, N = b.shape
        states = scan(A_soft.contiguous(), sel.contiguous(),
                      mag.reshape(Bsz * H, L, N).contiguous(), b.reshape(Bsz * H, L, N).contiguous())
        states = states.reshape(Bsz, H, L, N).permute(0, 2, 1, 3).reshape(Bsz, L, N * H)
        return self.o_proj(self.norm(states, Ct))

    def forward(self, x):
        sel, A_soft, z, b, Ct = self._inputs(x)
        return self._output(A_soft, sel, torch.sigmoid(z), b, Ct)


class ConstitutiveFlashPDSSM(FlashPDSSM):
    """FlashPDSSM driven by a step size nu (see module docstring).

    Extra args:
      tau_init: initial tau of the identity weight, in units of nu.
      short_conv: default False here.
    forward(x, nu): x (batch, length, embed_size), nu (batch, length) >= 0.
    """

    def __init__(self, embed_size, short_conv=False, tau_init=0.1, **kw):
        super().__init__(embed_size, short_conv=short_conv, **kw)
        self.log_tau = nn.Parameter(torch.full((self.num_heads,), math.log(tau_init)))

    def forward(self, x, nu):
        sel, A_soft, z, b, Ct = self._inputs(x)
        H, N = self.num_heads, self.state_size
        nu = nu.to(z.dtype)
        mag = torch.exp(nu[:, None, :, None] * F.logsigmoid(z))                             # sigmoid(z) ** nu
        b = b * nu[:, None, :, None]
        g = 1.0 - torch.exp(-nu[:, None, :] / torch.exp(self.log_tau)[None, :, None])       # (B, H, L)
        sel = torch.cat([sel * g[..., None], (1.0 - g)[..., None]], dim=-1)                 # + identity entry
        eye = torch.eye(N, device=x.device, dtype=A_soft.dtype).expand(H, 1, N, N)
        A_soft = torch.cat([A_soft, eye], dim=1)                                            # (H, K + 1, N, N)
        return self._output(A_soft, sel, mag, b, Ct)
