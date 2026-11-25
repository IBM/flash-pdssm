"""The PD-SSM scan  x_t = P_t (D_t * x_{t-1}) + b_t,  x_{-1} = 0.

P_t is the dictionary entry picked by argmax of the selection weights, with each column
hardened to one-hot (argmax over rows). Gradients reach the soft dictionary and the soft
selection weights by a straight-through estimator.

`scan` uses the CUDA kernel (forward and reverse scan) and Triton (dictionary / selection
gradients). `reference_scan` is a sequential PyTorch version with the same forward values and
the same gradients, for CPUs and for testing.
"""
import os

import torch
import torch.nn.functional as F

try:
    from . import _C
except ImportError:  # built without the CUDA extension
    _C = None


class _ScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_dict_soft, sel_soft, D_re, b_re):
        B, H, L, K = sel_soft.shape
        Dh = A_dict_soft.shape[-1]
        out_dtype = A_dict_soft.dtype

        # for each column n, the row it pushes to; and the selected dictionary entry per step
        A_col_argmax = A_dict_soft.argmax(dim=-2).to(torch.int16)   # (H, K, Dh)
        hard_idx = sel_soft.argmax(dim=-1)                          # (B, H, L)
        A_cols = A_col_argmax.unsqueeze(0).unsqueeze(2)             # (1, H, 1, K, Dh)
        gather_idx = hard_idx.unsqueeze(-1).unsqueeze(-1)           # (B, H, L, 1, 1)
        P_int = torch.take_along_dim(A_cols, gather_idx, dim=3).squeeze(3)
        P_int = P_int.reshape(B * H, L, Dh).contiguous()

        # the kernel reads D and b with one dtype (D's): under bf16 autocast D (a sigmoid of an
        # fp32 bias) is fp32 while b may be bf16, so cast b
        ctx.b_dtype = b_re.dtype
        b_out_re = _C.scan_forward_compact(P_int, D_re.contiguous(), b_re.to(D_re.dtype).contiguous())
        ctx.save_for_backward(P_int, D_re, b_out_re, hard_idx, A_col_argmax)
        ctx.shapes = (B, H, L, K, Dh)
        return b_out_re.to(out_dtype)

    @staticmethod
    def backward(ctx, gb_out_re):
        P_int, D_re, x_re, hard_idx, A_col_argmax = ctx.saved_tensors
        B, H, L, K, Dh = ctx.shapes
        dtype = D_re.dtype
        gb_r, gD_re, Dy_re = _C.scan_backward_compact(
            P_int, D_re.contiguous(), gb_out_re.to(dtype).contiguous(), x_re.to(dtype).contiguous())
        grad_x_re = gb_r.reshape(B, H, L, -1)
        Dy_re = Dy_re.reshape(B, H, L, Dh)
        grad_A, grad_sel = dict_and_selector_grads(grad_x_re, Dy_re, hard_idx, A_col_argmax, K)
        return grad_A, grad_sel, gD_re.to(dtype), gb_r.to(ctx.b_dtype)


# ---------------------------------------------------------------------------
# Dictionary / selector gradients of the straight-through surrogate
#   grad_A[h,k,m,n]   = sum_{b,l : k*_{b,h,l} = k} gx[b,h,l,m] * dy[b,h,l,n]
#   grad_sel[b,h,l,k] = sum_n gx[b,h,l, P_k[n]] * dy[b,h,l,n]
# gx = dL/dx_t, dy = D_t * x_{t-1}; P_k[n] = argmax_m A_dict_soft[h,k,m,n].
# Implementations, chosen by PDSSM_GRAD_IMPL (default "triton"):
#   triton       two masked Triton kernels: no sort, no layout copies, any strides (grad_A in TF32);
#                fastest measured on H200 (100M-3B shapes)
#   fused        one Triton kernel, one pass over gx / dy for both gradients (grad_A in TF32);
#                fewer bytes but less parallelism: 0-30% slower than "triton" on H200
#   fused_ieee   the same with grad_A in full fp32
#   triton_ieee  the same with grad_A in full fp32
#   triton_v1    the original kernels (sort by entry + segmented outer products; contiguous)
#   torch        dense PyTorch (reference speed / exactness check)
# ---------------------------------------------------------------------------
GRAD_IMPLS = ("fused", "fused_ieee", "triton", "triton_ieee", "triton_v1", "torch")
_FUSED_BROKEN = []


def grad_impl():
    return os.environ.get("PDSSM_GRAD_IMPL", "triton").lower()


def grad_A_torch(gx, dy, hard_idx, K):
    """(B,H,L,N), (B,H,L,N), (B,H,L) -> (H,K,N,N)."""
    B, H, L, N = gx.shape
    gx_t = gx.permute(1, 3, 0, 2).reshape(H, N, B * L)               # (H, N, BL)
    dy_t = dy.permute(1, 0, 2, 3).reshape(H, B * L, N)               # (H, BL, N)
    sel = hard_idx.permute(1, 0, 2).reshape(H, 1, B * L)
    out = gx.new_empty(H, K, N, N)
    for k in range(K):
        out[:, k] = torch.bmm(gx_t * (sel == k).to(gx.dtype), dy_t)
    return out


def grad_sel_torch(gx, dy, A_col_argmax):
    """(B,H,L,N), (B,H,L,N), (H,K,N) -> (B,H,L,K)."""
    B, H, L, N = gx.shape
    K = A_col_argmax.shape[1]
    idx = A_col_argmax.long()
    out = gx.new_empty(B, H, L, K)
    for k in range(K):
        g = torch.gather(gx, -1, idx[:, k].view(1, H, 1, N).expand(B, H, L, N))
        out[..., k] = (g * dy).sum(-1)
    return out


def dict_and_selector_grads(gx, dy, hard_idx, A_col_argmax, K, impl=None):
    """gx, dy: (B, H, L, N); the triton / triton_ieee kernels accept any strides."""
    impl = impl or grad_impl()
    if impl in ("fused", "fused_ieee") and not _FUSED_BROKEN:
        from .triton_kernels import grads_fused
        try:
            return grads_fused(gx, dy, hard_idx, A_col_argmax, ieee=impl == "fused_ieee")
        except Exception as e:  # noqa: BLE001  (e.g. triton OutOfResources on a smaller GPU)
            import warnings
            warnings.warn(f"pdssm: fused gradient kernel unavailable ({type(e).__name__}: {e}); "
                          f"using the two-kernel path")
            _FUSED_BROKEN.append(True)
    if impl in ("fused", "fused_ieee"):
        impl = "triton_ieee" if impl == "fused_ieee" else "triton"
    if impl == "torch":
        return grad_A_torch(gx, dy, hard_idx, K), grad_sel_torch(gx, dy, A_col_argmax)
    if impl in ("triton", "triton_ieee"):
        from .triton_kernels import grad_A_masked, grad_sel_tiled
        return (grad_A_masked(gx, dy, hard_idx, K, ieee=impl == "triton_ieee"),
                grad_sel_tiled(gx, dy, A_col_argmax))
    if impl != "triton_v1":
        raise ValueError(f"PDSSM_GRAD_IMPL must be one of {GRAD_IMPLS}, got {impl!r}")
    from .triton_kernels import grad_A_triton_nopack, grad_sel_soft_triton
    gx, dy = gx.contiguous(), dy.contiguous()
    return (grad_A_triton_nopack(gx, dy, hard_idx, K),
            grad_sel_soft_triton(gx, dy, A_col_argmax, block_m=128, block_k=8))


# ---------------------------------------------------------------------------
# Strided scan: (B, H, L, N) views in, (B, H, L, N) view out, no layout copies
# ---------------------------------------------------------------------------
def _is_dense(t):
    """True if t's elements exactly fill its storage span (a permutation of a contiguous layout)."""
    dims = sorted((st, sz) for st, sz in zip(t.stride(), t.shape) if sz != 1)
    expect = 1
    for st, sz in dims:
        if st != expect:
            return False
        expect *= sz
    return True


def _push_indices(A_dict_soft, sel_soft):
    B, H, L, K = sel_soft.shape
    N = A_dict_soft.shape[-1]
    A_col_argmax = A_dict_soft.argmax(dim=-2).to(torch.int16)       # (H, K, N)
    hard_idx = sel_soft.argmax(dim=-1)                              # (B, H, L)
    P = torch.take_along_dim(A_col_argmax[None, :, None], hard_idx[..., None, None], dim=3)
    return P.reshape(B * H, L, N).contiguous(), hard_idx, A_col_argmax


class _ScanStridedFn(torch.autograd.Function):
    """x = scan(A_dict_soft, sel_soft, D, b) on (B, H, L, N) views of any strides.

    D is used in fp32 (cast if needed); b may be fp32 or bf16 and is read in place. x is fp32,
    returned as a (B, H, L, N) view of (B, L, H, N) storage: x.transpose(1, 2).reshape(B, L, H*N)
    is free. The gradient for b comes back with b's strides and dtype.
    """

    @staticmethod
    def forward(ctx, A_dict_soft, sel_soft, D4, b4):
        B, H, L, K = sel_soft.shape
        P, hard_idx, A_col_argmax = _push_indices(A_dict_soft, sel_soft)
        D32 = D4 if D4.dtype == torch.float32 else D4.float()
        bk = b4 if b4.dtype in (torch.float32, torch.bfloat16) else b4.float()
        x = _C.scan_forward_strided(P, D32, bk)
        ctx.save_for_backward(P, D32, x, hard_idx, A_col_argmax)
        ctx.meta = (K, b4.shape, b4.stride(), b4.dtype, D4.dtype, A_dict_soft.dtype, sel_soft.dtype,
                    _is_dense(b4))
        return x

    @staticmethod
    def backward(ctx, gx_out):
        P, D32, x, hard_idx, A_col_argmax = ctx.saved_tensors
        K, b_shape, b_stride, b_dtype, D_dtype, A_dtype, s_dtype, b_dense = ctx.meta
        G = gx_out if gx_out.dtype == torch.float32 else gx_out.float()
        if b_dense:
            gb_buf = torch.empty_strided(b_shape, b_stride, dtype=torch.float32, device=x.device)
        else:
            gb_buf = torch.empty(b_shape, dtype=torch.float32, device=x.device)
        gb, gD, Dy = _C.scan_backward_strided(P, D32, G, x, gb_buf)
        grad_A, grad_sel = dict_and_selector_grads(gb, Dy, hard_idx, A_col_argmax, K)
        return grad_A.to(A_dtype), grad_sel.to(s_dtype), gD.to(D_dtype), gb.to(b_dtype)


def has_strided():
    return _C is not None and hasattr(_C, "scan_forward_strided")


def scan_strided(A_dict_soft, sel_soft, D4, b4, layout=None):
    """(B, H, L, N) views in, fp32 (B, H, L, N) out (stored as (B, L, H, N)). Uses the strided
    CUDA kernels when the extension has them, else the contiguous kernels, else the reference.

    layout (default PDSSM_LAYOUT, else "contig"):
      contig   D and b are made contiguous (B, H, L, N) first: one copy each, but every kernel
               then reads coalesced memory. b stays bf16 (no fp32 cast); x needs no copy.
      strided  D and b are read in place. Saves the copies, but a (B, L, N*H) b is read with
               stride H, which costs more than the copy at 1B+ (measured on H200).
    """
    B, H, L, N = D4.shape
    if D4.is_cuda and has_strided():
        layout = layout or os.environ.get("PDSSM_LAYOUT", "contig")
        if layout == "contig":
            D4, b4 = D4.contiguous(), b4.contiguous()
        elif layout != "strided":
            raise ValueError(f"PDSSM_LAYOUT must be contig or strided, got {layout!r}")
        return _ScanStridedFn.apply(A_dict_soft, sel_soft, D4, b4)
    D = D4.float().reshape(B * H, L, N)
    b = b4.float().reshape(B * H, L, N)
    return scan(A_dict_soft.float(), sel_soft.float(), D, b).float().view(B, H, L, N)


def reference_scan(A_dict_soft, sel_soft, D_re, b_re):
    """Sequential PyTorch scan, same values and gradients as `scan`.

    A_dict_soft (H, K, N, N) column-stochastic; sel_soft (B, H, L, K); D_re, b_re (B*H, L, N).
    Returns (B*H, L, N).
    """
    H, K, N, _ = A_dict_soft.shape
    BH, L, _ = D_re.shape
    A_hard = F.one_hot(A_dict_soft.argmax(dim=-2), N).transpose(-1, -2).to(A_dict_soft.dtype)
    s_hard = F.one_hot(sel_soft.argmax(dim=-1), K).to(sel_soft.dtype)
    A_ste = A_hard + A_dict_soft - A_dict_soft.detach()      # values hard, gradients soft
    s_ste = s_hard + sel_soft - sel_soft.detach()
    M = torch.einsum("bhlk,hkij->bhlij", s_ste, A_ste).reshape(BH, L, N, N)
    x = D_re.new_zeros(BH, N)
    out = []
    for t in range(L):
        x = torch.einsum("bij,bj->bi", M[:, t], D_re[:, t] * x) + b_re[:, t]
        out.append(x)
    return torch.stack(out, dim=1).to(A_dict_soft.dtype)


def scan(A_dict_soft, sel_soft, D_re, b_re):
    """CUDA scan when available for these tensors, otherwise the sequential reference."""
    if _C is not None and D_re.is_cuda:
        return _ScanFn.apply(A_dict_soft, sel_soft, D_re, b_re)
    return reference_scan(A_dict_soft, sel_soft, D_re, b_re)
