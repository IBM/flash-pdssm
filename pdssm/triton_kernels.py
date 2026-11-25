import torch
import triton
import triton.language as tl


# ===========================================================================
# grad wrt the dictionary matrices (A_dict_soft), via segmented outer products
# grad_A[h,k] = sum_{t : k*(u_t)=k} grad_x_t (D_t x_{t-1})^T
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256}, num_warps=8, num_stages=2),
    ],
    key=[]
)
@triton.jit
def seg_outer_kernel_nopack(
        Xr_ptr, Yr_ptr,  # [T, Dh] each (row-major), T = H*N after sort
        ROWPTR_ptr,  # [G+1] int32, group segment starts/ends in the sorted arrays
        C_ptr,  # [G, Dh, Dh] (row-major), fp32
        Dh: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        input_BF16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    k_start = tl.load(ROWPTR_ptr + g)
    k_end = tl.load(ROWPTR_ptr + g + 1)

    dtype_tl = tl.bfloat16 if input_BF16 else tl.float32

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    ldx = Dh;
    ldy = Dh;
    ldc = Dh
    stride_cg = Dh * Dh

    tl.multiple_of(offs_m, 8);
    tl.multiple_of(offs_n, 8)

    k = k_start
    while k < k_end:
        offs_k = k + tl.arange(0, BLOCK_K)
        mk = offs_k < k_end
        mm = offs_m < Dh
        mn = offs_n < Dh

        xr = tl.load(Xr_ptr + offs_k[:, None] * ldx + offs_m[None, :], mask=mk[:, None] & mm[None, :], other=0)
        yr = tl.load(Yr_ptr + offs_k[:, None] * ldy + offs_n[None, :], mask=mk[:, None] & mn[None, :], other=0)

        acc += tl.dot(tl.trans(xr), yr)  # fp16/bf16 -> TC, fp32 accum
        k += BLOCK_K

    base = g * stride_cg
    tl.store(C_ptr + base + offs_m[:, None] * ldc + offs_n[None, :],
             acc.to(dtype_tl), mask=(offs_m < Dh)[:, None] & (offs_n < Dh)[None, :])


@torch.no_grad()
def grad_A_triton_nopack(grad_x_re, Dy_re, hard_idx, K):
    device = grad_x_re.device
    dtype = grad_x_re.dtype
    B, H, L, Dh = grad_x_re.shape
    N = B * L
    G = H * K
    T = H * N

    input_BF16 = (dtype == torch.bfloat16)

    idx = hard_idx.permute(1, 0, 2).contiguous().view(H, N)  # [H, N]

    h_ids = torch.arange(H, device=device).unsqueeze(1).expand(H, N)
    keys = (h_ids * K + idx).reshape(-1)  # [T]
    order = torch.argsort(keys)  # [T]
    keys_sorted = keys[order]

    Xr = grad_x_re.permute(1, 0, 2, 3).reshape(T, Dh).index_select(0, order).to(dtype).contiguous()
    Yr = Dy_re.permute(1, 0, 2, 3).reshape(T, Dh).index_select(0, order).to(dtype).contiguous()

    counts = torch.bincount(keys_sorted, minlength=G)
    rowptr = torch.zeros(G + 1, device=device, dtype=torch.int32)
    rowptr[1:] = counts.cumsum(0).to(torch.int32)

    C = torch.zeros((G, Dh, Dh), device=device, dtype=dtype)

    grid = lambda META: (
        triton.cdiv(Dh, META['BLOCK_M']),
        triton.cdiv(Dh, META['BLOCK_N']),
        G)

    seg_outer_kernel_nopack[grid](Xr, Yr, rowptr, C, Dh, input_BF16=input_BF16)
    return C.view(H, K, Dh, Dh)


# ===========================================================================
# grad wrt the selector logits (sel_soft), gathering through the push index P
# ===========================================================================
@triton.jit
def _grad_sel_soft_triton(
        gx_re_ptr,  # float*  [B,H,L,Dh]
        dy_re_ptr,  # float*  [B,H,L,N]
        sel_idx_ptr,  # int16*  [H,K,Dh]    (m* per (h,k,n))
        out_ptr,  # float*  [B,H,L,K]

        B: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
        Dh: tl.constexpr, K: tl.constexpr, N: tl.constexpr,

        s_gx_b, s_gx_h, s_gx_l, s_gx_m,
        s_dy_b, s_dy_h, s_dy_l, s_dy_n,
        s_sel_h, s_sel_k, s_sel_m,
        s_out_b, s_out_h, s_out_l, s_out_k,

        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr
):
    pid_bhl = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    tmp = pid_bhl
    l = tmp % L
    tmp = tmp // L
    h = tmp % H
    b = tmp // H

    k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K

    gx_re_base = gx_re_ptr + b * s_gx_b + h * s_gx_h + l * s_gx_l
    dy_re_base = dy_re_ptr + b * s_dy_b + h * s_dy_h + l * s_dy_l

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Sum over source columns 'n', looking up the destination row 'm' = P[n].
    for n0 in range(0, Dh, BLOCK_M):
        n_off = n0 + tl.arange(0, BLOCK_M)
        n_mask = n_off < Dh

        dy_re = tl.load(dy_re_base + n_off * s_dy_n, mask=n_mask, other=0.0).to(tl.float32)

        sel_ptrs = (sel_idx_ptr
                    + h * s_sel_h
                    + k_off[:, None] * s_sel_k
                    + n_off[None, :] * s_sel_m)
        m_idx_i16 = tl.load(sel_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        m_idx = m_idx_i16.to(tl.int32)

        gx_re_vals = tl.load(gx_re_base + m_idx * s_gx_m,
                             mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        part = gx_re_vals * dy_re[None, :]
        acc += tl.sum(part, axis=1)

    out_ptrs = (out_ptr
                + b * s_out_b
                + h * s_out_h
                + l * s_out_l
                + k_off * s_out_k)
    tl.store(out_ptrs, acc, mask=k_mask)


def grad_sel_soft_triton(grad_x_re, Dy_re, sel_idx,
                         block_m: int = 128, block_k: int = 8):

    B, H, L, Dh = grad_x_re.shape
    _, _, _, N = Dy_re.shape
    Hs, K, Dh_ = sel_idx.shape
    assert Hs == H and Dh_ == Dh, "sel_idx must be (H,K,Dh)"
    assert sel_idx.dtype in (torch.int16, torch.int32), "sel_idx must be int16/int32"
    assert grad_x_re.is_contiguous()
    assert Dy_re.is_contiguous()
    assert sel_idx.is_contiguous()

    dtype = grad_x_re.dtype
    out = torch.empty((B, H, L, K), dtype=dtype, device=grad_x_re.device)

    grid = (B * H * L, triton.cdiv(K, block_k))
    num_warps = 4 if Dh <= 128 else 8
    num_stages = 2

    _grad_sel_soft_triton[grid](
        grad_x_re, Dy_re, sel_idx, out,
        B, H, L, Dh, K, N,
        *grad_x_re.stride(),
        *Dy_re.stride(),
        *sel_idx.stride(),
        *out.stride(),
        BLOCK_M=block_m, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages
    )
    return out

# ===========================================================================
# v2 gradient kernels: no sorting, no layout copies, any input strides
# ===========================================================================
# grad_A[h,k,m,n] = sum_{b,l : sel[b,h,l]=k} gx[b,h,l,m] * dy[b,h,l,n]
# One program per (h, k, row chunk): it streams its chunk of the B*L rows of head h, zeroes the
# rows that chose another entry, accumulates gx^T dy with tl.dot and atomically adds the N x N
# result. Replaces argsort + two index_select copies + bincount + the segmented kernel.
@triton.jit
def _grad_A_masked_kernel(
        gx_ptr, dy_ptr, sel_ptr, out_ptr,
        L, N, K, ROWS, ROWS_PER_PROG,
        s_gx_b, s_gx_h, s_gx_l, s_gx_n,
        s_dy_b, s_dy_h, s_dy_l, s_dy_n,
        s_sel_b, s_sel_h, s_sel_l,
        NP: tl.constexpr, BLOCK_T: tl.constexpr, IEEE: tl.constexpr,
):
    hk = tl.program_id(0)
    chunk = tl.program_id(1)
    h = hk // K
    k = hk % K
    offs_n = tl.arange(0, NP)
    mn = offs_n < N
    acc = tl.zeros((NP, NP), dtype=tl.float32)
    r0 = chunk * ROWS_PER_PROG
    r_end = tl.minimum(r0 + ROWS_PER_PROG, ROWS)
    for r in range(r0, r_end, BLOCK_T):
        rows = r + tl.arange(0, BLOCK_T)
        mr = rows < r_end
        b = rows // L
        l = rows % L
        s = tl.load(sel_ptr + b * s_sel_b + h * s_sel_h + l * s_sel_l, mask=mr, other=-1)
        keep = mr & (s == k)
        gx = tl.load(gx_ptr + (b * s_gx_b + h * s_gx_h + l * s_gx_l)[:, None] + offs_n[None, :] * s_gx_n,
                     mask=keep[:, None] & mn[None, :], other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + (b * s_dy_b + h * s_dy_h + l * s_dy_l)[:, None] + offs_n[None, :] * s_dy_n,
                     mask=keep[:, None] & mn[None, :], other=0.0).to(tl.float32)
        if IEEE:
            acc += tl.dot(tl.trans(gx), dy, input_precision="ieee")
        else:
            acc += tl.dot(tl.trans(gx), dy)
    base = out_ptr + hk * N * N
    tl.atomic_add(base + offs_n[:, None] * N + offs_n[None, :], acc, mask=mn[:, None] & mn[None, :])


@torch.no_grad()
def grad_A_masked(gx, dy, hard_idx, K, ieee=False, rows_per_prog=None):
    """gx, dy (B,H,L,N) any strides; hard_idx (B,H,L). Returns (H,K,N,N) fp32."""
    B, H, L, N = gx.shape
    rows = B * L
    if rows_per_prog is None:
        # about 4 programs per SM in total
        sms = torch.cuda.get_device_properties(gx.device).multi_processor_count if gx.is_cuda else 132
        n_chunks = max(1, min(triton.cdiv(rows, 256), triton.cdiv(4 * sms, H * K)))
        rows_per_prog = triton.cdiv(rows, n_chunks)
    BLOCK_T = 64
    rows_per_prog = triton.cdiv(rows_per_prog, BLOCK_T) * BLOCK_T
    out = torch.zeros(H, K, N, N, device=gx.device, dtype=torch.float32)
    NP = max(16, triton.next_power_of_2(N))
    grid = (H * K, triton.cdiv(rows, rows_per_prog))
    _grad_A_masked_kernel[grid](
        gx, dy, hard_idx, out, L, N, K, rows, rows_per_prog,
        *gx.stride(), *dy.stride(), *hard_idx.stride(),
        NP=NP, BLOCK_T=BLOCK_T, IEEE=ieee, num_warps=4)
    return out


# grad_sel[b,h,l,k] = sum_n gx[b,h,l, P[h,k,n]] * dy[b,h,l,n]
# One program per (b*h, block of BLOCK_L positions), all K entries; any strides.
@triton.jit
def _grad_sel_tiled_kernel(
        gx_ptr, dy_ptr, p_ptr, out_ptr,
        H, L, N,
        s_gx_b, s_gx_h, s_gx_l, s_gx_n,
        s_dy_b, s_dy_h, s_dy_l, s_dy_n,
        s_p_h, s_p_k, s_p_n,
        s_o_b, s_o_h, s_o_l, s_o_k,
        K: tl.constexpr, NP: tl.constexpr, BLOCK_L: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    offs_l = tl.program_id(1) * BLOCK_L + tl.arange(0, BLOCK_L)
    ml = offs_l < L
    offs_n = tl.arange(0, NP)
    mn = offs_n < N
    m2 = ml[:, None] & mn[None, :]
    dy = tl.load(dy_ptr + b * s_dy_b + h * s_dy_h + offs_l[:, None] * s_dy_l + offs_n[None, :] * s_dy_n,
                 mask=m2, other=0.0).to(tl.float32)
    gx_base = gx_ptr + b * s_gx_b + h * s_gx_h + offs_l[:, None] * s_gx_l
    o_base = out_ptr + b * s_o_b + h * s_o_h + offs_l * s_o_l
    for k in tl.static_range(K):
        p = tl.load(p_ptr + h * s_p_h + k * s_p_k + offs_n * s_p_n, mask=mn, other=0).to(tl.int32)
        g = tl.load(gx_base + p[None, :] * s_gx_n, mask=m2, other=0.0).to(tl.float32)
        tl.store(o_base + k * s_o_k, tl.sum(g * dy, axis=1), mask=ml)


@torch.no_grad()
def grad_sel_tiled(gx, dy, A_col_argmax, block_l=64):
    """gx, dy (B,H,L,N) any strides; A_col_argmax (H,K,N). Returns (B,H,L,K) fp32."""
    B, H, L, N = gx.shape
    K = A_col_argmax.shape[1]
    out = torch.empty(B, H, L, K, device=gx.device, dtype=torch.float32)
    NP = max(16, triton.next_power_of_2(N))
    grid = (B * H, triton.cdiv(L, block_l))
    _grad_sel_tiled_kernel[grid](
        gx, dy, A_col_argmax, out, H, L, N,
        *gx.stride(), *dy.stride(), *A_col_argmax.stride(), *out.stride(),
        K=K, NP=NP, BLOCK_L=block_l, num_warps=4)
    return out


# ===========================================================================
# v3: dictionary AND selector gradients in one pass over gx / dy
# ===========================================================================
# One program per (head h, chunk of the B*L rows). Each row tile of gx and dy is read from memory
# once; the K dictionary-gradient blocks share one dot product through a "replicated" operand
#   Xrep[t, k*NP + m] = gx[t, m] if sel[t] == k else 0      (T x KP*NP)
#   acc  += Xrep^T @ dy                                      (KP*NP x NP) = all K blocks at once
# and the selector gradient re-reads gx at the permuted columns P_k[n] (cache hits):
#   grad_sel[t, k] = sum_n gx[t, P_k[n]] * dy[t, n]
@triton.jit
def _grads_fused_kernel(
        gx_ptr, dy_ptr, sel_ptr, p_ptr, gA_ptr, gS_ptr,
        L, N, ROWS, ROWS_PER_PROG,
        s_gx_b, s_gx_h, s_gx_l, s_gx_n,
        s_dy_b, s_dy_h, s_dy_l, s_dy_n,
        s_sel_b, s_sel_h, s_sel_l,
        s_p_h, s_p_k, s_p_n,
        s_o_b, s_o_h, s_o_l, s_o_k,
        K: tl.constexpr, KP: tl.constexpr, NP: tl.constexpr, BLOCK_T: tl.constexpr, IEEE: tl.constexpr,
):
    h = tl.program_id(0)
    chunk = tl.program_id(1)
    offs_n = tl.arange(0, NP)
    mn = offs_n < N
    cols = tl.arange(0, KP * NP)            # replicated column c = k*NP + m
    c_k = cols // NP
    c_m = cols % NP
    mc = (c_m < N) & (c_k < K)
    acc = tl.zeros((KP * NP, NP), dtype=tl.float32)
    r0 = chunk * ROWS_PER_PROG
    r_end = tl.minimum(r0 + ROWS_PER_PROG, ROWS)
    for r in range(r0, r_end, BLOCK_T):
        rows = r + tl.arange(0, BLOCK_T)
        mr = rows < r_end
        b = rows // L
        l = rows % L
        s = tl.load(sel_ptr + b * s_sel_b + h * s_sel_h + l * s_sel_l, mask=mr, other=-1)
        gx_row = gx_ptr + (b * s_gx_b + h * s_gx_h + l * s_gx_l)
        dy = tl.load(dy_ptr + (b * s_dy_b + h * s_dy_h + l * s_dy_l)[:, None] + offs_n[None, :] * s_dy_n,
                     mask=mr[:, None] & mn[None, :], other=0.0).to(tl.float32)
        # dictionary gradient: all K entries in one dot
        keep = mr[:, None] & mc[None, :] & (s[:, None] == c_k[None, :])
        xrep = tl.load(gx_row[:, None] + c_m[None, :] * s_gx_n, mask=keep, other=0.0).to(tl.float32)
        if IEEE:
            acc += tl.dot(tl.trans(xrep), dy, input_precision="ieee")
        else:
            acc += tl.dot(tl.trans(xrep), dy)
        # selector gradient
        o_row = gS_ptr + b * s_o_b + h * s_o_h + l * s_o_l
        for k in tl.static_range(K):
            p = tl.load(p_ptr + h * s_p_h + k * s_p_k + offs_n * s_p_n, mask=mn, other=0).to(tl.int32)
            g = tl.load(gx_row[:, None] + p[None, :] * s_gx_n, mask=mr[:, None] & mn[None, :], other=0.0).to(tl.float32)
            tl.store(o_row + k * s_o_k, tl.sum(g * dy, axis=1), mask=mr)
    # gA is (H, K, N, N) contiguous: row c = k*NP + m of acc -> gA[h, k, m, :]
    dst = gA_ptr + h * K * N * N + (c_k * N * N + c_m * N)[:, None] + offs_n[None, :]
    tl.atomic_add(dst, acc, mask=mc[:, None] & mn[None, :])


@torch.no_grad()
def grads_fused(gx, dy, hard_idx, A_col_argmax, ieee=False, rows_per_prog=None):
    """gx, dy (B,H,L,N) any strides; hard_idx (B,H,L); A_col_argmax (H,K,N).
    Returns grad_A (H,K,N,N) and grad_sel (B,H,L,K), both fp32."""
    B, H, L, N = gx.shape
    K = A_col_argmax.shape[1]
    rows = B * L
    BLOCK_T = 32
    if rows_per_prog is None:
        sms = torch.cuda.get_device_properties(gx.device).multi_processor_count if gx.is_cuda else 132
        n_chunks = max(1, min(triton.cdiv(rows, 4 * BLOCK_T), triton.cdiv(4 * sms, H)))
        rows_per_prog = triton.cdiv(rows, n_chunks)
    rows_per_prog = triton.cdiv(rows_per_prog, BLOCK_T) * BLOCK_T
    gA = torch.zeros(H, K, N, N, device=gx.device, dtype=torch.float32)
    gS = torch.empty(B, H, L, K, device=gx.device, dtype=torch.float32)
    NP = max(16, triton.next_power_of_2(N))
    KP = triton.next_power_of_2(K)
    grid = (H, triton.cdiv(rows, rows_per_prog))
    _grads_fused_kernel[grid](
        gx, dy, hard_idx, A_col_argmax, gA, gS, L, N, rows, rows_per_prog,
        *gx.stride(), *dy.stride(), *hard_idx.stride(), *A_col_argmax.stride(), *gS.stride(),
        K=K, KP=KP, NP=NP, BLOCK_T=BLOCK_T, IEEE=ieee, num_warps=8)
    return gA, gS
