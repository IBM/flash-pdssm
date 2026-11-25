"""python tests/test_pdssm.py   (or pytest). The kernel test needs a GPU."""
import torch
import torch.nn.functional as F

from pdssm import ConstitutiveFlashPDSSM, FlashPDSSM, reference_scan, scan


def _inputs(device, B=2, H=3, L=40, K=5, N=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = F.softmax(torch.randn(H, K, N, N, generator=g) * 2, dim=-2)
    sel = F.softmax(torch.randn(B, H, L, K, generator=g) * 2, dim=-1)
    D = torch.rand(B * H, L, N, generator=g) * 0.5 + 0.45
    b = torch.randn(B * H, L, N, generator=g)
    return [t.to(device).requires_grad_() for t in (A, sel, D, b)]


def test_reference_gradients():
    """Straight-through gradients of the reference = the kernel's closed-form backward."""
    A, sel, D, b = _inputs("cpu")
    x = reference_scan(A, sel, D, b)
    gx_out = torch.randn_like(x)
    (x * gx_out).sum().backward()
    H, K, N, _ = A.shape
    BH, L, _ = D.shape
    with torch.no_grad():  # adjoint: lam_t = gx_t + (P_{t+1} diag(D_{t+1}))^T lam_{t+1}
        P = F.one_hot(A.argmax(-2), N).transpose(-1, -2).float()            # (H, K, N, N)
        k = sel.argmax(-1).reshape(BH, L)
        Pt = P[torch.arange(BH)[:, None] % H, k]                             # (BH, L, N, N)
        lam = torch.zeros_like(x)
        nxt = torch.zeros(BH, N)
        for t in reversed(range(L)):
            lam[:, t] = gx_out[:, t] + nxt
            nxt = torch.einsum("bij,bi->bj", Pt[:, t], lam[:, t]) * D[:, t]
        x_prev = torch.cat([torch.zeros(BH, 1, N), x[:, :-1]], 1)
        Dy = D * x_prev
        gA = torch.zeros_like(A)
        gsel = torch.zeros_like(sel)
        for bh in range(BH):
            bb, hh = divmod(bh, H)
            for t in range(L):
                gA[hh, k[bh, t]] += torch.outer(lam[bh, t], Dy[bh, t])
                gsel[bb, hh, t] = torch.einsum("kij,i,j->k", P[hh], lam[bh, t], Dy[bh, t])
    assert torch.allclose(b.grad, lam, atol=1e-4)
    assert torch.allclose(A.grad, gA, atol=1e-4)
    assert torch.allclose(sel.grad, gsel, atol=1e-4)
    gD = torch.einsum("blij,bli->blj", Pt, lam) * x_prev
    assert torch.allclose(D.grad, gD, atol=1e-4)


def test_layers_cpu():
    x, nu = torch.randn(2, 20, 32), torch.rand(2, 20)
    for m, args in [(FlashPDSSM(32, state_size=16, num_heads=2, dict_size=4), (x,)),
                    (ConstitutiveFlashPDSSM(32, state_size=16, num_heads=2, dict_size=4), (x, nu))]:
        y = m(*args)
        y.square().mean().backward()
        assert y.shape == (2, 20, 32) and torch.isfinite(y).all()
        assert all(p.grad is not None for p in m.parameters())


def test_stationarity():
    """ConstitutiveFlashPDSSM: zero steps appended at the end leave the output unchanged."""
    torch.manual_seed(0)
    m = ConstitutiveFlashPDSSM(32, state_size=16, num_heads=2, dict_size=4)
    x, nu = torch.randn(2, 20, 32), torch.rand(2, 20) + 0.5
    x0 = torch.cat([x, x[:, -1:].expand(2, 10, 32)], 1)       # same input, zero step
    nu0 = torch.cat([nu, torch.zeros(2, 10)], 1)
    with torch.no_grad():
        y0 = m(x0, nu0)
    assert (y0[:, 19:] - y0[:, 19:20]).abs().max() < 1e-5


def test_kernel_matches_reference():
    if not torch.cuda.is_available():
        print("no GPU: kernel test skipped")
        return
    from pdssm.ops import _C
    assert _C is not None, "CUDA extension not built"
    outs, grads = [], []
    for fn in (scan, reference_scan):
        A, sel, D, b = _inputs("cuda")
        x = fn(A, sel, D, b)
        torch.manual_seed(1)
        (x * torch.randn_like(x)).sum().backward()
        outs.append(x.detach())
        grads.append([t.grad for t in (A, sel, D, b)])
    assert torch.allclose(outs[0], outs[1], atol=1e-4, rtol=1e-4), (outs[0] - outs[1]).abs().max()
    for name, g0, g1 in zip(["A_dict", "sel", "D", "b"], *grads):
        # Triton uses tf32 for the dictionary gradient: allow ~1e-2 relative error there
        tol = 1e-2 * g1.abs().max().item() + 1e-4
        assert (g0 - g1).abs().max().item() < tol, f"grad {name}: {(g0 - g1).abs().max().item():.2e}"


if __name__ == "__main__":
    for f in [test_reference_gradients, test_layers_cpu, test_stationarity, test_kernel_matches_reference]:
        f()
        print(f"{f.__name__}: ok")
