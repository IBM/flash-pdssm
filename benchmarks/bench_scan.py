"""Speed and correctness of the PD-SSM scan, piece by piece, at training shapes.

    python benchmarks/bench_scan.py                       # all shapes, tiles, grad implementations
    python benchmarks/bench_scan.py --shapes 100m --tiles 32 64 128 256
    python benchmarks/bench_scan.py --out scan.json

For each shape (B, H, L, N, K) it times, in fp32 as in training (the scan runs in fp32 under
bf16 autocast):
  fwd        _C.scan_forward_compact (three CUDA kernels)
  bwd_scan   _C.scan_backward_compact (reverse scan, also writes dD and D*x_{t-1})
  grads      dictionary + selector gradients (PDSSM_GRAD_IMPL)
  old / new / strided   forward + backward as the flame layer calls it: old = layout copies +
             fp32 cast + contiguous kernels; new = scan_strided (contiguous bf16 b, x written in
             the output layout); strided = scan_strided reading D / b in place
for every tile length (PDSSM_TILE_L; "auto" = the built-in choice) and gradient implementation.
First it checks every variant against reference_scan (sequential PyTorch, same surrogate).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pdssm.ops as ops  # noqa: E402

# (B, H, L, N, K) of the PD layers in flame's benchmark presets (micro-batch 4 x 2048)
SHAPES = {
    "100m": (4, 12, 2048, 38, 4),
    "230m": (4, 16, 2048, 38, 4),
    "1b": (4, 32, 2048, 38, 4),
    "500m": (4, 24, 2048, 38, 4),
    "3b": (4, 48, 2048, 38, 4),
    "7b": (4, 64, 2048, 38, 4),
}


def inputs(B, H, L, N, K, dtype=torch.float32, seed=0, grad=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(H, K, N, N, device="cuda", generator=g).softmax(-2).to(dtype)
    s = torch.randn(B, H, L, K, device="cuda", generator=g).softmax(-1).to(dtype)
    D = (0.9 + 0.1 * torch.rand(B * H, L, N, device="cuda", generator=g)).to(dtype)
    b = torch.randn(B * H, L, N, device="cuda", generator=g).to(dtype)
    if grad:
        for t in (A, s, D, b):
            t.requires_grad_(True)
    return A, s, D, b


def timeit(fn, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def layer_inputs(B, H, L, N, K, b_dtype=torch.bfloat16, seed=0):
    """The layouts the flame layer produces: D (B,H,L,N) fp32, b a (B,H,L,N) view of (B,L,N*H)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(H, K, N, N, device="cuda", generator=g).softmax(-2).requires_grad_()
    s = torch.randn(B, H, L, K, device="cuda", generator=g).softmax(-1).requires_grad_()
    D = (0.9 + 0.1 * torch.rand(B, H, L, N, device="cuda", generator=g)).requires_grad_()
    b_flat = torch.randn(B, L, N * H, device="cuda", generator=g).to(b_dtype).requires_grad_()
    return A, s, D, b_flat


def old_path(A, s, D, b_flat):
    """What the flame layer did before: copies to (B*H, L, N), cast b, scan, copy back."""
    B, H, L, N = D.shape
    b4 = b_flat.view(B, L, N, H).permute(0, 3, 1, 2)
    x = ops._ScanFn.apply(A, s, D.reshape(B * H, L, N), b4.reshape(B * H, L, N).to(D.dtype))
    return x.view(B, H, L, N).permute(0, 2, 1, 3).reshape(B, L, H * N)


def new_path(A, s, D, b_flat, layout="contig"):
    B, H, L, N = D.shape
    x = ops.scan_strided(A, s, D, b_flat.view(B, L, N, H).permute(0, 3, 1, 2), layout=layout)
    return x.transpose(1, 2).reshape(B, L, H * N)


def strided_path(A, s, D, b_flat):
    return new_path(A, s, D, b_flat, layout="strided")


def hn_path(A, s, D_blhn, b_flat):
    """flame's state_layout="hn": D and b are (B, L, H, N) storage, read in place (coalesced)."""
    B, L, H, N = D_blhn.shape
    x = ops.scan_strided(A, s, D_blhn.transpose(1, 2), b_flat.view(B, L, H, N).transpose(1, 2), layout="strided")
    return x.transpose(1, 2).reshape(B, L, H * N)


def check(tiles, impls):
    print("-- correctness vs reference_scan (fp32 inputs; strided path also with bf16 b)")
    A, s, D, b = inputs(2, 3, 300, 38, 4, grad=True)
    out = ops.reference_scan(A, s, D, b)
    G = torch.randn_like(out)
    ref = [out.detach()] + list(torch.autograd.grad((out * G).sum(), (A, s, D, b)))
    names = ["x", "dA", "dsel", "dD", "db"]
    ok = True

    def report(tag, got, ref, loose_dA, loose_db=False):
        nonlocal ok
        errs = [((g.float() - r.float()).abs().max() / r.float().abs().max().clamp_min(1e-12)).item()
                for g, r in zip(got, ref)]
        tol = [1e-5, 3e-3 if loose_dA else 1e-5, 1e-5, 1e-5, 8e-3 if loose_db else 1e-5]
        bad = [n for n, e, t in zip(names, errs, tol) if e > t]
        ok &= not bad
        print(f"  {tag:<34}" + "  ".join(f"{n} {e:.1e}" for n, e in zip(names, errs))
              + ("" if not bad else f"   <-- FAIL {bad}"))

    for impl in impls:
        os.environ["PDSSM_GRAD_IMPL"] = impl
        for tile in tiles:
            os.environ["PDSSM_TILE_L"] = str(tile)
            o = ops._ScanFn.apply(A, s, D, b)
            report(f"contiguous {impl} tile {tile}", [o.detach()] + list(torch.autograd.grad((o * G).sum(), (A, s, D, b))),
                   ref, impl in ("triton", "triton_v1", "fused"))
    os.environ.pop("PDSSM_TILE_L", None)
    if ops.has_strided():
        for impl in impls:
            os.environ["PDSSM_GRAD_IMPL"] = impl
            for b_dtype in (torch.float32, torch.bfloat16):
                A2, s2, D2, bf = layer_inputs(2, 3, 300, 38, 4, b_dtype=b_dtype, seed=1)
                B_, H_, L_, N_ = D2.shape
                b4 = bf.view(B_, L_, N_, H_).permute(0, 3, 1, 2)
                r = ops.reference_scan(A2, s2, D2.reshape(B_ * H_, L_, N_), b4.float().reshape(B_ * H_, L_, N_))
                G2 = torch.randn_like(r)
                rr = [r.detach()] + list(torch.autograd.grad((r * G2).sum(), (A2, s2, D2, bf)))
                x = ops.scan_strided(A2, s2, D2, b4)
                xr = x.reshape(B_ * H_, L_, N_)
                got = [xr.detach()] + list(torch.autograd.grad((xr * G2).sum(), (A2, s2, D2, bf)))
                # a bf16 b gets a bf16 gradient: both sides are rounded to bf16 (up to 2^-8 apart)
                report(f"strided {impl} b={str(b_dtype)[6:]}", got, rr, impl in ("triton", "triton_v1", "fused"),
                       loose_db=b_dtype == torch.bfloat16)
    else:
        print("  (extension has no strided kernels: rebuild pdssm)")
    os.environ.pop("PDSSM_GRAD_IMPL", None)
    return ok


def bench(name, shape, tiles, impls, warmup, reps):
    B, H, L, N, K = shape
    rows = []
    A, s, D, b = inputs(B, H, L, N, K)
    hard = s.argmax(-1)
    Acol = A.argmax(-2).to(torch.int16)
    P = torch.take_along_dim(Acol[None, :, None], hard[..., None, None], dim=3).squeeze(3)
    P = P.reshape(B * H, L, N).contiguous()
    La = layer_inputs(B, H, L, N, K)
    D_blhn = La[2].detach().transpose(1, 2).contiguous().requires_grad_()
    Lh = (La[0], La[1], D_blhn, La[3])
    gout = torch.randn(B, L, H * N, device="cuda")
    for tile in tiles + ["auto"]:
        if tile == "auto":
            os.environ.pop("PDSSM_TILE_L", None)
        else:
            os.environ["PDSSM_TILE_L"] = str(tile)
        x = ops._C.scan_forward_compact(P, D, b)
        gx_ = torch.randn_like(x)
        t_fwd = timeit(lambda: ops._C.scan_forward_compact(P, D, b), warmup, reps)
        t_bwd = timeit(lambda: ops._C.scan_backward_compact(P, D, gx_, x), warmup, reps)
        gx, _, dy = ops._C.scan_backward_compact(P, D, gx_, x)
        gx, dy = gx.view(B, H, L, N), dy.view(B, H, L, N)
        for impl in impls:
            os.environ["PDSSM_GRAD_IMPL"] = impl
            t_g = timeit(lambda: ops.dict_and_selector_grads(gx, dy, hard, Acol, K), warmup, reps)

            def run(path, args=La):
                def f():
                    y = path(*args)
                    torch.autograd.grad(y, args, gout)
                return f
            t_old = timeit(run(old_path), warmup, reps)
            t_new = timeit(run(new_path), warmup, reps) if ops.has_strided() else float("nan")
            t_str = timeit(run(strided_path), warmup, reps) if ops.has_strided() else float("nan")
            t_hn = timeit(run(hn_path, Lh), warmup, reps) if ops.has_strided() else float("nan")
            r = dict(shape=name, B=B, H=H, L=L, N=N, K=K, tile=tile, grad_impl=impl, fwd_ms=t_fwd,
                     bwd_scan_ms=t_bwd, grads_ms=t_g, layer_old_ms=t_old, layer_new_ms=t_new,
                     layer_strided_ms=t_str, layer_hn_ms=t_hn)
            rows.append(r)
            print(f"  {name:<5} tile {str(tile):>4} {impl:<11} fwd {t_fwd:6.3f}  bwd_scan {t_bwd:6.3f}  "
                  f"grads {t_g:6.3f}  | layer fwd+bwd: old {t_old:6.3f}  new {t_new:6.3f}  strided {t_str:6.3f}  hn {t_hn:6.3f} ms")
    os.environ.pop("PDSSM_GRAD_IMPL", None)
    os.environ.pop("PDSSM_TILE_L", None)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    ap.add_argument("--tiles", nargs="+", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--impls", nargs="+", default=["triton_v1", "triton", "fused"], choices=list(ops.GRAD_IMPLS))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out")
    a = ap.parse_args()
    if ops._C is None or not torch.cuda.is_available():
        raise SystemExit("needs a GPU and pdssm built with its CUDA extension")
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    ok = check(a.tiles, a.impls)
    rows = []
    print("\n-- timings (median ms per call, one PD layer)")
    for name in a.shapes:
        rows += bench(name, SHAPES[name], a.tiles, a.impls, a.warmup, a.reps)
    print("\n-- per shape: original (tile 256, triton_v1, old layout path) vs best")
    for name in a.shapes:
        rs = [r for r in rows if r["shape"] == name]
        base = next((r for r in rs if r["tile"] == 256 and r["grad_impl"] == "triton_v1"), None)
        paths = ("layer_old_ms", "layer_new_ms", "layer_hn_ms")
        best = min(rs, key=lambda r: min(r[k] for k in paths))
        best_path = min(paths, key=lambda k: best[k])
        auto = [r for r in rs if r["tile"] == "auto"]
        b_ms = best[best_path]
        msg = f"  {name:<5} best: tile {best['tile']}, {best['grad_impl']}, {best_path[6:-3]} path  {b_ms:.3f} ms"
        if base:
            msg += f"   original {base['layer_old_ms']:.3f} ms -> {base['layer_old_ms'] / b_ms:.2f}x"
        if auto:
            a_ = next((r for r in auto if r["grad_impl"] == ops.grad_impl()), auto[0])
            msg += (f"   (defaults, auto tile + {a_['grad_impl']}: new {a_['layer_new_ms']:.3f} ms, "
                    f"hn {a_['layer_hn_ms']:.3f} ms)")
        print(msg)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"wrote {a.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
