"""
Correctness + speed tests for the FFT-accelerated HydroShear dilation kernel.

Run:  python -m pytest rl/hydroshear_sensors/tests/test_fft_dilation.py -v
  or:  python rl/hydroshear_sensors/tests/test_fft_dilation.py    (plain runner)

These tests are self-contained (no isaacgym / open3d) and check that the FFT
path reproduces the reference dense O(N^2) dilation to floating-point tolerance
on a regular planar grid -- which is exactly the condition the paper assumes.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fft_dilation import (  # noqa: E402
    DilationFFT,
    dilation_dense,
    infer_grid_geometry,
)
from fft_shear import shear_dense, shear_matmul  # noqa: E402


def make_planar_grid(H, W, sy, sx, tangent_axes=(0, 1), normal_offset=0.037,
                     device="cpu", dtype=torch.float32):
    """Build a regular planar taxel grid, flattened row-major (r*W + c)."""
    normal_axis = ({0, 1, 2} - set(tangent_axes)).pop()
    rr = torch.arange(H, device=device, dtype=dtype)
    cc = torch.arange(W, device=device, dtype=dtype)
    grid_r, grid_c = torch.meshgrid(rr, cc, indexing="ij")     # (H, W)
    pts = torch.zeros(H, W, 3, device=device, dtype=dtype)
    pts[..., tangent_axes[0]] = grid_c * sx                     # columns -> x-tangent
    pts[..., tangent_axes[1]] = grid_r * sy                     # rows    -> y-tangent
    pts[..., normal_axis] = normal_offset
    return pts.reshape(-1, 3)


def rel_err(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_matches_dense_planar_scalar_coeffs():
    """FFT == dense on a planar grid, scalar lambda_d / dilate_scale."""
    torch.manual_seed(0)
    H, W = 9, 7                       # matches the repo's num_divs=[7,9] (W=7 cols, H=9 rows)
    sy, sx = 0.0016, 0.0016
    E = 4
    lambda_d, dilate_scale = 8000.0, 100.0

    pts1 = make_planar_grid(H, W, sy, sx, dtype=torch.float64)
    pts = pts1.unsqueeze(0).expand(E, -1, -1).contiguous()
    height = torch.rand(E, H * W, dtype=torch.float64)

    dense = dilation_dense(pts, height, lambda_d, dilate_scale)

    fft = DilationFFT((H, W), (sy, sx), tangent_axes=(0, 1), dtype=torch.float64)
    got = fft(height, lambda_d, dilate_scale)

    err = rel_err(got, dense)
    print(f"[scalar]  rel_err = {err:.3e}")
    assert err < 1e-10, f"FFT dilation deviates from dense: rel_err={err:.3e}"


def test_matches_dense_per_env_coeffs():
    """Per-environment randomized lambda_d / dilate_scale (domain randomization)."""
    torch.manual_seed(1)
    H, W = 12, 10
    sy, sx = 0.0016, 0.0021           # anisotropic spacing
    E = 6

    pts1 = make_planar_grid(H, W, sy, sx, dtype=torch.float64)
    pts = pts1.unsqueeze(0).expand(E, -1, -1).contiguous()
    height = torch.rand(E, H * W, dtype=torch.float64)

    lambda_d = (5000.0 + torch.rand(E, dtype=torch.float64) * 6000.0).reshape(E, 1)
    dilate_scale = (80.0 + torch.rand(E, dtype=torch.float64) * 40.0).reshape(E, 1)

    dense = dilation_dense(pts, height, lambda_d, dilate_scale)
    fft = DilationFFT((H, W), (sy, sx), tangent_axes=(0, 1), dtype=torch.float64)
    got = fft(height, lambda_d, dilate_scale)

    err = rel_err(got, dense)
    print(f"[per-env] rel_err = {err:.3e}")
    assert err < 1e-10, f"per-env FFT dilation deviates: rel_err={err:.3e}"


def test_matches_dense_permuted_axes():
    """Grid whose tangent axes are (x=2, y=0), normal=1 -- like the slim-axis layout."""
    torch.manual_seed(2)
    H, W = 8, 8
    sy, sx = 0.0018, 0.0018
    E = 2
    tangent_axes = (2, 0)             # columns->axis2, rows->axis0, normal=axis1

    pts1 = make_planar_grid(H, W, sy, sx, tangent_axes=tangent_axes, dtype=torch.float64)
    pts = pts1.unsqueeze(0).expand(E, -1, -1).contiguous()
    height = torch.rand(E, H * W, dtype=torch.float64)

    dense = dilation_dense(pts, height, 7000.0, 100.0)
    fft = DilationFFT((H, W), (sy, sx), tangent_axes=tangent_axes, dtype=torch.float64)
    got = fft(height, 7000.0, 100.0)

    err = rel_err(got, dense)
    print(f"[axes]    rel_err = {err:.3e}")
    assert err < 1e-10, f"permuted-axis FFT dilation deviates: rel_err={err:.3e}"


def test_geometry_inference():
    """infer_grid_geometry recovers spacing / axes / planarity from raw points."""
    H, W = 9, 7
    sy, sx = 0.0016, 0.0019
    pts1 = make_planar_grid(H, W, sy, sx, tangent_axes=(0, 1), dtype=torch.float64)
    info = infer_grid_geometry(pts1, (H, W))
    assert info["tangent_axes"] == (0, 1)
    assert info["normal_axis"] == 2
    assert abs(info["spacing"][0] - sy) < 1e-9
    assert abs(info["spacing"][1] - sx) < 1e-9
    assert info["is_planar"]
    print(f"[infer]   spacing={info['spacing']}, planar_residual={info['planar_residual']:.2e}")


def test_curved_grid_is_close_approximation():
    """On a slightly curved (non-planar) grid, FFT is a close approximation.

    The paper's FFT rewrite is exact only for a planar grid. The repo's raycast
    surface is nearly-but-not-perfectly planar; we quantify the residual so the
    approximation is documented rather than silent.
    """
    torch.manual_seed(3)
    H, W = 9, 7
    sy, sx = 0.0016, 0.0016
    E = 2
    pts1 = make_planar_grid(H, W, sy, sx, dtype=torch.float64)
    # Add a gentle spherical bulge along the normal axis (~0.1 mm over the pad).
    rr = torch.arange(H, dtype=torch.float64) - (H - 1) / 2
    cc = torch.arange(W, dtype=torch.float64) - (W - 1) / 2
    gr, gc = torch.meshgrid(rr, cc, indexing="ij")
    bulge = (1e-4) * (1.0 - ((gr / H) ** 2 + (gc / W) ** 2))
    pts_curved = pts1.clone().reshape(H, W, 3)
    pts_curved[..., 2] += bulge
    pts_curved = pts_curved.reshape(-1, 3)

    pts = pts_curved.unsqueeze(0).expand(E, -1, -1).contiguous()
    height = torch.rand(E, H * W, dtype=torch.float64)

    dense = dilation_dense(pts, height, 8000.0, 100.0)
    fft = DilationFFT((H, W), (sy, sx), tangent_axes=(0, 1), dtype=torch.float64)
    got = fft(height, 8000.0, 100.0)
    err = rel_err(got, dense)
    print(f"[curved]  rel_err = {err:.3e}  (bulge ~0.1mm; expected small, not zero)")
    assert err < 5e-2, f"curved-grid approximation unexpectedly large: {err:.3e}"


# ---------------------------------------------------------------------------
# Shear: matmul reformulation correctness
# ---------------------------------------------------------------------------

def _rand_shear_inputs(E, P, N, dtype=torch.float64, device="cpu"):
    tactile = torch.rand(E, N, 3, dtype=dtype, device=device) * 0.02
    indenter = torch.rand(E, P, 3, dtype=dtype, device=device) * 0.02
    fbar = (torch.rand(E, P, 3, dtype=dtype, device=device) - 0.5) * 1e-3
    return tactile, indenter, fbar


def test_shear_matmul_matches_dense_scalar():
    torch.manual_seed(10)
    tactile, indenter, fbar = _rand_shear_inputs(E=6, P=800, N=63)
    dense = shear_dense(tactile, indenter, fbar, 8000.0, 1000 / 0.065)
    fast = shear_matmul(tactile, indenter, fbar, 8000.0, 1000 / 0.065)
    err = rel_err(fast, dense)
    print(f"[shear scalar]  rel_err = {err:.3e}")
    assert err < 1e-10, f"shear matmul deviates from dense: {err:.3e}"


def test_shear_matmul_matches_dense_per_env():
    torch.manual_seed(11)
    E = 5
    tactile, indenter, fbar = _rand_shear_inputs(E=E, P=1200, N=441)
    lam = (5000.0 + torch.rand(E, 1, dtype=torch.float64) * 6000.0)
    scl = (900.0 + torch.rand(E, 1, dtype=torch.float64) * 200.0)
    dense = shear_dense(tactile, indenter, fbar, lam, scl)
    fast = shear_matmul(tactile, indenter, fbar, lam, scl)
    err = rel_err(fast, dense)
    print(f"[shear perenv]  rel_err = {err:.3e}")
    assert err < 1e-10, f"per-env shear matmul deviates: {err:.3e}"


def test_shear_matmul_no_reverse_z():
    torch.manual_seed(12)
    tactile, indenter, fbar = _rand_shear_inputs(E=3, P=500, N=100)
    dense = shear_dense(tactile, indenter, fbar, 7000.0, 100.0, reverse_z=False)
    fast = shear_matmul(tactile, indenter, fbar, 7000.0, 100.0, reverse_z=False)
    err = rel_err(fast, dense)
    print(f"[shear !revz]   rel_err = {err:.3e}")
    assert err < 1e-10, f"shear matmul (reverse_z=False) deviates: {err:.3e}"


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

def _bench(device):
    import time
    print(f"\n=== benchmark on {device} ===")
    sy = sx = 0.0016
    E = 256
    for (H, W) in [(9, 7), (32, 32), (64, 64), (100, 100)]:
        N = H * W
        pts1 = make_planar_grid(H, W, sy, sx, device=device)
        pts = pts1.unsqueeze(0).expand(E, -1, -1).contiguous()
        height = torch.rand(E, N, device=device)
        fft = DilationFFT((H, W), (sy, sx), device=device)

        def sync():
            if device == "cuda":
                torch.cuda.synchronize()

        # warmup
        for _ in range(3):
            fft(height, 8000.0, 100.0)
        sync()
        t0 = time.perf_counter()
        for _ in range(20):
            fft(height, 8000.0, 100.0)
        sync()
        t_fft = (time.perf_counter() - t0) / 20

        t_dense = float("nan")
        # Dense is O(E*N^2*3); skip when the pairwise tensor would be huge.
        if E * N * N * 3 <= 256 * 4096 * 4096 * 3 // 8:
            for _ in range(2):
                dilation_dense(pts, height, 8000.0, 100.0)
            sync()
            t0 = time.perf_counter()
            for _ in range(5):
                dilation_dense(pts, height, 8000.0, 100.0)
            sync()
            t_dense = (time.perf_counter() - t0) / 5

        speedup = (t_dense / t_fft) if t_dense == t_dense else float("nan")
        print(f"  N={N:6d} (E={E})  fft={t_fft*1e3:8.3f} ms  "
              f"dense={t_dense*1e3:9.3f} ms  speedup={speedup:6.1f}x")


def _bench_shear(device):
    import time
    print(f"\n=== shear benchmark on {device} ===")
    E = 256
    for (P, N) in [(3000, 63), (3000, 441), (6000, 1024)]:
        tactile, indenter, fbar = _rand_shear_inputs(E, P, N, dtype=torch.float32, device=device)

        def sync():
            if device == "cuda":
                torch.cuda.synchronize()

        def run(fn):
            for _ in range(3):
                fn(tactile, indenter, fbar, 8000.0, 100.0)
            sync()
            t0 = time.perf_counter()
            for _ in range(10):
                fn(tactile, indenter, fbar, 8000.0, 100.0)
            sync()
            return (time.perf_counter() - t0) / 10

        try:
            t_mm = run(shear_matmul)
            mm_tag = f"matmul={t_mm*1e3:8.3f} ms"
        except RuntimeError as e:
            if device == "cuda":
                torch.cuda.empty_cache()
            t_mm = None
            mm_tag = f"matmul=OOM (weight matrix (E,N,P) too big; batch envs)"
        try:
            t_d = run(shear_dense)
            tag = (f"dense={t_d*1e3:9.3f} ms  speedup={t_d/t_mm:6.1f}x"
                   if t_mm else f"dense={t_d*1e3:9.3f} ms")
        except RuntimeError as e:
            if device == "cuda":
                torch.cuda.empty_cache()
            tag = "dense=OOM  (matmul completes)" if t_mm else "dense=OOM too"
        print(f"  P={P:6d} N={N:6d} (E={E})  {mm_tag}  {tag}")


# ---------------------------------------------------------------------------
# Plain runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_matches_dense_planar_scalar_coeffs,
        test_matches_dense_per_env_coeffs,
        test_matches_dense_permuted_axes,
        test_geometry_inference,
        test_curved_grid_is_close_approximation,
        test_shear_matmul_matches_dense_scalar,
        test_shear_matmul_matches_dense_per_env,
        test_shear_matmul_no_reverse_z,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    _bench("cpu")
    _bench_shear("cpu")
    if torch.cuda.is_available():
        _bench("cuda")
        _bench_shear("cuda")
    print(f"\n{len(tests) - failed}/{len(tests)} correctness tests passed.")
    sys.exit(1 if failed else 0)
