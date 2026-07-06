"""
FFT-accelerated HydroShear dilation kernel.

Background
----------
The HydroShear marker-dilation term (see ``HydroFotsSensor.get_marker_dilation``)
computes, for every tactile point ``i``, a Gaussian-weighted sum of the in-plane
offsets to every other tactile point ``j``::

    M[i, c] = dilate_scale * sum_j  h[j] * (p[i, c] - p[j, c])
                                    * exp(-lambda_d * || p[i] - p[j] ||^2)

The reference implementation materialises the dense (E, N, N, 3) pairwise tensor,
so its cost is O(N^2) per environment.  At the taxel counts targeted by the
Tactile Genesis paper (1000+ taxels/hand) this dominates the elastomer sensor.

The paper (Appendix A.7) notes that *for a regular planar taxel grid* the sum can
be reformulated as a 2D convolution and evaluated with an FFT in O(N log N):

    > "For regular planar grids we accelerate the dilation term with FFT: tangent
    >  channels convolve h_i and the normal channel convolves h_i^alpha; shear is
    >  accumulated directly."

Why it is a convolution
-----------------------
On a regular grid, taxel ``i`` sits at grid index ``(r_i, c_i)`` and its 3D
position is ``p[i] = origin + c_i * sx * e_x + r_i * sy * e_y`` (planar => constant
along the normal axis ``e_z``).  Then the offset ``p[i] - p[j]`` and the Gaussian
weight depend *only* on the index offset ``(dr, dc) = (r_i - r_j, c_i - c_j)``:

    K_x[dr, dc] = (dc * sx) * exp(-lambda_d * ((dc*sx)^2 + (dr*sy)^2))
    K_y[dr, dc] = (dr * sy) * exp(-lambda_d * ((dc*sx)^2 + (dr*sy)^2))
    K_z[dr, dc] = 0                      (planar pad => no normal-axis spread)

so ``M_c[r, c] = dilate_scale * sum_{r',c'} h[r',c'] * K_c[r-r', c-c']`` is a plain
2D convolution of the height field ``h`` with the fixed kernel ``K_c``.

This is an *exact* rewrite of the dense formula whenever the grid is planar and
regular; ``dilation_fft`` and ``dilation_dense`` agree to floating-point tolerance
(see ``tests/test_fft_dilation.py``).  On a slightly curved raycast surface the
normal-axis coordinate varies a little, so the FFT path is a close approximation
rather than bit-exact -- the test suite quantifies that residual.

This module has no isaacgym / open3d / trimesh dependencies so it can be imported
and unit-tested in isolation.
"""

import math
import torch


def _as_env_param(value, num_envs, device, dtype):
    """Broadcast a scalar / (E,) / (E,1) hydro coefficient to shape (E, 1, 1)."""
    if not torch.is_tensor(value):
        value = torch.tensor(value, device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype)
    value = value.reshape(-1)                       # () or (E,) -> (E,) or (1,)
    if value.numel() == 1:
        value = value.expand(num_envs)
    assert value.numel() == num_envs, (
        f"coefficient has {value.numel()} entries, expected 1 or num_envs={num_envs}")
    return value.reshape(num_envs, 1, 1)


def dilation_dense(tactile_pts, height, lambda_d, dilate_scale):
    """Reference dense O(N^2) dilation, mirroring ``get_marker_dilation``.

    This is intentionally a faithful copy of the pairwise computation in
    ``HydroFotsSensor.get_marker_dilation`` (given the per-point height ``h``
    already queried from the indenter SDF) so it can serve as ground truth for
    the FFT path in tests.

    Args:
        tactile_pts:  (E, N, 3) tactile point positions in the elastomer frame.
        height:       (E, N)    per-point height h[j] = relu(-sdf).
        lambda_d:     scalar, (E,) or (E,1) Gaussian decay.
        dilate_scale: scalar, (E,) or (E,1) output scale.

    Returns:
        (E, N, 3) marker dilation displacement.
    """
    num_envs, num_pts, _ = tactile_pts.shape
    lam = _as_env_param(lambda_d, num_envs, tactile_pts.device, tactile_pts.dtype)      # (E,1,1)
    scale = _as_env_param(dilate_scale, num_envs, tactile_pts.device, tactile_pts.dtype)

    dvec = tactile_pts.unsqueeze(2) - tactile_pts.unsqueeze(1)   # (E, N, N, 3)
    norm_dvec = torch.norm(dvec, dim=-1) ** 2                     # (E, N, N)
    gaussian = torch.exp(-lam * norm_dvec).unsqueeze(-1)         # (E, N, N, 1)
    h = height.unsqueeze(1).unsqueeze(-1)                         # (E, 1, N, 1)
    Mdilate = (scale.unsqueeze(-1) * h * dvec * gaussian).sum(dim=2)  # (E, N, 3)
    return Mdilate


class DilationFFT:
    """Precomputes the grid geometry for FFT-based dilation.

    A regular planar taxel grid is described by:
      * ``grid_shape`` = (H, W)   -- rows (H) and columns (W). Total N = H*W.
      * ``spacing``    = (sy, sx) -- physical step between adjacent rows / cols.
      * ``tangent_axes`` = (ax_col, ax_row) -- which world axes columns / rows map
        to in the 3D elastomer frame. The remaining axis is the (planar) normal.

    The flattening convention matches ``get_marker_dilation``: point index
    ``n = r * W + c`` (row-major), which is what ``generate_tactile_points`` +
    the ``reshape(E, num_divs[1], num_divs[0], 3)`` in the field sensor produce,
    with H = num_divs[1] (rows, y) and W = num_divs[0] (cols, x).
    """

    def __init__(self, grid_shape, spacing, tangent_axes=(0, 1), device="cpu",
                 dtype=torch.float32):
        self.H, self.W = int(grid_shape[0]), int(grid_shape[1])
        self.sy, self.sx = float(spacing[0]), float(spacing[1])
        self.ax_col, self.ax_row = int(tangent_axes[0]), int(tangent_axes[1])
        self.normal_axis = ({0, 1, 2} - {self.ax_col, self.ax_row}).pop()
        self.device = device
        self.dtype = dtype

        # Linear-convolution FFT sizes. Output index n in [0,H), input m in [0,H),
        # kernel offset dr = n-m in [-(H-1), H-1]  ->  full conv length 3H-2.
        # We take the valid slice [H-1 : 2H-1] afterwards. Pad up to a fast size.
        self.full_h = 3 * self.H - 2
        self.full_w = 3 * self.W - 2
        self.fft_h = _next_fast_len(self.full_h)
        self.fft_w = _next_fast_len(self.full_w)

        # Integer offset grids for the kernel, indexed as dr in [-(H-1), H-1].
        dr = torch.arange(-(self.H - 1), self.H, device=device, dtype=dtype)   # (2H-1,)
        dc = torch.arange(-(self.W - 1), self.W, device=device, dtype=dtype)   # (2W-1,)
        # Physical offsets (kH, kW).
        self._off_y = (dr * self.sy).reshape(-1, 1).expand(2 * self.H - 1, 2 * self.W - 1)
        self._off_x = (dc * self.sx).reshape(1, -1).expand(2 * self.H - 1, 2 * self.W - 1)
        self._r2 = self._off_x ** 2 + self._off_y ** 2                          # (kH, kW)

    def build_kernels(self, lambda_d):
        """Kernels K_x, K_y for a (batched) Gaussian decay lambda_d.

        Returns tensors of shape (E, kH, kW) where kH=2H-1, kW=2W-1.
        """
        lam = _as_env_param(lambda_d, _num_envs(lambda_d), self.device, self.dtype)  # (E,1,1)
        gaussian = torch.exp(-lam * self._r2.unsqueeze(0))          # (E, kH, kW)
        kx = self._off_x.unsqueeze(0) * gaussian                    # (E, kH, kW)
        ky = self._off_y.unsqueeze(0) * gaussian
        return kx, ky

    def _fft_conv2d(self, h_grid, kernel):
        """Exact 2D linear convolution sum_{r',c'} h[r',c'] * K[r-r', c-c'].

        h_grid: (E, H, W). kernel: (E, kH, kW) with kH=2H-1, kW=2W-1 indexed by
        offset (dr, dc) in [-(H-1),H-1] x [-(W-1),W-1].
        Returns (E, H, W): the valid slice of the full linear convolution.
        """
        Hf, Wf = self.fft_h, self.fft_w
        Hf_slice = slice(self.H - 1, 2 * self.H - 1)
        Wf_slice = slice(self.W - 1, 2 * self.W - 1)
        Hfft = torch.fft.rfft2(h_grid, s=(Hf, Wf))
        Kfft = torch.fft.rfft2(kernel, s=(Hf, Wf))
        full = torch.fft.irfft2(Hfft * Kfft, s=(Hf, Wf))            # (E, Hf, Wf)
        return full[:, Hf_slice, Wf_slice]                          # (E, H, W)

    def __call__(self, height, lambda_d, dilate_scale):
        """FFT dilation.

        Args:
            height:       (E, N) per-point height, N = H*W, row-major (r*W + c).
            lambda_d:     scalar / (E,) / (E,1) Gaussian decay.
            dilate_scale: scalar / (E,) / (E,1) output scale.

        Returns:
            (E, N, 3) marker dilation displacement, laid out to match the dense
            path (component along ``ax_col`` = x-tangent, ``ax_row`` = y-tangent,
            ``normal_axis`` = 0).
        """
        num_envs = height.shape[0]
        h_grid = height.reshape(num_envs, self.H, self.W).to(self.dtype)
        scale = _as_env_param(dilate_scale, num_envs, self.device, self.dtype).reshape(num_envs, 1)

        kx, ky = self.build_kernels(lambda_d)
        # Broadcast single-env kernels to all envs if lambda_d was scalar.
        if kx.shape[0] == 1 and num_envs > 1:
            kx = kx.expand(num_envs, -1, -1)
            ky = ky.expand(num_envs, -1, -1)

        mx = self._fft_conv2d(h_grid, kx).reshape(num_envs, -1)     # (E, N)
        my = self._fft_conv2d(h_grid, ky).reshape(num_envs, -1)     # (E, N)

        out = torch.zeros(num_envs, self.H * self.W, 3, device=self.device, dtype=self.dtype)
        out[:, :, self.ax_col] = scale * mx
        out[:, :, self.ax_row] = scale * my
        # normal-axis component is identically zero for a planar grid
        return out


def _num_envs(value):
    if torch.is_tensor(value):
        n = value.reshape(-1).numel()
        return n
    return 1


def _next_fast_len(n):
    """Smallest 5-smooth (2,3,5) integer >= n -- a fast size for the FFT."""
    if n <= 1:
        return 1
    best = None
    # 2^a * 3^b * 5^c search; ranges are ample for taxel-grid sizes.
    p5 = 1
    while p5 < n * 5:
        p35 = p5
        while p35 < n * 5:
            v = p35
            while v < n:
                v *= 2
            if best is None or v < best:
                best = v
            p35 *= 3
        p5 *= 5
    return best


def infer_grid_geometry(tactile_pts_single, grid_shape, atol_planar=1e-4):
    """Infer (spacing, tangent_axes, planarity residual) from actual tactile pts.

    Args:
        tactile_pts_single: (N, 3) tactile points for ONE environment, laid out
            row-major as (r*W + c) matching ``grid_shape``.
        grid_shape: (H, W).
        atol_planar: threshold on the normal-axis spread for the planar warning.

    Returns:
        dict with keys: spacing=(sy,sx), tangent_axes=(ax_col,ax_row),
        normal_axis, planar_residual (max normal-coord deviation, metres),
        is_planar (bool).
    """
    H, W = int(grid_shape[0]), int(grid_shape[1])
    pts = tactile_pts_single.reshape(H, W, 3)
    # Per-axis variation across rows vs cols identifies which axes are tangent.
    col_step = (pts[:, 1:, :] - pts[:, :-1, :]).reshape(-1, 3).abs().mean(0)   # along W
    row_step = (pts[1:, :, :] - pts[:-1, :, :]).reshape(-1, 3).abs().mean(0)   # along H
    ax_col = int(torch.argmax(col_step).item())
    ax_row = int(torch.argmax(row_step).item())
    if ax_col == ax_row:
        # Degenerate detection; fall back to the two largest-variance axes.
        total = col_step + row_step
        order = torch.argsort(total, descending=True)
        ax_col, ax_row = int(order[0].item()), int(order[1].item())
    normal_axis = ({0, 1, 2} - {ax_col, ax_row}).pop()
    sx = float(col_step[ax_col].item())
    sy = float(row_step[ax_row].item())
    planar_residual = float((pts[..., normal_axis] - pts[..., normal_axis].mean()).abs().max().item())
    return {
        "spacing": (sy, sx),
        "tangent_axes": (ax_col, ax_row),
        "normal_axis": normal_axis,
        "planar_residual": planar_residual,
        "is_planar": planar_residual <= atol_planar,
    }
