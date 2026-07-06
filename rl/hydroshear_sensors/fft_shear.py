"""
Matmul (GEMM) reformulation of the HydroShear marker-shear kernel.

Despite the ``fft_`` sibling module name, shear is *not* an FFT case: the sum runs
over irregularly-placed indenter points, so ``||p_i - a_j||^2`` is not a function
of an index offset and there is no convolution structure (see FFT.md, section 2).

What shear *is* is a matrix multiply. The reference computation
(``HydroFotsSensor.get_marker_shear``) is::

    M[i, c] = -shear_scale * sum_j  h_j * fbar[j, c]
                                    * exp(-lambda_s * || p_i - a_j ||^2)

where a_j = indenter_pts[j] + fbar[j] and h_j = fbar[j, normal_axis]. Splitting the
summand into a part that depends on (i, j) and a part that depends only on j:

    W[i, j] = exp(-lambda_s * || p_i - a_j ||^2)      (E, N, P)
    Q[j, c] = h_j * fbar[j, c]                        (E, P, 3)
    M       = -shear_scale * (W @ Q)                  (E, N, 3)

This is algebraically identical to the dense sum for *any* point layout (no grid
assumption), and never materialises the dense (E, P, N, 3) pairwise tensor that
the reference builds -- which is the real memory bottleneck since P >> N. It is
both faster and far lighter on memory, and it completes cases where the dense path
runs out of GPU memory.

Dependency-free (torch only) so it can be unit-tested without isaacgym / open3d.
"""

import torch


def _as_env_param(value, num_envs, device, dtype):
    """Broadcast a scalar / (E,) / (E,1) coefficient to (E, 1, 1)."""
    if not torch.is_tensor(value):
        value = torch.tensor(value, device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype).reshape(-1)
    if value.numel() == 1:
        value = value.expand(num_envs)
    assert value.numel() == num_envs, (
        f"coefficient has {value.numel()} entries, expected 1 or num_envs={num_envs}")
    return value.reshape(num_envs, 1, 1)


def shear_dense(tactile_pts, indenter_pts, fbar, lambda_s, shear_scale,
                normal_axis=2, reverse_z=True):
    """Reference dense O(P*N) shear, mirroring ``get_marker_shear`` (given fbar).

    Args:
        tactile_pts:  (E, N, 3) tactile points in the elastomer frame.
        indenter_pts: (E, P, 3) indenter points in the elastomer frame.
        fbar:         (E, P, 3) hydrosoft forces at the indenter points.
        lambda_s:     scalar / (E,) / (E,1) Gaussian decay.
        shear_scale:  scalar / (E,) / (E,1) output scale.
        normal_axis:  which component of fbar is the "height" h_j.
        reverse_z:    sign convention flag, matching the reference.

    Returns:
        (E, N, 3) marker shear displacement.
    """
    E, P, _ = indenter_pts.shape
    N = tactile_pts.shape[1]
    lam = _as_env_param(lambda_s, E, tactile_pts.device, tactile_pts.dtype)
    scale = _as_env_param(shear_scale, E, tactile_pts.device, tactile_pts.dtype)

    a = indenter_pts + fbar                                   # (E, P, 3)
    h = fbar[:, :, normal_axis]                               # (E, P)
    tp = tactile_pts.unsqueeze(1).expand(E, P, N, 3)
    ap = a.unsqueeze(2).expand(E, P, N, 3)
    fb = fbar.unsqueeze(2).expand(E, P, N, 3)
    norm = torch.norm(tp - ap, dim=-1)                        # (E, P, N)
    hh = h.unsqueeze(-1).unsqueeze(-1).expand(E, P, N, 3)
    ge = torch.exp(-lam * norm ** 2).unsqueeze(-1).expand(E, P, N, 3)
    sign = -1.0 * (-1 if reverse_z else 1)
    M = torch.sum(scale.unsqueeze(-1) * hh * fb * ge * sign, dim=1)   # (E, N, 3)
    return M


def shear_matmul(tactile_pts, indenter_pts, fbar, lambda_s, shear_scale,
                 normal_axis=2, reverse_z=True):
    """GEMM shear: exact, memory-light replacement for ``shear_dense``.

    Same arguments and return shape as ``shear_dense``. Never forms the
    (E, P, N, 3) intermediate; peak extra memory is the (E, N, P) weight matrix.
    """
    E = indenter_pts.shape[0]
    lam = _as_env_param(lambda_s, E, tactile_pts.device, tactile_pts.dtype)      # (E,1,1)
    scale = _as_env_param(shear_scale, E, tactile_pts.device, tactile_pts.dtype).reshape(E, 1, 1)

    a = indenter_pts + fbar                                   # (E, P, 3)
    h = fbar[:, :, normal_axis]                               # (E, P)
    d2 = torch.cdist(tactile_pts, a).pow(2)                   # (E, N, P)
    w = torch.exp(-lam * d2)                                  # (E, N, P)
    q = h.unsqueeze(-1) * fbar                                # (E, P, 3)
    sign = -1.0 * (-1 if reverse_z else 1)
    M = scale * sign * torch.bmm(w, q)                        # (E, N, 3)
    return M
