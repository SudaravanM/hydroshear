# Fast HydroShear marker displacement: derivations

HydroShear produces a per-taxel marker displacement field

$$M_{\text{total}} = M_{\text{dilate}} + M_{\text{shear}}$$

on each elastomer pad. This note derives the fast reformulations of each term,
states exactly when each is valid, and records the measured accuracy/speed.
Reference implementation: `HydroFotsSensor.get_marker_dilation` /
`get_marker_shear` in `hydroshear_sensors.py`. Standalone, dependency-free
implementations and tests live in `fft_dilation.py`,
`fft_shear.py`, and `tests/`.

Notation:
- `E` = number of parallel environments.
- `N` = number of tactile points (taxels) on a pad, arranged on a grid `H x W`
  (`N = H*W`), row-major index `n = r*W + c`.
- `P` = number of sampled indenter (object surface) points, **irregularly**
  placed (Poisson-disk sample of the object mesh).
- `p_i` = tactile point position, `x_j` = indenter point position (elastomer
  frame). `h_j` = per-point "height" (relu of penetration). `s`, `λ` = scale and
  Gaussian-decay coefficients (subscript `d` = dilation, `s` = shear).

---

## 1. `Mdilate` — dense sum → 2D FFT convolution  (O(N²) → O(N log N))

### 1.1 The dense computation

For each tactile point `i`, dilation is a Gaussian-weighted sum of the in-plane
offsets to every *other* tactile point `j` (`hydroshear_sensors.py:get_marker_dilation`):

$$
M_{\text{dilate}}[i, c] \;=\; s_d \sum_{j=1}^{N} h[j]\,\bigl(p_{i,c} - p_{j,c}\bigr)\,
\exp\!\bigl(-\lambda_d \lVert p_i - p_j\rVert^2\bigr),
\qquad c \in \{x,y,z\}.
$$

The reference code materializes the dense pairwise tensor `dvec` of shape
`(E, N, N, 3)` and reduces over the middle axis. Cost and memory are **O(N²)**
per environment — the bottleneck at the 1000+ taxel/hand regime.

### 1.2 Key observation: it is a convolution on a regular planar grid

On a **regular planar grid**, taxel `i` at grid index `(r_i, c_i)` has position

$$
p_i = \text{origin} + c_i\, s_x\, \mathbf{e}_x + r_i\, s_y\, \mathbf{e}_y,
$$

i.e. the coordinate along the pad normal `e_z` is **constant** (planar), and the
two tangent coordinates are linear in the integer grid indices. Therefore the
offset between two taxels depends **only on the index offset**
`(Δr, Δc) = (r_i - r_j, c_i - c_j)`:

$$
p_i - p_j = (\Delta c\, s_x)\,\mathbf{e}_x + (\Delta r\, s_y)\,\mathbf{e}_y,
\qquad
\lVert p_i - p_j\rVert^2 = (\Delta c\, s_x)^2 + (\Delta r\, s_y)^2 .
$$

Substituting, the per-channel sum becomes a **2D discrete convolution** of the
height field `h[r,c]` with a *fixed, shift-invariant kernel* `K_c`:

$$
\boxed{\;M_{\text{dilate}}[r, c, \chi] = s_d \sum_{r',c'} h[r',c']\; K_\chi[\,r-r',\; c-c'\,]\;}
$$

with

$$
\begin{aligned}
K_x[\Delta r, \Delta c] &= (\Delta c\, s_x)\; e^{-\lambda_d\,\rho^2}, \\
K_y[\Delta r, \Delta c] &= (\Delta r\, s_y)\; e^{-\lambda_d\,\rho^2}, \\
K_z &\equiv 0, \qquad\qquad \rho^2 = (\Delta c\, s_x)^2 + (\Delta r\, s_y)^2 .
\end{aligned}
$$

`K_z ≡ 0` because a planar pad has no normal-axis offset, so dilation is purely
in-plane. (This matches the paper's Appendix A.7 statement that on planar grids
"tangent channels convolve `h_i`"; the normal channel is handled by the shear /
depth path, not dilation.)

### 1.3 Evaluate the convolution with an FFT

A discrete convolution is a pointwise product in the Fourier domain. For a linear
(non-circular) convolution we zero-pad both operands to length ≥ (signal +
kernel − 1) along each axis, then

$$
M_\chi = s_d\;\operatorname{IFFT2}\!\Bigl(\operatorname{FFT2}(h)\odot \operatorname{FFT2}(K_\chi)\Bigr)\big|_{\text{valid crop}} .
$$

- The kernel offset `(Δr, Δc)` ranges over `[-(H-1), H-1] × [-(W-1), W-1]`, so the
  full linear-convolution length is `3H-2 × 3W-2`; we pad up to the next
  5-smooth FFT size and crop the valid `H×W` block.
- Complexity drops from **O(N²) = O(H²W²)** to **O(N log N)**.
- Per-environment domain randomization of `λ_d` / `s_d` is handled by building a
  batched kernel `(E, kH, kW)` — `λ_d` only changes the kernel, not the FFT plan.
- **Separable kernel (further win, optional):** since
  `e^{-λ_d ρ²} = e^{-λ_d (Δc·s_x)²}\, e^{-λ_d (Δr·s_y)²}`, each `K_χ` is an outer
  product of two 1D kernels, so the 2D FFT can be replaced by 1D FFTs along rows
  then columns. `rfft2` is already fast, so the current code uses the plain 2D
  form; separability is noted for the very-large-grid regime.

### 1.4 Validity and accuracy

- **Exact** (to floating point, `rel_err ≈ 1e-16`) iff the grid is *regular* and
  *planar*. This is precisely the paper's stated condition.
- The repo's taxels come from raycasting onto the elastomer surface, which may be
  slightly **curved**. Then the normal coordinate is not constant and the rewrite
  is an *approximation*: measured `rel_err ≈ 2.5e-3` for a 0.1 mm bulge.
  `enable_fft_dilation()` inspects the grid and warns if the normal-axis spread
  exceeds a tolerance.
- **Crossover:** FFT has fixed overhead, so it only wins past a few hundred
  taxels. At the repo default `num_divs=[7,9]` (63 taxels) the dense path is
  faster, so the fast path is gated by `fft_dilation_min_pts` (default 256).

Measured (CUDA, E=256): `N=1024 → ~120×`; scales cleanly to `N=10000`.

---

## 2. `Mshear` — dense sum → batched matmul (GEMM)  (memory O(P·N) → O(P+N); ~3× faster, no OOM)

### 2.1 The dense computation

Shear sums over the **indenter** points `j = 1..P` for each tactile point `i`
(`hydroshear_sensors.py:get_marker_shear`, lines 336–350). With
`a_j = x_j + f̄_j` the "affected marker position" and `h_j = f̄_{j,z}` the
normal-force component:

$$
M_{\text{shear}}[i, c] \;=\; -\,s_s \sum_{j=1}^{P} h_j\, \bar f_{j,c}\,
\exp\!\bigl(-\lambda_s \lVert p_i - a_j\rVert^2\bigr).
$$

The reference code materializes `(E, P, N, 3)` and reduces over `P`. Since `P`
(≈3000) ≫ `N`, this tensor is the **real memory bottleneck** and OOMs at scale.

### 2.2 Why NOT an FFT

The FFT trick requires shift-invariance: the kernel must depend only on an index
offset. Here the summation is over **indenter points, which are irregularly
placed** (Poisson-disk sample of the object mesh, moving every step). There is no
grid, no fixed stride, so `‖p_i − a_j‖²` is *not* a function of `i − j`. The sum
is therefore **not a convolution** and FFT does not apply. The paper says the same:
"shear is accumulated directly."

### 2.3 What DOES apply: it is a matrix multiply

Split the summand into a factor depending only on `(i,j)` and a factor depending
only on `j`:

$$
M_{\text{shear}}[i, c] = -\,s_s \sum_{j} \underbrace{\exp\!\bigl(-\lambda_s \lVert p_i - a_j\rVert^2\bigr)}_{W[i,j]}\;
\underbrace{\bigl(h_j\, \bar f_{j,c}\bigr)}_{Q[j,c]} .
$$

This is exactly a matrix product over the `j` axis:

$$
\boxed{\;M_{\text{shear}} = -\,s_s\, \bigl(W\, Q\bigr),\qquad W \in \mathbb{R}^{E\times N\times P},\;\; Q \in \mathbb{R}^{E\times P\times 3}.\;}
$$

- `W = exp(-λ_s · cdist(p, a)²)` is the `(E, N, P)` Gaussian weight matrix
  (`torch.cdist` computes the pairwise distances without a `(N,P,3)` temporary).
- `Q[j] = h_j · f̄_j` is `(E, P, 3)`.
- `M = -s_s · torch.bmm(W, Q)` → `(E, N, 3)`.

The `(E,P,N,3)` intermediate is **never formed**: peak extra memory is `O(E·N·P)`
for the weight matrix `W`, versus `O(E·N·P·3)` materialized *plus* the
`dvec`/`gaussian_exp`/`fbar` broadcasts (several more `×3` tensors) in the dense
path. That constant-factor drop (roughly the paper's ~5× memory reduction) is why
matmul completes on cases where dense OOMs.

**Caveat — matmul is still `O(E·N·P)` in memory.** It is strictly lighter than
dense but not free: at very large batches (e.g. `E=256, N=1024, P=6000` the `W`
matrix alone is ~5.9 GiB in fp32) it too runs out of memory. The scalable fix at
that regime is to **chunk over environments** (or tile `N`): loop `bmm` over
slices of the `E` axis so `W` is only ever `(chunk, N, P)`. This keeps the GEMM
speed while bounding memory, and is the recommended pattern for the 1000+ taxel /
16k-env configs. `cdist`+`bmm` also both accept fp16/bf16 for a further ~2× if the
policy tolerates it.

### 2.4 Validity and accuracy

- **Algebraically identical** to the dense sum for *any* point configuration
  (no grid assumption) — measured `rel_err ≈ 3e-16`.
- Applies to the augmented-force path (`aug_hydrosoft_forces`) verbatim.

Measured (CUDA, E=256): `~2.5–2.9×` faster (`P=3000, N=63..441`); on CPU `~3.5–4.3×`.
It also completes cases where the dense path OOMs. Because it helps even at small
`N`, it does not need the size gate that dilation does. (At the most extreme size
both OOM — see the memory caveat above; chunk over `E`.)

---

## 3. `Mtotal`

`Mtotal = Mdilate + Mshear` is a cheap elementwise add; no reformulation needed.
The two terms are independent and could be computed on separate CUDA streams, but
that is a micro-optimization relative to the two above.

---

## 4. Summary

| Term      | Sum over        | Structure                         | Fast form            | Exact?                        | Measured (CUDA, E=256)                 |
|-----------|-----------------|-----------------------------------|----------------------|-------------------------------|----------------------------------------|
| `Mdilate` | taxels (grid)   | shift-invariant → **convolution** | 2D FFT `O(N log N)`  | exact on planar grid (~1e-16) | ~120× at N=1024                        |
| `Mshear`  | indenter pts    | irregular → **not** a convolution | batched matmul (GEMM)| exact, any layout (~3e-16)    | ~2.5–4×, lighter memory (chunk `E` at scale) |
| `Mtotal`  | —               | elementwise add                   | —                    | —                             | —                                      |

**Takeaways**

1. FFT is the right tool *only* for `Mdilate`, because only the taxel–taxel sum
   lives on a regular planar grid. Shear's irregular indenter points rule out a
   convolution.
2. `Mshear` — actually the larger cost, since `P ≫ N` — gets a different but
   equally exact speedup: factor the summand and call GEMM. This is both faster
   and dramatically lighter on memory (it removes the `(E,P,N,3)` blowup).
3. Both are drop-in exact replacements under their stated assumptions and are
   validated against the dense reference in `tests/`.
