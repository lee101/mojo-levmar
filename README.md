# mojo-levmar

`mojo-levmar` is a standalone Mojo port of the compute-heavy solvers in
[levmar 2.6](https://github.com/alemuntoni/levmar), Manolis Lourakis's
Levenberg-Marquardt nonlinear least-squares library. It exposes a NumPy-friendly
Python API while keeping the optimization loop, finite differences, normal
equations, damping, box projection, and constraint transform kernels in a
single Mojo compilation unit.

This is a derived work of levmar. The upstream headers license levmar under
GPL-2.0-or-later, so this repository is also GPL-2.0-or-later and carries
upstream's complete GPL v2 text in [LICENSE](LICENSE). The source used for the
port is the levmar 2.6 fork at commit
`fdb689a81ff771a650683fe8026b0780ac7160e0`.

## Coverage

Implemented in both Float64 and Float32:

- unconstrained LM with an analytic Jacobian (`d/slevmar_der`);
- unconstrained LM with forward or central differences and levmar's rank-one
  Jacobian updates (`d/slevmar_dif`);
- box-constrained LM with analytic or finite-difference Jacobians
  (`d/slevmar_bc_der` and `d/slevmar_bc_dif`);
- linear equality constraints `A @ p == b`, using the upstream null-space
  reduction and Mojo ports of `LMLEC_FUNC` and `LMLEC_JACF`;
- upstream defaults and stop codes, including `1e-3` initial damping,
  `1e-17` stop thresholds, `1e-6` difference delta, projection of infeasible
  box starts, and invalid-value termination;
- the ten-element levmar `info` vector, exposed through `Result.info`.

Not covered:

- simultaneous box plus linear constraints (`blec`), linear inequalities
  (`lic`, `blic`, `leic`, and `bleic`);
- the `dscl` box-scaling option;
- covariance matrices, Jacobian checking, and the auxiliary statistics API;
- user-supplied work buffers at the Python level.

Linear equality elimination uses NumPy's LAPACK-backed QR and least-squares
routines to construct the particular solution and orthonormal null-space
basis, matching upstream's own LAPACK requirement for `lec`. The per-evaluation
affine and Jacobian transforms run in Mojo. Supplying box bounds together with
linear equalities currently raises `NotImplementedError`.

## Install and run

```bash
pixi install
pixi run build
```

The build writes `dist/libmojo-levmar.so`. The Python package is available in
the Pixi environment through the configured `PYTHONPATH`.

This complete example fits an exponential model with an analytic Jacobian:

```python
import numpy as np
from mojo_levmar import solve

t = np.linspace(0.0, 3.0, 80)
observed = 2.5 * np.exp(-0.7 * t) + 0.2

def model(p):
    return p[0] * np.exp(-p[1] * t) + p[2]

def jacobian(p):
    e = np.exp(-p[1] * t)
    return np.column_stack((e, -p[0] * t * e, np.ones_like(t)))

result = solve(
    model,
    p0=[1.0, 1.2, 0.0],
    target=observed,
    jac=jacobian,
    bounds=([0.0, 0.0, -1.0], [4.0, 3.0, 1.0]),
)

print(result.x)       # [2.5 0.7 0.2]
print(result.reason)  # small residual
```

Omit `jac` for forward differences. Pass a negative difference delta for
central differences:

```python
result = solve(
    model,
    [1.0, 1.2, 0.0],
    observed,
    options=[1e-3, 1e-17, 1e-17, 1e-17, -1e-6],
)
```

Linear equalities are passed as `A` and `b`:

```python
result = solve(model, [1.0, 1.2, 0.0], observed, jac=jacobian,
               A=[[1.0, 0.0, 1.0]], b=[2.7])
```

If `target` is omitted, `model(p)` is treated as a residual vector and fitted
to zero.

The solver accepts only real-valued data. It infers Float32 only from a
Float32 `p0`; otherwise it uses Float64. When Float32 is inferred, other input
arrays must also be Float32 so that precision is not lost silently. Pass
`dtype=np.float32` to request explicit conversion.

## Correctness

There is no maintained installable Python package binding this exact levmar
source. The parity suite therefore compiles the actual upstream C checkout and
calls its `dlevmar_*` and `slevmar_*` functions through `ctypes`. Tests compare
parameters, squared error, success behavior, and, for finite differences, the
iteration/evaluation/linear-solve counters exactly. Coverage includes analytic
and numerical exponential fitting, Rosenbrock, interior and active box bounds,
analytic and numerical linear equalities, and every advertised solver variant
in both Float32 and Float64. It also covers zero residuals, constant/singular
Jacobians, NaN models, infeasible starts, invalid dimensions, inverted bounds,
callback failures, rank-deficient constraints, complex-value rejection, dtype
narrowing, and SIMD remainder paths.

Set `LEVMAR_UPSTREAM` to a levmar 2.6 checkout. For development layouts, the
test and benchmark scripts also look for `../vendor-src/levmar`. Tests that
require the C reference skip if it is absent; invariant and input-validation
tests still run. The benchmark requires the reference checkout.

```bash
pixi run build
pixi run test
```

## How it works

Parameters, targets, Jacobians, and one flat scratch allocation are contiguous
row-major NumPy arrays. Python passes their addresses as 64-bit integers across
the C ABI, and Mojo reconstructs typed `UnsafePointer` views. The exported
symbols are concrete Float32 and Float64 wrappers over dtype-parameterized Mojo
kernels.

Python model functions cross as `ctypes` function pointers. A tiny C trampoline
performs only the indirect call; it holds no state and performs no numerical
work. This leaves callback scheduling—including levmar's finite
difference refresh and rank-one update rules—inside Mojo.

Each iteration accumulates `J.T @ J` and `J.T @ e` from the row-major Jacobian,
augments the diagonal by `mu`, and solves the dense system with pivoted LU.
Small problems use explicit Mojo SIMD with scalar remainder loops. Large dense
problems dispatch to the installed BLAS for its tuned SIMD and threading, as
does the large `J @ Z` equality-constraint transform. Accepted steps use
levmar's cubic damping update; rejected steps increase `mu` geometrically. Box
solves project trial steps and use a backtracking path when the full projected
LM step is rejected. Equality solves parameterize every feasible point as
`p = c + Z @ q` and optimize the free coordinates `q`.

There is no GPU path. Model and Jacobian callbacks produce host NumPy arrays,
and this port keeps the complete solve on the CPU.

## Benchmarks

Measured with `pixi run bench` on an Intel(R) Xeon(R) CPU E5-2697 v4 @
2.30GHz, Linux x86_64. Each cell is the best of five complete solves after
warmup. The reference is a fresh `-O3` build of the actual upstream C source
linked to LAPACK/BLAS. The ratio is `upstream / Mojo`, so values above 1 mean
Mojo is faster.

| case | Mojo | upstream C | upstream / Mojo |
|---|---:|---:|---:|
| analytic exponential, n=20k | 10.14 ms | 10.34 ms | 1.02x |
| finite-difference exponential, n=20k | 12.13 ms | 12.68 ms | 1.05x |
| analytic dense linear, 30k x 16 | 38.00 ms | 56.00 ms | 1.47x |
| box analytic exponential, n=20k | 11.35 ms | 8.07 ms | 0.71x |
| linear constraint, 30k x 16 | 39.03 ms | 64.98 ms | 1.66x |

These are complete solve timings, including Python callbacks and allocations.
