"""Python API for the Mojo port of levmar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ._lib import CALLBACK, address, lib

Array = np.ndarray
Model = Callable[[Array], Sequence[float] | Array]
Jacobian = Callable[[Array], Sequence[Sequence[float]] | Array]

LM_INIT_MU = 1.0e-3
LM_STOP_THRESH = 1.0e-17
LM_DIFF_DELTA = 1.0e-6

_REASONS = {
    0: "not started",
    1: "small gradient",
    2: "small parameter step",
    3: "maximum iterations",
    4: "singular matrix",
    5: "no further error reduction",
    6: "small residual",
    7: "invalid model values",
}


@dataclass(frozen=True)
class Result:
    """A levmar solution and the upstream-compatible ten-element info vector."""

    x: Array
    iterations: int
    info: Array
    success: bool
    reason: str
    nfev: int
    njev: int
    n_linear_solves: int

    @property
    def initial_sse(self) -> float:
        return float(self.info[0])

    @property
    def sse(self) -> float:
        return float(self.info[1])


def _dtype_of(p0, dtype) -> tuple[np.dtype, bool]:
    if dtype is not None:
        result = np.dtype(dtype)
        explicit = True
    else:
        source = np.asarray(p0)
        result = np.dtype(np.float32 if source.dtype == np.float32 else np.float64)
        explicit = False
    if result not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("dtype must be float32 or float64")
    return result, explicit


def _check_conversion(value, dtype: np.dtype, name: str, explicit: bool) -> None:
    source = np.asarray(value)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError(f"{name} must be real-valued")
    if (
        not explicit
        and dtype == np.dtype(np.float32)
        and np.issubdtype(source.dtype, np.floating)
        and source.dtype.itemsize > dtype.itemsize
    ):
        raise TypeError(
            f"{name} would be narrowed to float32; pass dtype=np.float32 "
            "to request that conversion"
        )


def _as_vector(
    value, dtype: np.dtype, name: str, *, explicit: bool = True
) -> Array:
    _check_conversion(value, dtype, name, explicit)
    result = np.ascontiguousarray(value, dtype=dtype)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _parameter_vector(value, dtype: np.dtype) -> Array:
    _check_conversion(value, dtype, "p0", True)
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.ndim != 1:
        raise ValueError("p0 must be one-dimensional")
    return result


def _bound_vector(
    value, m: int, dtype: np.dtype, name: str, *, explicit: bool
) -> Array:
    _check_conversion(value, dtype, name, explicit)
    result = np.broadcast_to(np.asarray(value, dtype=dtype), (m,))
    result = np.ascontiguousarray(result)
    if np.isnan(result).any():
        raise ValueError(f"{name} contains NaN")
    return result


def _options(options, dtype: np.dtype, *, explicit: bool) -> Array:
    defaults = [LM_INIT_MU, LM_STOP_THRESH, LM_STOP_THRESH, LM_STOP_THRESH, LM_DIFF_DELTA]
    if options is None:
        return np.asarray(defaults, dtype=dtype)
    values = list(options)
    if len(values) not in (4, 5):
        raise ValueError("options must contain tau, eps1, eps2, eps3[, delta]")
    if len(values) == 4:
        values.append(LM_DIFF_DELTA)
    _check_conversion(values, dtype, "options", explicit)
    result = np.asarray(values, dtype=dtype)
    if not np.isfinite(result[:4]).all() or result[0] < 0 or np.any(result[1:4] < 0):
        raise ValueError("tau and stopping tolerances must be finite and nonnegative")
    if not np.isfinite(result[4]) or result[4] == 0:
        raise ValueError("finite-difference delta must be finite and nonzero")
    return np.ascontiguousarray(result)


def _view(addr: int, size: int, dtype: np.dtype) -> Array:
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    return np.ctypeslib.as_array((ctype * size).from_address(addr))


class _Callbacks:
    def __init__(
        self,
        model: Model,
        jacobian: Jacobian | None,
        dtype: np.dtype,
        m: int,
        n: int,
    ):
        self.model = model
        self.jacobian = jacobian
        self.dtype = dtype
        self.m = m
        self.n = n
        self.error: BaseException | None = None
        self.views: dict[tuple[int, int], Array] = {}
        self.function = CALLBACK(self._function)
        self.jac = CALLBACK(self._jacobian) if jacobian is not None else None

    def _view(self, addr: int, size: int) -> Array:
        key = (addr, size)
        value = self.views.get(key)
        if value is None:
            value = _view(addr, size, self.dtype)
            self.views[key] = value
        return value

    def _function(self, p_addr, dst_addr, m, n, _data):
        try:
            p = self._view(p_addr, m)
            raw = self.model(p)
            _check_conversion(raw, self.dtype, "model output", True)
            value = np.asarray(raw, dtype=self.dtype)
            if value.shape != (n,):
                raise ValueError(f"model returned {value.shape}, expected {(n,)}")
            self._view(dst_addr, n)[:] = value
        except BaseException as exc:
            if self.error is None:
                self.error = exc
            self._view(dst_addr, n).fill(np.nan)

    def _jacobian(self, p_addr, dst_addr, m, n, _data):
        try:
            p = self._view(p_addr, m)
            raw = self.jacobian(p)
            _check_conversion(raw, self.dtype, "jacobian output", True)
            value = np.asarray(raw, dtype=self.dtype)
            if value.shape != (n, m):
                raise ValueError(f"jacobian returned {value.shape}, expected {(n, m)}")
            self._view(dst_addr, n * m)[:] = value.ravel()
        except BaseException as exc:
            if self.error is None:
                self.error = exc
            self._view(dst_addr, n * m).fill(np.nan)


def _callback_address(callback) -> int:
    import ctypes

    return int(ctypes.cast(callback, ctypes.c_void_p).value)


def _solve_unconstrained_or_box(
    model: Model,
    p0: Array,
    target: Array,
    jacobian: Jacobian | None,
    bounds: tuple[Array, Array] | None,
    max_iter: int,
    options: Array,
) -> Result:
    dtype = p0.dtype
    m = p0.size
    n = target.size
    if m == 0:
        raise ValueError("at least one parameter is required")
    if n < m:
        raise ValueError(f"levmar requires measurements ({n}) >= parameters ({m})")
    callbacks = _Callbacks(model, jacobian, dtype, m, n)
    info = np.zeros(10, dtype=dtype)
    work_size = 4 * n + 4 * m + n * m + 2 * m * m
    work = np.empty(work_size, dtype=dtype)
    suffix = "f32" if dtype == np.float32 else "f64"
    function_address = _callback_address(callbacks.function)
    library = lib()
    if bounds is None:
        if jacobian is None:
            native = getattr(library, f"mlm_dif_{suffix}")
            iterations = native(
                function_address,
                0,
                address(p0),
                address(target),
                address(options),
                address(info),
                address(work),
                m,
                n,
                max_iter,
            )
        else:
            native = getattr(library, f"mlm_der_{suffix}")
            iterations = native(
                function_address,
                _callback_address(callbacks.jac),
                0,
                address(p0),
                address(target),
                address(options),
                address(info),
                address(work),
                m,
                n,
                max_iter,
            )
    else:
        lower, upper = bounds
        if jacobian is None:
            native = getattr(library, f"mlm_bc_dif_{suffix}")
            iterations = native(
                function_address,
                0,
                address(p0),
                address(target),
                address(lower),
                address(upper),
                address(options),
                address(info),
                address(work),
                m,
                n,
                max_iter,
            )
        else:
            native = getattr(library, f"mlm_bc_der_{suffix}")
            iterations = native(
                function_address,
                _callback_address(callbacks.jac),
                0,
                address(p0),
                address(target),
                address(lower),
                address(upper),
                address(options),
                address(info),
                address(work),
                m,
                n,
                max_iter,
            )
    if callbacks.error is not None:
        raise callbacks.error
    stop = int(info[6])
    success = iterations >= 0 and stop not in (3, 4, 5, 7)
    return Result(
        x=p0.copy(),
        iterations=int(iterations),
        info=info.copy(),
        success=success,
        reason=_REASONS.get(stop, f"unknown stop code {stop}"),
        nfev=int(info[7]),
        njev=int(info[8]),
        n_linear_solves=int(info[9]),
    )


def _linear_reduction(
    model: Model,
    jacobian: Jacobian | None,
    p0: Array,
    matrix: Array,
    rhs: Array,
):
    dtype = p0.dtype
    m = p0.size
    k = matrix.shape[0]
    singular = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    threshold = max(1.0e-12, m * 10.0 * np.finfo(dtype).eps * (singular[0] if singular.size else 0))
    rank = int(np.count_nonzero(singular > threshold))
    if rank != k:
        raise ValueError("linear constraint matrix must have full row rank")
    particular = np.linalg.lstsq(
        matrix.astype(np.float64), rhs.astype(np.float64), rcond=None
    )[0].astype(dtype)
    q, _ = np.linalg.qr(matrix.T.astype(np.float64), mode="complete")
    basis = np.ascontiguousarray(q[:, k:].astype(dtype))
    free0 = np.ascontiguousarray(basis.T @ (p0 - particular), dtype=dtype)
    full = np.empty(m, dtype=dtype)
    full_jac = None
    suffix = "f32" if dtype == np.float32 else "f64"
    library = lib()
    affine = getattr(library, f"mlm_affine_{suffix}")
    jac_basis = getattr(library, f"mlm_jac_basis_{suffix}")

    def expand(free):
        affine(
            address(free),
            address(particular),
            address(basis),
            address(full),
            m,
            m - k,
        )
        return full

    def reduced_model(free):
        return model(expand(free))

    if jacobian is not None:
        def reduced_jacobian(free):
            nonlocal full_jac
            value = np.ascontiguousarray(jacobian(expand(free)), dtype=dtype)
            if value.ndim != 2 or value.shape[1] != m:
                raise ValueError("jacobian has the wrong shape")
            if full_jac is None or full_jac.shape != (value.shape[0], m - k):
                full_jac = np.empty((value.shape[0], m - k), dtype=dtype)
            jac_basis(
                address(value),
                address(basis),
                address(full_jac),
                value.shape[0],
                m,
                m - k,
            )
            return full_jac
    else:
        reduced_jacobian = None
    return particular, basis, free0, reduced_model, reduced_jacobian


def solve(
    model: Model,
    p0,
    target=None,
    *,
    jac: Jacobian | None = None,
    bounds=None,
    A=None,
    b=None,
    max_iter: int = 100,
    options=None,
    dtype=None,
) -> Result:
    """Fit model parameters using levmar's Levenberg-Marquardt method.

    ``model(p)`` returns predicted measurements and ``target`` contains the
    measurements to fit. If ``target`` is omitted, the model output is treated
    as a residual vector and fitted to zero. An analytic Jacobian is optional.
    Bounds are ``(lower, upper)`` and linear equalities use ``A @ p == b``.
    """
    if not callable(model):
        raise TypeError("model must be callable")
    if jac is not None and not callable(jac):
        raise TypeError("jac must be callable")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    selected, explicit_dtype = _dtype_of(p0, dtype)
    parameters = _parameter_vector(p0, selected)
    if target is None:
        raw_first = model(parameters)
        _check_conversion(raw_first, selected, "model output", True)
        first = np.asarray(raw_first, dtype=selected)
        if first.ndim != 1:
            raise ValueError("model must return a one-dimensional array")
        measurements = np.zeros(first.size, dtype=selected)
    else:
        measurements = _as_vector(
            target, selected, "target", explicit=explicit_dtype
        )
    opts = _options(options, selected, explicit=explicit_dtype)

    box = None
    if bounds is not None:
        if len(bounds) != 2:
            raise ValueError("bounds must be a (lower, upper) pair")
        lower = _bound_vector(
            bounds[0], parameters.size, selected, "lower bound",
            explicit=explicit_dtype,
        )
        upper = _bound_vector(
            bounds[1], parameters.size, selected, "upper bound",
            explicit=explicit_dtype,
        )
        if np.any(lower > upper):
            raise ValueError("a lower bound exceeds its upper bound")
        box = (lower, upper)

    if (A is None) != (b is None):
        raise ValueError("A and b must be supplied together")
    if A is None:
        return _solve_unconstrained_or_box(
            model, parameters, measurements, jac, box, max_iter, opts
        )
    if box is not None:
        raise NotImplementedError(
            "simultaneous box and linear constraints are not yet supported"
        )
    _check_conversion(A, selected, "A", explicit_dtype)
    matrix = np.ascontiguousarray(A, dtype=selected)
    rhs = _as_vector(b, selected, "b", explicit=explicit_dtype)
    if matrix.ndim != 2 or matrix.shape[1] != parameters.size:
        raise ValueError("A must have shape (constraints, parameters)")
    if matrix.shape[0] != rhs.size or matrix.shape[0] >= parameters.size:
        raise ValueError("A must have fewer rows than columns and match b")
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError("A and b must contain only finite values")
    particular, basis, free0, reduced_model, reduced_jac = _linear_reduction(
        model, jac, parameters, matrix, rhs
    )
    reduced = _solve_unconstrained_or_box(
        reduced_model,
        free0,
        measurements,
        reduced_jac,
        None,
        max_iter,
        opts,
    )
    solution = np.ascontiguousarray(particular + basis @ reduced.x, dtype=selected)
    return Result(
        x=solution,
        iterations=reduced.iterations,
        info=reduced.info,
        success=reduced.success,
        reason=reduced.reason,
        nfev=reduced.nfev,
        njev=reduced.njev,
        n_linear_solves=reduced.n_linear_solves,
    )


least_squares = solve
levmar = solve
