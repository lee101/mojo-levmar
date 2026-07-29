from __future__ import annotations

import numpy as np
import pytest

import mojo_levmar as mlm
from mojo_levmar._lib import address, lib


@pytest.fixture
def exponential():
    t = np.linspace(0.0, 3.0, 80)
    target = 2.5 * np.exp(-0.7 * t) + 0.2

    def model(p):
        return p[0] * np.exp(-p[1] * t) + p[2]

    def jacobian(p):
        exponential = np.exp(-p[1] * t)
        return np.column_stack(
            (exponential, -p[0] * t * exponential, np.ones_like(t))
        )

    return model, jacobian, target


def assert_matches_upstream(ours, reference, atol=1e-8, rtol=1e-7):
    parameters, info, iterations = reference
    assert np.allclose(ours.x, parameters, atol=atol, rtol=rtol)
    assert ours.sse == pytest.approx(float(info[1]), abs=atol, rel=rtol)
    assert ours.success == (iterations >= 0 and int(info[6]) not in (3, 4, 5, 7))


def test_analytic_exponential_parity(upstream, exponential):
    model, jacobian, target = exponential
    p0 = np.array([1.0, 1.2, 0.0])
    ours = mlm.solve(model, p0, target, jac=jacobian)
    reference = upstream.solve(model, p0, target, jac=jacobian)
    assert_matches_upstream(ours, reference, atol=1e-9)


@pytest.mark.parametrize("delta", [1e-6, -1e-6])
def test_finite_difference_exponential_parity(upstream, exponential, delta):
    model, _, target = exponential
    p0 = np.array([1.0, 1.2, 0.0])
    options = [1e-3, 1e-17, 1e-17, 1e-17, delta]
    ours = mlm.solve(model, p0, target, options=options)
    reference = upstream.solve(model, p0, target, options=options)
    assert_matches_upstream(ours, reference, atol=2e-8, rtol=1e-6)
    assert np.array_equal(ours.info[5:], reference[1][5:])


def test_rosenbrock_parity(upstream):
    def model(p):
        return np.array([10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]])

    def jacobian(p):
        return np.array([[-20.0 * p[0], 10.0], [-1.0, 0.0]])

    p0 = np.array([-1.2, 1.0])
    target = np.zeros(2)
    ours = mlm.solve(model, p0, target, jac=jacobian, max_iter=200)
    reference = upstream.solve(model, p0, target, jac=jacobian, max_iter=200)
    assert_matches_upstream(ours, reference, atol=1e-9)


def test_box_interior_parity(upstream, exponential):
    model, jacobian, target = exponential
    p0 = np.array([1.0, 1.2, 0.0])
    bounds = (np.array([0.0, 0.0, -1.0]), np.array([4.0, 3.0, 1.0]))
    ours = mlm.solve(model, p0, target, jac=jacobian, bounds=bounds)
    reference = upstream.solve(model, p0, target, jac=jacobian, bounds=bounds)
    assert_matches_upstream(ours, reference, atol=1e-8)


def test_box_boundary_parity(upstream):
    grid = np.linspace(-1.0, 1.0, 30)

    def model(p):
        return p[0] * grid + p[1]

    def jacobian(_):
        return np.column_stack((grid, np.ones_like(grid)))

    target = 3.0 * grid + 2.0
    p0 = np.array([-10.0, 10.0])
    bounds = (np.array([0.0, -1.0]), np.array([1.5, 1.0]))
    ours = mlm.solve(model, p0, target, jac=jacobian, bounds=bounds)
    reference = upstream.solve(model, p0, target, jac=jacobian, bounds=bounds)
    assert_matches_upstream(ours, reference, atol=2e-7)
    assert np.allclose(ours.x, [1.5, 1.0], atol=2e-7)


def test_box_numeric_parity(upstream, exponential):
    model, _, target = exponential
    p0 = np.array([1.0, 1.2, 0.0])
    bounds = (np.array([0.0, 0.0, -1.0]), np.array([4.0, 3.0, 1.0]))
    ours = mlm.solve(model, p0, target, bounds=bounds)
    reference = upstream.solve(model, p0, target, bounds=bounds)
    assert_matches_upstream(ours, reference, atol=2e-7)


@pytest.mark.parametrize("analytic", [False, True])
def test_linear_equality_parity(upstream, analytic):
    points = np.arange(1.0, 7.0)

    def model(p):
        return p[0] + p[1] * points + p[2] * points * points

    def jacobian(_):
        return np.column_stack((np.ones_like(points), points, points * points))

    target = 1.0 + 2.0 * points + 0.5 * points * points
    matrix = np.array([[1.0, 1.0, 0.0]])
    rhs = np.array([3.0])
    p0 = np.zeros(3)
    selected_jac = jacobian if analytic else None
    ours = mlm.solve(model, p0, target, jac=selected_jac, A=matrix, b=rhs)
    reference = upstream.solve(
        model, p0, target, jac=selected_jac, A=matrix, b=rhs
    )
    assert_matches_upstream(ours, reference, atol=2e-7)
    assert matrix @ ours.x == pytest.approx(rhs, abs=1e-12)


def test_float32_parity(upstream, exponential):
    model, jacobian, target = exponential
    p0 = np.array([1.0, 1.2, 0.0], dtype=np.float32)
    target = target.astype(np.float32)
    ours = mlm.solve(model, p0, target, jac=jacobian)
    reference = upstream.solve(
        model, p0, target, jac=jacobian, dtype=np.float32
    )
    assert_matches_upstream(ours, reference, atol=3e-4, rtol=3e-4)
    assert ours.x.dtype == np.float32


@pytest.mark.parametrize(
    "analytic,bounded,constrained",
    [
        (False, False, False),
        (True, True, False),
        (False, True, False),
        (True, False, True),
        (False, False, True),
    ],
)
def test_float32_supported_solver_surface(
    upstream, exponential, analytic, bounded, constrained
):
    model, jacobian, target = exponential
    p0 = np.array([1.0, 1.2, 0.0], dtype=np.float32)
    target = target.astype(np.float32)
    selected_jac = jacobian if analytic else None
    bounds = (
        (
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            np.array([4.0, 3.0, 1.0], dtype=np.float32),
        )
        if bounded
        else None
    )
    matrix = (
        np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
        if constrained
        else None
    )
    rhs = np.array([2.7], dtype=np.float32) if constrained else None
    ours = mlm.solve(
        model, p0, target, jac=selected_jac, bounds=bounds, A=matrix, b=rhs
    )
    reference = upstream.solve(
        model,
        p0,
        target,
        jac=selected_jac,
        bounds=bounds,
        A=matrix,
        b=rhs,
        dtype=np.float32,
    )
    assert_matches_upstream(ours, reference, atol=2e-3, rtol=2e-3)
    assert ours.x.dtype == np.float32


def test_zero_residual_stops_immediately():
    result = mlm.solve(lambda p: p.copy(), [0.0, 0.0], [0.0, 0.0])
    assert result.iterations == 0
    assert result.reason == "small residual"
    assert result.sse == 0.0


def test_constant_model_stops_on_gradient():
    result = mlm.solve(
        lambda p: np.ones(4),
        [2.0, 3.0],
        np.zeros(4),
        jac=lambda p: np.zeros((4, 2)),
    )
    assert result.reason == "small gradient"
    assert np.array_equal(result.x, [2.0, 3.0])


def test_initial_point_is_projected():
    result = mlm.solve(
        lambda p: np.array([p[0], p[0]]),
        [10.0],
        [2.0, 2.0],
        bounds=([0.0], [1.0]),
    )
    assert 0.0 <= result.x[0] <= 1.0
    assert result.x[0] == pytest.approx(1.0)


def test_invalid_model_values_are_reported():
    result = mlm.solve(lambda p: np.array([np.nan]), [1.0], [0.0])
    assert not result.success
    assert result.iterations == -1
    assert result.reason == "invalid model values"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"p0": [], "target": []}, "at least one parameter"),
        ({"p0": [1.0, 2.0], "target": [0.0]}, "measurements"),
        (
            {"p0": [1.0], "target": [0.0], "bounds": ([2.0], [1.0])},
            "lower bound",
        ),
    ],
)
def test_degenerate_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mlm.solve(lambda p: np.zeros(len(kwargs["target"])), **kwargs)


def test_rank_deficient_linear_constraints_rejected():
    with pytest.raises(ValueError, match="full row rank"):
        mlm.solve(
            lambda p: p,
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            A=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            b=[1.0, 2.0],
        )


def test_callback_error_propagates():
    def broken(_):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        mlm.solve(broken, [1.0], [0.0])


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"p0": [1.0 + 2.0j], "target": [0.0]}, "p0"),
        ({"p0": [1.0], "target": [0.0 + 2.0j]}, "target"),
        (
            {
                "p0": [1.0],
                "target": [0.0],
                "bounds": ([0.0 + 1.0j], [2.0]),
            },
            "lower bound",
        ),
    ],
)
def test_complex_inputs_are_rejected(kwargs, name):
    with pytest.raises(TypeError, match=name):
        mlm.solve(lambda p: p, **kwargs)


def test_complex_callback_output_is_rejected():
    with pytest.raises(TypeError, match="model output"):
        mlm.solve(lambda p: np.array([1.0 + 1.0j]), [1.0], [0.0])

    with pytest.raises(TypeError, match="jacobian output"):
        mlm.solve(
            lambda p: p,
            [1.0],
            [0.0],
            jac=lambda p: np.array([[1.0 + 1.0j]]),
        )


def test_inferred_float32_does_not_narrow_other_inputs():
    with pytest.raises(TypeError, match="target would be narrowed"):
        mlm.solve(
            lambda p: p,
            np.array([1.0], dtype=np.float32),
            np.array([0.0], dtype=np.float64),
        )

    result = mlm.solve(
        lambda p: p,
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float64),
        dtype=np.float32,
    )
    assert result.x.dtype == np.float32


def test_nonfinite_linear_constraints_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        mlm.solve(
            lambda p: p,
            [0.0, 0.0],
            [1.0, 2.0],
            A=[[np.nan, 0.0]],
            b=[1.0],
        )


def test_large_normal_equations_blas_threshold():
    rng = np.random.default_rng(12)
    design = np.ascontiguousarray(rng.normal(size=(20_001, 5)))
    truth = rng.normal(size=5)
    target = design @ truth

    result = mlm.solve(
        lambda p: design @ p,
        np.zeros(5),
        target,
        jac=lambda _: design,
    )

    assert result.success
    assert np.allclose(result.x, truth, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [31, 11_112])
def test_jacobian_basis_simd_tail_and_parallel_threshold(dtype, n):
    rng = np.random.default_rng(n)
    jacobian = np.ascontiguousarray(rng.normal(size=(n, 10)), dtype=dtype)
    basis = np.ascontiguousarray(rng.normal(size=(10, 9)), dtype=dtype)
    transformed = np.empty((n, 9), dtype=dtype)

    transform = getattr(
        lib(), "mlm_jac_basis_f32" if dtype == np.float32 else "mlm_jac_basis_f64"
    )
    transform(
        address(jacobian),
        address(basis),
        address(transformed),
        n,
        10,
        9,
    )

    tolerance = 2e-5 if dtype == np.float32 else 1e-12
    assert np.allclose(
        transformed, jacobian @ basis, atol=tolerance, rtol=tolerance
    )
