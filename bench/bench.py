"""Benchmark mojo-levmar against the actual upstream levmar 2.6 C library."""

from __future__ import annotations

import math
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import mojo_levmar as mlm  # noqa: E402
from tests.conftest import UpstreamLevmar  # noqa: E402


def build_reference() -> UpstreamLevmar:
    source = pathlib.Path(
        os.environ.get("LEVMAR_UPSTREAM", ROOT.parent / "vendor-src" / "levmar")
    )
    if not (source / "lm.c").exists():
        raise RuntimeError("set LEVMAR_UPSTREAM to the levmar 2.6 source checkout")
    output = pathlib.Path(tempfile.mkdtemp(prefix="mojo-levmar-bench-")) / "levmar.so"
    subprocess.run(
        [
            "cc", "-O3", "-fPIC", "-shared", "-o", str(output),
            "lm.c", "Axb.c", "misc.c", "lmbc.c", "lmlec.c", "lmblec.c",
            "lmbleic.c", "-llapack", "-lblas", "-lm",
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    return UpstreamLevmar(output)


def best_time(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    upstream = build_reference()
    cases = []

    t = np.linspace(0.0, 6.0, 20_000)
    target = 2.5 * np.exp(-0.7 * t) + 0.2

    def exponential(p):
        return p[0] * np.exp(-p[1] * t) + p[2]

    def exponential_jac(p):
        exp = np.exp(-p[1] * t)
        return np.column_stack((exp, -p[0] * t * exp, np.ones_like(t)))

    cases.append(
        (
            "analytic exponential, n=20k",
            lambda: mlm.solve(exponential, [1.0, 1.2, 0.0], target, jac=exponential_jac),
            lambda: upstream.solve(
                exponential, [1.0, 1.2, 0.0], target, jac=exponential_jac
            ),
        )
    )
    cases.append(
        (
            "finite-difference exponential, n=20k",
            lambda: mlm.solve(exponential, [1.0, 1.2, 0.0], target),
            lambda: upstream.solve(exponential, [1.0, 1.2, 0.0], target),
        )
    )

    rng = np.random.default_rng(4)
    design = np.ascontiguousarray(rng.normal(size=(30_000, 16)))
    truth = rng.normal(size=16)
    dense_target = design @ truth

    def dense_model(p):
        return design @ p

    def dense_jac(_):
        return design

    cases.append(
        (
            "analytic dense linear, 30k x 16",
            lambda: mlm.solve(dense_model, np.zeros(16), dense_target, jac=dense_jac),
            lambda: upstream.solve(
                dense_model, np.zeros(16), dense_target, jac=dense_jac
            ),
        )
    )

    lower = np.array([0.0, 0.0, -1.0])
    upper = np.array([4.0, 3.0, 1.0])
    cases.append(
        (
            "box analytic exponential, n=20k",
            lambda: mlm.solve(
                exponential,
                [1.0, 1.2, 0.0],
                target,
                jac=exponential_jac,
                bounds=(lower, upper),
            ),
            lambda: upstream.solve(
                exponential,
                [1.0, 1.2, 0.0],
                target,
                jac=exponential_jac,
                bounds=(lower, upper),
            ),
        )
    )

    constraint = np.zeros((1, 16))
    constraint[0, :2] = 1.0
    constraint_rhs = np.array([truth[0] + truth[1]])
    constrained_start = np.zeros(16)
    constrained_start[:2] = constraint_rhs[0] / 2.0
    cases.append(
        (
            "linear constraint, 30k x 16",
            lambda: mlm.solve(
                dense_model,
                constrained_start,
                dense_target,
                jac=dense_jac,
                A=constraint,
                b=constraint_rhs,
            ),
            lambda: upstream.solve(
                dense_model,
                constrained_start,
                dense_target,
                jac=dense_jac,
                A=constraint,
                b=constraint_rhs,
            ),
        )
    )

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| case | Mojo | upstream C | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        ours()
        theirs()
        mojo_seconds = best_time(ours)
        upstream_seconds = best_time(theirs)
        ratio = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{upstream_seconds * 1e3:.2f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
