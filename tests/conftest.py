from __future__ import annotations

import ctypes
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


class UpstreamLevmar:
    def __init__(self, library: pathlib.Path):
        self.lib = ctypes.CDLL(str(library))

    def solve(
        self,
        model,
        p0,
        target,
        *,
        jac=None,
        bounds=None,
        A=None,
        b=None,
        options=None,
        max_iter=100,
        dtype=np.float64,
    ):
        dtype = np.dtype(dtype)
        ctype = ctypes.c_float if dtype == np.float32 else ctypes.c_double
        p = np.ascontiguousarray(p0, dtype=dtype).copy()
        x = np.ascontiguousarray(target, dtype=dtype)
        m, n = p.size, x.size
        opts = np.ascontiguousarray(
            options
            if options is not None
            else [1e-3, 1e-17, 1e-17, 1e-17, 1e-6],
            dtype=dtype,
        )
        info = np.zeros(10, dtype=dtype)
        callback_type = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctype),
            ctypes.POINTER(ctype),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        )

        def call_model(pp, dst, mm, nn, _):
            pv = np.ctypeslib.as_array(pp, shape=(mm,))
            np.ctypeslib.as_array(dst, shape=(nn,))[:] = model(pv)

        function = callback_type(call_model)
        jacobian = None
        if jac is not None:
            def call_jac(pp, dst, mm, nn, _):
                pv = np.ctypeslib.as_array(pp, shape=(mm,))
                np.ctypeslib.as_array(dst, shape=(nn * mm,))[:] = np.asarray(
                    jac(pv), dtype=dtype
                ).ravel()

            jacobian = callback_type(call_jac)
        prefix = "s" if dtype == np.float32 else "d"
        ptr = ctypes.POINTER(ctype)
        common = [
            p.ctypes.data_as(ptr),
            x.ctypes.data_as(ptr),
            m,
            n,
        ]
        if A is not None:
            matrix = np.ascontiguousarray(A, dtype=dtype)
            rhs = np.ascontiguousarray(b, dtype=dtype)
            name = f"{prefix}levmar_lec_{'der' if jac is not None else 'dif'}"
            args = (
                [function, jacobian] if jac is not None else [function]
            ) + common + [
                matrix.ctypes.data_as(ptr),
                rhs.ctypes.data_as(ptr),
                matrix.shape[0],
                max_iter,
                opts.ctypes.data_as(ptr),
                info.ctypes.data_as(ptr),
                None,
                None,
                None,
            ]
        elif bounds is not None:
            lower = np.ascontiguousarray(bounds[0], dtype=dtype)
            upper = np.ascontiguousarray(bounds[1], dtype=dtype)
            name = f"{prefix}levmar_bc_{'der' if jac is not None else 'dif'}"
            args = (
                [function, jacobian] if jac is not None else [function]
            ) + common + [
                lower.ctypes.data_as(ptr),
                upper.ctypes.data_as(ptr),
                None,
                max_iter,
                opts.ctypes.data_as(ptr),
                info.ctypes.data_as(ptr),
                None,
                None,
                None,
            ]
        else:
            name = f"{prefix}levmar_{'der' if jac is not None else 'dif'}"
            args = (
                [function, jacobian] if jac is not None else [function]
            ) + common + [
                max_iter,
                opts.ctypes.data_as(ptr),
                info.ctypes.data_as(ptr),
                None,
                None,
                None,
            ]
        native = getattr(self.lib, name)
        native.restype = ctypes.c_int
        iterations = native(*args)
        return p, info, iterations


@pytest.fixture(scope="session")
def upstream(tmp_path_factory):
    source = pathlib.Path(
        os.environ.get("LEVMAR_UPSTREAM", ROOT.parent / "vendor-src" / "levmar")
    )
    if not (source / "lm.c").exists():
        pytest.skip("set LEVMAR_UPSTREAM to a levmar 2.6 source checkout")
    output = tmp_path_factory.mktemp("upstream") / "liblevmar-reference.so"
    command = [
        "cc",
        "-O2",
        "-fPIC",
        "-shared",
        "-o",
        str(output),
        "lm.c",
        "Axb.c",
        "misc.c",
        "lmbc.c",
        "lmlec.c",
        "lmblec.c",
        "lmbleic.c",
        "-llapack",
        "-lblas",
        "-lm",
    ]
    subprocess.run(command, cwd=source, check=True, capture_output=True, text=True)
    return UpstreamLevmar(output)
