"""ctypes bindings for the compiled Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_LEVMAR_LIB") or os.path.join(
    ROOT, "dist", "libmojo-levmar.so"
)

I = ctypes.c_int64
CALLBACK = ctypes.CFUNCTYPE(None, I, I, I, I, I)

_SIGNATURES = {
    "mlm_der_f64": ([I] * 11, I),
    "mlm_dif_f64": ([I] * 10, I),
    "mlm_bc_der_f64": ([I] * 13, I),
    "mlm_bc_dif_f64": ([I] * 12, I),
    "mlm_der_f32": ([I] * 11, I),
    "mlm_dif_f32": ([I] * 10, I),
    "mlm_bc_der_f32": ([I] * 13, I),
    "mlm_bc_dif_f32": ([I] * 12, I),
    "mlm_affine_f64": ([I] * 6, None),
    "mlm_jac_basis_f64": ([I] * 6, None),
    "mlm_affine_f32": ([I] * 6, None),
    "mlm_jac_basis_f32": ([I] * 6, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    sources = [
        os.path.join(ROOT, "src", "levmar.mojo"),
        os.path.join(ROOT, "src", "callback.c"),
        os.path.join(ROOT, "build", "build.sh"),
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources):
            return LIB
    pixi = shutil.which("pixi")
    command = (
        [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
        if pixi
        else ["bash", os.path.join(ROOT, "build", "build.sh")]
    )
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=1800
    )
    if result.returncode or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


def address(array) -> int:
    return int(array.ctypes.data)
