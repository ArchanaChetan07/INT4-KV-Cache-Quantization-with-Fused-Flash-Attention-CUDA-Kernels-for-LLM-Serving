"""JIT compilation of the CUDA extension via torch.utils.cpp_extension.

Opt-in: set the project's *_JIT_CUDA env var to compile csrc/ on import
(first compile takes ~1-2 min; later loads hit torch's build cache).
On Windows, cl.exe is located automatically from Visual Studio installs.
"""

import glob
import os
import shutil
import sys


_MSVC_GLOBS = [
    r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2019\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
]


def _find_msvc_dir():
    """Locate the Hostx64 cl.exe directory from a VS install, or None."""
    if sys.platform != "win32":
        return None
    cl = shutil.which("cl.exe")
    if cl:
        return os.path.dirname(cl)
    for pattern in _MSVC_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def load_extension(name: str, env_flag: str):
    """Compile and load csrc/*.cu + *.cpp if env_flag is set. Returns the
    module, or None (flag unset, no GPU, or compile failure)."""
    if not os.environ.get(env_flag):
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        from torch.utils.cpp_extension import load

        cuda_flags = []
        msvc = _find_msvc_dir()
        if msvc:
            # PATH edits don't reliably reach nvcc's cl.exe subprocess
            # (e.g. under MSYS/Git Bash shells) — -ccbin points nvcc at
            # the host compiler by absolute path instead
            os.environ["PATH"] = msvc + os.pathsep + os.environ["PATH"]
            cuda_flags = ["-ccbin", os.path.join(msvc, "cl.exe")]

        csrc = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")
        sources = sorted(
            glob.glob(os.path.join(csrc, "*.cu")) +
            glob.glob(os.path.join(csrc, "*.cpp")))
        if not sources:
            return None
        return load(name=name, sources=sources,
                    extra_cuda_cflags=cuda_flags, verbose=False)
    except Exception as e:
        import warnings
        warnings.warn(f"{env_flag} set but JIT compile failed: {e}")
        return None
