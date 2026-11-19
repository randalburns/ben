"""Serialise TensorFlow calls when the process runs on gevent greenlets.

TensorFlow keeps its eager/graph execution flag per OS thread, and gevent
cannot patch that away: every greenlet in gameapi.py shares a single OS thread,
so they all share the flag. While a tf.function is being traced the flag says
"graph mode"; if the tracing greenlet yields, whichever greenlet takes over
runs its own eager call with that flag still set and dies with

    RuntimeError: Attempting to capture an EagerTensor without building a
    function.

Holding tf_lock across every TensorFlow entry point keeps two greenlets from
being inside TensorFlow at the same time, so a trace always finishes - and
restores the flag - before anyone else looks at it. Greenlets are cooperative,
so a TensorFlow call only ever gives the lock up at a yield point, which means
this costs nothing in the normal case.

Without gevent the flag really is per thread, there is nothing to guard, and
tf_lock is a no-op so real threads keep running inference in parallel.
"""

import os
from contextlib import nullcontext
from threading import RLock

# On a GPU, take memory as the models need it rather than all of it at start-up
# (TensorFlow's default), so several BEN processes can share one GPU. TensorFlow
# reads this when it first sets up the GPU, which is after this module is
# imported by every model wrapper. Set it to "false" to restore the default.
# CUDA_VISIBLE_DEVICES=-1 keeps TensorFlow on the CPU.
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

# TensorFlow 2.21 has no CUDA kernels compiled for GPUs newer than its build
# (e.g. Blackwell, compute capability 12.0): the driver JIT-compiles them from
# PTX on first use and caches the result in ~/.nv/ComputeCache. The default
# cache is too small to hold them, so they would be recompiled run after run.
os.environ.setdefault('CUDA_CACHE_MAXSIZE', str(4 << 30))

# Full float32 on the GPU. TensorFlow (like ONNX Runtime) runs float32 matmuls
# and cuDNN LSTMs in TF32 on Ampere and newer GPUs, keeping 10 bits of mantissa:
# bid scores then differ from the CPU's by ~5e-4, enough for 3 of 8 test boards
# to be bid or played differently. NVIDIA_TF32_OVERRIDE=0 makes cuBLAS and cuDNN
# skip TF32 for everything in the process; it is read when they start, after
# this module is imported. BEN_TF32=1 keeps TF32 (a little faster, see
# nn/onnx_config.py for ONNX Runtime's setting).
if os.environ.get('BEN_TF32') != '1':
    os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')


def _preload_cusolver():
    """tensorflow[and-cuda] installs the CUDA libraries as pip packages, and
    TensorFlow 2.21 finds them through its RUNPATH - which lists every one of
    them except cuSOLVER. Without libcusolver it registers no GPU at all
    ("Cannot dlopen some GPU libraries"). Loading it from the pip package first
    makes TensorFlow's own dlopen by name find it. Does nothing without the
    package (CPU-only installs) or when the library is already found."""
    try:
        import ctypes
        import nvidia.cusolver
    except ImportError:
        return
    for base in nvidia.cusolver.__path__:
        lib = os.path.join(base, 'lib', 'libcusolver.so.11')
        if os.path.isfile(lib):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            return


if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
    _preload_cusolver()


def _running_on_gevent():
    try:
        from gevent import monkey
    except ImportError:
        return False
    # patch_all() runs before this package is imported, so this is decided once.
    return monkey.is_module_patched('threading')


# RLock, so a guarded call that reaches another guarded call cannot deadlock.
tf_lock = RLock() if _running_on_gevent() else nullcontext()
