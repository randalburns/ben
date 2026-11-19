"""
Shared ONNX Runtime configuration for all models.

Centralizes session options to prevent thread contention, and the device:
the GPU (CUDA) when onnxruntime-gpu is installed, else the CPU.
"""

import os
import sys

import numpy as np
import onnxruntime as ort

# Global session options - shared by all ONNX models
def get_session_options():
    """Get optimized session options for ONNX Runtime."""
    sess_options = ort.SessionOptions()

    # Memory optimization
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True

    # Thread settings - limit to prevent contention
    # Use 1 thread per model to avoid contention with multiple models
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    # Execution mode - sequential is often faster for small batches
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Graph optimization - use extended for balance of speed vs memory
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    return sess_options


def _use_cuda():
    """BEN_ONNX_DEVICE=cuda|cpu picks the device; unset, CUDA when onnxruntime-gpu
    is installed and CUDA_VISIBLE_DEVICES does not hide the GPU (-1)."""
    choice = os.getenv('BEN_ONNX_DEVICE', '').strip().lower()
    if choice not in ('', 'cuda', 'cpu'):
        raise ValueError(f"BEN_ONNX_DEVICE={choice!r}: use 'cuda' or 'cpu' (or leave it unset)")
    if choice == 'cpu' or os.getenv('CUDA_VISIBLE_DEVICES') == '-1':
        return False
    return 'CUDAExecutionProvider' in ort.get_available_providers()


_cuda_ready = None


def _prepare_cuda():
    """Load the CUDA and cuDNN libraries once. onnxruntime-gpu finds them in the
    nvidia-* pip packages (installed by tensorflow[and-cuda]) through
    preload_dlls(); a system CUDA install works without it."""
    global _cuda_ready
    if _cuda_ready is None:
        try:
            if hasattr(ort, 'preload_dlls'):
                ort.preload_dlls()
            _cuda_ready = True
        except Exception as ex:
            sys.stderr.write(f"ONNX Runtime: could not load the CUDA libraries ({ex}); using the CPU\n")
            _cuda_ready = False
    return _cuda_ready


# CUDA execution provider options. arena_extend_strategy kSameAsRequested grows
# GPU memory as needed instead of in doubling chunks, so several BEN processes
# can share one GPU (as TF_FORCE_GPU_ALLOW_GROWTH does for TensorFlow).
_CUDA_OPTIONS = {
    'device_id': 0,
    'arena_extend_strategy': 'kSameAsRequested',
}


# On the GPU, input values per run: BEN's sampling sends up to 30,000 hands through
# a whole auction, and ONNX Runtime's CUDA LSTM needs about 150 bytes of working
# memory per input value (30,000 hands x 8 steps of the bidder took 7 GB). Larger
# inputs are run in row chunks of at most this many values (~1.2 GB) and the
# outputs joined; every BEN model predicts each row on its own, so the result is
# the same. BEN_ONNX_GPU_CHUNK overrides it.
_GPU_CHUNK_VALUES = int(os.getenv('BEN_ONNX_GPU_CHUNK', 8_000_000))


class _ChunkedSession:
    """An InferenceSession on the GPU whose run() splits large batches into row
    chunks and gives its GPU memory back afterwards. ONNX Runtime keeps a memory
    arena per session and never shrinks it; BEN calls every model with changing
    shapes, so 16 sessions' arenas grew until an 8 GB GPU ran out during a game.
    Shrinking the arena after each run keeps only what the next run needs."""

    def __init__(self, session, max_values):
        self._session = session
        self._max_values = max_values
        self._run_options = ort.RunOptions()
        self._run_options.add_run_config_entry('memory.enable_memory_arena_shrinkage', 'gpu:0')

    def __getattr__(self, name):
        return getattr(self._session, name)

    def run(self, output_names, input_feed, run_options=None):
        return self._run(output_names, input_feed, run_options or self._run_options)

    def _run(self, output_names, input_feed, run_options):
        arrays = list(input_feed.values())
        rows = arrays[0].shape[0] if arrays and getattr(arrays[0], 'ndim', 0) else 0
        per_row = sum(a.size // max(1, a.shape[0]) for a in arrays) if rows else 0
        if rows <= 1 or per_row * rows <= self._max_values:
            return self._session.run(output_names, input_feed, run_options)
        step = max(1, self._max_values // per_row)
        parts = [self._session.run(output_names, {k: v[i:i + step] for k, v in input_feed.items()}, run_options)
                 for i in range(0, rows, step)]
        return [np.concatenate([p[j] for p in parts]) for j in range(len(parts[0]))]


_sessions = {}


def create_session(model_path):
    """Create an ONNX inference session with optimized settings: on the GPU
    (CUDA, with the CPU for any operator CUDA lacks) when available, else on
    the CPU. One session per model file, so models loaded twice (the bidder and
    the opponent model) share it, and on the GPU its memory."""
    key = os.path.abspath(model_path)
    if key not in _sessions:
        _sessions[key] = _new_session(model_path)
    return _sessions[key]


def _new_session(model_path):
    if _use_cuda() and _prepare_cuda():
        try:
            session = ort.InferenceSession(model_path, get_session_options(),
                                           providers=[('CUDAExecutionProvider', _CUDA_OPTIONS),
                                                      'CPUExecutionProvider'])
            return _ChunkedSession(session, _GPU_CHUNK_VALUES)
        except Exception as ex:
            sys.stderr.write(f"ONNX Runtime: CUDA session for {os.path.basename(model_path)} "
                             f"failed ({ex}); using the CPU\n")
    return ort.InferenceSession(model_path, get_session_options(),
                                providers=['CPUExecutionProvider'])


def session_device(session):
    """'cuda' or 'cpu': where a session from create_session runs."""
    return 'cuda' if 'CUDAExecutionProvider' in session.get_providers() else 'cpu'
