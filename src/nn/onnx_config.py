"""
Shared ONNX Runtime configuration for all models.

Centralizes session options to prevent thread contention, and the device:
the GPU (CUDA) when onnxruntime-gpu is installed, else the CPU.
"""

import os
import sys

import numpy as np
import onnxruntime as ort

from nn.lstm_onnx import input_dtype  # noqa: F401 (for the model wrappers)

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
#
# use_tf32 0: ONNX Runtime's CUDA matmuls default to TF32 on Ampere and newer
# GPUs, which keeps 10 bits of mantissa. That, not BEN's float16 inputs, is what
# made GPU bid scores differ from the CPU's by up to ~0.003 (enough to move a
# sampled hand across a threshold). Full float32 brings the difference to ~2e-6
# for 10-30 % more time in the largest calls only. BEN_TF32=1 restores TF32
# (for TensorFlow too, see nn/tf_guard.py).
_CUDA_OPTIONS = {
    'device_id': 0,
    'arena_extend_strategy': 'kSameAsRequested',
    'use_tf32': '1' if os.getenv('BEN_TF32', '0') == '1' else '0',
}


# On the GPU, input values per run: BEN's sampling sends up to 30,000 hands through
# a whole auction. Larger inputs are run in row chunks of at most this many values
# and the outputs joined; every BEN model predicts each row on its own, so the
# result is the same. This bounds the GPU working memory of one run (the bidding
# models' tf2onnx Loop graphs needed ~150 bytes per input value: 30,000 hands x 8
# steps took 7 GB). BEN_ONNX_GPU_CHUNK overrides it.
_GPU_CHUNK_VALUES = int(os.getenv('BEN_ONNX_GPU_CHUNK', 8_000_000))


# GPU memory: ONNX Runtime keeps a memory arena per session and never gives it
# back. BEN calls every model with changing shapes, so 16 sessions' arenas grew
# until an 8 GB GPU ran out during a game. BEN_ONNX_GPU_ARENA picks the remedy:
#   shared  (default) one arena for all sessions, registered with the ONNX
#           Runtime environment: GPU memory follows the largest single run
#           (bounded by the chunking above), not the sum over 16 models, and
#           no run pays for giving memory back.
#   shrink  an arena per session, shrunk after every run (1.3-1.8x slower).
_GPU_ARENA = os.getenv('BEN_ONNX_GPU_ARENA', 'shared').strip().lower()
if _GPU_ARENA not in ('shared', 'shrink'):
    raise ValueError(f"BEN_ONNX_GPU_ARENA={_GPU_ARENA!r}: use 'shared' or 'shrink'")

_shared_arena = None


def _register_shared_arena():
    """Register one CUDA arena allocator with the ONNX Runtime environment, once.
    Sessions created with session.use_env_allocators=1 use it instead of their own."""
    global _shared_arena
    if _shared_arena is None:
        try:
            info = ort.OrtMemoryInfo('Cuda', ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
            # arena_extend_strategy 1 = kSameAsRequested; max_mem 0 = no limit
            # beyond the GPU's own (one run's need is bounded by the chunking).
            arena = ort.OrtArenaCfg({'max_mem': 0, 'arena_extend_strategy': 1})
            ort.create_and_register_allocator_v2('CUDAExecutionProvider', info, {'device_id': '0'}, arena)
            _shared_arena = True
        except Exception as ex:
            sys.stderr.write(f"ONNX Runtime: no shared GPU arena ({ex}); shrinking per-session arenas\n")
            _shared_arena = False
    return _shared_arena


class _ChunkedSession:
    """An InferenceSession on the GPU whose run() splits large batches into row
    chunks and, with per-session arenas, gives its GPU memory back afterwards."""

    def __init__(self, session, max_values, shrink):
        self._session = session
        self._max_values = max_values
        self._run_options = None
        if shrink:
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


# The bidding models' tf2onnx graphs run each LSTM as a Loop node, one step per
# iteration; by default they run as the same network with native LSTM nodes
# (nn/lstm_onnx.py), ~10x faster for small batches on the GPU and ~4x on the CPU.
# BEN_ONNX_LSTM=loop keeps the graphs as they are.
_NATIVE_LSTM = os.getenv('BEN_ONNX_LSTM', 'native').strip().lower() != 'loop'

_sessions = {}


def create_session(model_path):
    """Create an ONNX inference session with optimized settings: on the GPU
    (CUDA, with the CPU for any operator CUDA lacks) when available, else on
    the CPU. One session per model file, so models loaded twice (the bidder and
    the opponent model) share it, and on the GPU its memory.

    model_path may also name a .keras bidding model (an LSTM/BatchNormalization
    stack, see nn/lstm_onnx.py), which then runs on ONNX Runtime."""
    key = os.path.abspath(model_path)
    if key not in _sessions:
        _sessions[key] = _new_session(_model_source(model_path), os.path.basename(model_path))
    return _sessions[key]


def _model_source(model_path):
    """What InferenceSession loads for model_path: the path itself, or the bytes
    of the native-LSTM form of a bidding model."""
    from nn import lstm_onnx
    if model_path.endswith('.keras'):
        rebuilt = lstm_onnx.rewrite_keras(model_path)
        if rebuilt is None:
            raise ValueError(f"{model_path}: not an LSTM model that ONNX Runtime can run")
        return rebuilt
    if _NATIVE_LSTM and lstm_onnx.has_loop_lstm(model_path):
        rebuilt = lstm_onnx.rewrite_loop_onnx(model_path)
        if rebuilt is not None:
            return rebuilt
        sys.stderr.write(f"ONNX Runtime: {os.path.basename(model_path)} runs as its tf2onnx Loop graph\n")
    return model_path


def _new_session(source, label):
    if _use_cuda() and _prepare_cuda():
        try:
            options = get_session_options()
            shared = _GPU_ARENA == 'shared' and _register_shared_arena()
            if shared:
                options.add_session_config_entry('session.use_env_allocators', '1')
            session = ort.InferenceSession(source, options,
                                           providers=[('CUDAExecutionProvider', _CUDA_OPTIONS),
                                                      'CPUExecutionProvider'])
            return _ChunkedSession(session, _GPU_CHUNK_VALUES, shrink=not shared)
        except Exception as ex:
            sys.stderr.write(f"ONNX Runtime: CUDA session for {label} failed ({ex}); using the CPU\n")
    return ort.InferenceSession(source, get_session_options(),
                                providers=['CPUExecutionProvider'])


def session_device(session):
    """'cuda' or 'cpu': where a session from create_session runs."""
    return 'cuda' if 'CUDAExecutionProvider' in session.get_providers() else 'cpu'
