"""
BEN's bidding models (bidder and bid info) as ONNX graphs with native LSTM nodes.

The bidding models are a stack of Keras LSTM layers, each followed by
BatchNormalization, and one or two time-distributed Dense heads. tf2onnx turned
the Keras LSTMs of models/onnx/bidder.onnx and bidinfo.onnx into ONNX Loop nodes
that run one time step per iteration. On the GPU a Loop synchronises with the
host every step, so even a one-hand call costs ~2 ms, and the unrolled cell
needs ~150 bytes of GPU memory per input value. ONNX's own LSTM operator runs
the whole sequence in one cuDNN call: the play models, which tf2onnx did convert
to LSTM nodes, take 0.4 ms for a small batch.

build() writes the same network - same weights, same arithmetic in float32 -
with LSTM nodes. The weights come from either source:

  spec_from_loop_onnx(path)   a tf2onnx graph with Loop-based LSTMs (bidder.onnx)
  spec_from_keras(path)       a .keras model (the default.conf bidding models)

rewrite_loop_onnx() and rewrite_keras() check the result against the source on
random input and return None when they disagree, so a model whose graph does not
look like the pattern here keeps running as it is. Nothing is written to disk.

The rebuilt graphs take float16 input, as the Keras wrappers do (bidder_tf2
casts to float16): BEN builds the bidding inputs as float16 arrays already.
"""

import os
import sys

import numpy as np

# Largest difference accepted between the rebuilt graph and its source on the
# self-check. Both are float32; reordering the sums gives ~1e-6.
_TOLERANCE = 1e-4


class Spec:
    """lstms: [(kernel [in, 4u], recurrent [u, 4u], bias [4u])] in Keras gate order
    i, f, c, o; norms: [(scale [u], shift [u])] applied after each LSTM;
    heads: [(kernel [u, k], bias [k], softmax?)]."""

    def __init__(self, lstms, norms, heads):
        self.lstms, self.norms, self.heads = lstms, norms, heads


def _keras_to_onnx_gates(w):
    """Keras packs the gates as i, f, c, o along the last axis; ONNX as i, o, f, c."""
    i, f, c, o = np.split(w, 4, axis=-1)
    return np.concatenate([i, o, f, c], axis=-1)


def build(spec, input_size, half_input=True):
    """The ONNX model (serialised bytes) for spec: input [batch, steps, input_size],
    one float32 output [batch, steps, k] per head. BEN builds the bidding inputs
    as float16 arrays; with half_input the graph takes them as they are and casts
    them to float32 itself (on the GPU), which is exact and moves half the bytes."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    nodes, inits = [], []

    def const(name, value):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(value, dtype=np.float32), name))
        return name

    def axes(name, values):
        inits.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    source = 'input'
    if half_input:
        nodes.append(helper.make_node('Cast', ['input'], ['input32'], to=TensorProto.FLOAT))
        source = 'input32'
    # [batch, steps, in] -> [steps, batch, in], the layout the LSTM operator takes.
    nodes.append(helper.make_node('Transpose', [source], ['x0'], perm=[1, 0, 2]))
    axis1 = axes('axis1', [1])
    x = 'x0'
    for n, ((kernel, recurrent, bias), (scale, shift)) in enumerate(zip(spec.lstms, spec.norms)):
        units = recurrent.shape[0]
        W = const(f'W{n}', _keras_to_onnx_gates(kernel).T[None])         # [1, 4u, in]
        R = const(f'R{n}', _keras_to_onnx_gates(recurrent).T[None])      # [1, 4u, u]
        B = const(f'B{n}', np.concatenate([_keras_to_onnx_gates(bias), np.zeros(4 * units)])[None])
        nodes.append(helper.make_node('LSTM', [x, W, R, B], [f'y{n}'], hidden_size=units))
        nodes.append(helper.make_node('Squeeze', [f'y{n}', axis1], [f'h{n}']))    # [steps, batch, u]
        nodes.append(helper.make_node('Mul', [f'h{n}', const(f'scale{n}', scale)], [f'm{n}']))
        nodes.append(helper.make_node('Add', [f'm{n}', const(f'shift{n}', shift)], [f'n{n}']))
        x = f'n{n}'

    outputs = []
    for k, (kernel, bias, softmax) in enumerate(spec.heads):
        nodes.append(helper.make_node('MatMul', [x, const(f'Wd{k}', kernel)], [f'd{k}']))
        nodes.append(helper.make_node('Add', [f'd{k}', const(f'bd{k}', bias)], [f'z{k}']))
        last = f'z{k}'
        if softmax:
            nodes.append(helper.make_node('Softmax', [last], [f's{k}'], axis=-1))
            last = f's{k}'
        nodes.append(helper.make_node('Transpose', [last], [f'out{k}'], perm=[1, 0, 2]))
        outputs.append(helper.make_tensor_value_info(f'out{k}', TensorProto.FLOAT, ['batch', 'steps', kernel.shape[1]]))

    graph = helper.make_graph(
        nodes, 'ben_lstm_stack',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT16 if half_input else TensorProto.FLOAT,
                                       ['batch', 'steps', input_size])],
        outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def spec_from_keras(path):
    """The Spec of a .keras bidding model: LSTM, Dropout, BatchNormalization, ...,
    TimeDistributed(Dense). Raises ValueError for anything else."""
    import keras
    from nn.keras_compat import load_model
    model = load_model(path, compile=False)
    lstms, norms, heads = [], [], []
    for layer in model.layers:
        if isinstance(layer, (keras.layers.InputLayer, keras.layers.Dropout)):
            continue
        if isinstance(layer, keras.layers.LSTM):
            c = layer.get_config()
            if (c['activation'], c['recurrent_activation']) != ('tanh', 'sigmoid') or not c['use_bias'] \
                    or c['go_backwards'] or c['stateful'] or not c['return_sequences']:
                raise ValueError(f'unsupported LSTM {layer.name}')
            lstms.append(tuple(np.asarray(w, np.float64) for w in layer.get_weights()))
        elif isinstance(layer, keras.layers.BatchNormalization):
            gamma, beta, mean, var = (np.asarray(w, np.float64) for w in layer.get_weights())
            scale = gamma / np.sqrt(var + layer.epsilon)
            norms.append((scale, beta - mean * scale))
        elif isinstance(layer, keras.layers.TimeDistributed) and isinstance(layer.layer, keras.layers.Dense):
            act = layer.layer.get_config()['activation']
            if act not in ('softmax', 'linear'):
                raise ValueError(f'unsupported head activation {act}')
            kernel, bias = layer.layer.get_weights()
            heads.append((kernel, bias, act == 'softmax'))
        else:
            raise ValueError(f'unsupported layer {type(layer).__name__}')
    if not lstms or len(lstms) != len(norms) or not heads:
        raise ValueError('not an LSTM/BatchNormalization stack')
    return Spec(lstms, norms, heads), model.inputs[0].shape[-1]


def spec_from_loop_onnx(path):
    """The Spec of a tf2onnx bidding model whose LSTMs are Loop nodes. Each LSTM
    Loop takes its kernel, recurrent kernel and bias as its last three inputs;
    the BatchNormalization after it is a Mul and an Add by constants; a head is a
    MatMul by a constant [u, k] matrix, an Add of a constant [k] bias and maybe a
    Softmax. Raises ValueError when the graph does not look like that."""
    import onnx
    from onnx import numpy_helper
    model = onnx.load(path)
    g = model.graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    consumers = {}
    for node in g.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def only(nodes, op):
        found = [n for n in nodes if n.op_type == op]
        if len(found) != 1:
            raise ValueError(f'expected one {op}, found {len(found)}')
        return found[0]

    def const_input(node):
        values = [init[i] for i in node.input if i in init]
        if len(values) != 1:
            raise ValueError(f'{node.name}: expected one constant input')
        return values[0]

    lstms, norms = [], []
    for node in g.node:
        if node.op_type != 'Loop' or len(node.input) < 3 or not all(i in init for i in node.input[-3:]):
            continue
        kernel, recurrent, bias = (init[i] for i in node.input[-3:])
        if kernel.ndim != 2 or recurrent.ndim != 2 or recurrent.shape[1] != 4 * recurrent.shape[0]:
            continue
        lstms.append((kernel, recurrent, bias))
        # The sequence output is consumed by the BatchNormalization Mul.
        muls = [c for out in node.output for c in consumers.get(out, []) if c.op_type == 'Mul']
        mul = only(muls, 'Mul')
        add = only(consumers.get(mul.output[0], []), 'Add')
        scale, shift = const_input(mul).reshape(-1), const_input(add).reshape(-1)
        norms.append((scale, shift))
        last_norm = add.output[0]
    if not lstms:
        raise ValueError('no Loop-based LSTM found')

    heads = []
    for node in g.node:
        if node.op_type != 'MatMul':
            continue
        weights = [init[i] for i in node.input if i in init and init[i].ndim == 2]
        if len(weights) != 1 or weights[0].shape[0] != lstms[-1][1].shape[0]:
            continue
        kernel = weights[0]
        # The bias Add may come after Reshapes; follow single-consumer chains.
        out, bias, softmax = node.output[0], None, False
        for _ in range(6):
            nxt = consumers.get(out, [])
            if len(nxt) != 1:
                break
            n = nxt[0]
            if n.op_type == 'Add' and bias is None:
                b = [init[i] for i in n.input if i in init and init[i].shape == (kernel.shape[1],)]
                if b:
                    bias = b[0]
            elif n.op_type == 'Softmax':
                softmax = True
            out = n.output[0]
        if bias is None:
            raise ValueError(f'{node.name}: no bias found')
        heads.append((kernel, bias, softmax, out))
    if not heads:
        raise ValueError('no output head found')
    # Heads in the order of the graph outputs, which follow from the head chains
    # (through Transposes and Identities).
    def reaches(start, target):
        seen, todo = set(), [start]
        while todo:
            t = todo.pop()
            if t == target:
                return True
            if t in seen:
                continue
            seen.add(t)
            todo.extend(o for c in consumers.get(t, []) for o in c.output)
        return False
    ordered = []
    for output in g.output:
        match = [h for h in heads if reaches(h[3], output.name)]
        if len(match) != 1:
            raise ValueError(f'cannot match output {output.name} to a head')
        ordered.append(match[0][:3])
    input_size = g.input[0].type.tensor_type.shape.dim[-1].dim_value
    return Spec(lstms, norms, ordered), input_size


def input_dtype(session):
    """The numpy dtype of a session's first input (float16 or float32)."""
    return np.float16 if session.get_inputs()[0].type == 'tensor(float16)' else np.float32


def has_loop_lstm(path):
    """Whether an ONNX file has Loop nodes (tf2onnx's form of a Keras LSTM)."""
    import onnx
    return any(node.op_type == 'Loop' for node in onnx.load(path, load_external_data=False).graph.node)


def _check(rebuilt, reference_run, input_size):
    """Max abs difference between the rebuilt graph (on the CPU) and the source."""
    import onnxruntime as ort
    rng = np.random.default_rng(1)
    x = (rng.random((16, 7, input_size)) < 0.15).astype(np.float32)
    session = ort.InferenceSession(rebuilt, providers=['CPUExecutionProvider'])
    got = session.run(None, {'input': x.astype(input_dtype(session))})
    want = reference_run(x)
    if len(got) != len(want):
        return np.inf
    return max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(got, want))


def _rejected(path, reason):
    sys.stderr.write(f'{os.path.basename(path)}: no native-LSTM graph ({reason})\n')
    return None


def rewrite_loop_onnx(path):
    """bytes of the LSTM-node version of a Loop-based bidding model, or None."""
    import onnxruntime as ort
    try:
        spec, input_size = spec_from_loop_onnx(path)
        rebuilt = build(spec, input_size)
        source = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        name = source.get_inputs()[0].name
        diff = _check(rebuilt, lambda x: source.run(None, {name: x}), input_size)
    except Exception as ex:
        return _rejected(path, f'{type(ex).__name__}: {ex}')
    return rebuilt if diff <= _TOLERANCE else _rejected(path, f'differs from the source by {diff:.2g}')


def rewrite_keras(path):
    """bytes of an ONNX graph for a .keras bidding model, or None."""
    try:
        spec, input_size = spec_from_keras(path)
        rebuilt = build(spec, input_size)
        from nn.keras_compat import load_model
        model = load_model(path, compile=False)

        import tensorflow as tf

        def reference(x):
            # Without TF32: TensorFlow's GPU matmuls use it by default, which
            # is ~5e-4 off the float32 result this check is about.
            tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
            tf.config.experimental.enable_tensor_float_32_execution(False)
            try:
                out = model(x, training=False)
            finally:
                tf.config.experimental.enable_tensor_float_32_execution(tf32)
            return out if isinstance(out, (list, tuple)) else [out]
        diff = _check(rebuilt, reference, input_size)
    except Exception as ex:
        return _rejected(path, f'{type(ex).__name__}: {ex}')
    return rebuilt if diff <= _TOLERANCE else _rejected(path, f'differs from the source by {diff:.2g}')
