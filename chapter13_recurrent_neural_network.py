"""LSTM(16) -> Dense(1), implemented with NumPy and Python's standard library.

Run: python jena_lstm_numpy.py --csv jena_climate_2009_2016.csv
Or:  python jena_lstm_numpy.py --download
Test the implementation: python jena_lstm_numpy.py --self-test
Only external dependency: numpy. No autodiff or ML frameworks.

The default window indices match chapter 13 of Deep Learning with Python:
https://deeplearningwithpython.io/chapters/chapter13_timeseries-forecasting/
As in that chapter, split boundaries constrain input windows, not shifted
labels; a few training labels extend past the training input boundary.
Use --strict-splits to keep both inputs and labels inside each split.
Checkpoints contain inference weights and normalization, not optimizer state.
Pure NumPy training is CPU-only and can be substantially slower than Keras.
"""

# Numeric-shape annotation edition. Only comments have been changed.
# Model-method comments use the DEFAULT FULL WEATHER BATCH:
# 256 sequences, 120 time steps, 14 input features, 16 hidden units, 64 gate values.
# These are concrete example shapes, NOT restrictions enforced by the model.
# Dataset-size comments assume the 420451 records stated in the conversation.
# The CSV length is determined at runtime; other input files may differ.
# With default splits, final input batches are:
# training (103,120,14), validation (206,120,14), test (118,120,14).
# Their hidden/cell states are (103,16), (206,16), (118,16), respectively.
# Their predictions are (103,1), (206,1), (118,1), respectively.
# With --strict-splits, final batch sizes instead become 215, 62, and 118.
# The self-tests intentionally use smaller sizes, annotated at their call sites:
# gradient test input (2,4,2), states (2,3), gates (2,12), output (2,1);
# learning test input (16,5,2), states (16,3), gates (16,12), output (16,1).
# Those tests use Wx (2,12), Wh (3,12), b (12,), Wy (3,1), by (1,).
# NumPy scalars have shape (); Python scalars/containers have no .shape.
# h is the integer 16 in __init__, but a hidden-state array in forward().
# c and c_new are cell-state arrays; they are computed states, not parameters.

import argparse
import csv
import time
import urllib.request
import zipfile
from pathlib import Path
import tempfile

import numpy as np


def sigmoid(x):
    # Stable sigmoid without clipping the input (preserves its derivative).
    # Input x: elementwise input; shape (256,16) for a default full-batch LSTM gate; output has the same shape
    return np.exp(-np.logaddexp(0, -x))


class WindowDataset:
    """Re-iterable, lazy batches. targets[s] is the label for window start s."""
    def __init__(self, data, targets, sequence_length=120, sampling_rate=6,
                 batch_size=256, start_index=0, end_index=None,
                 shuffle=False, seed=42):
        # Input data: shape (419593,14) for the stated 420451-row CSV after removing 858 trailing rows
        # Input targets: shape (419593,) for the stated CSV; one target per possible starting row
        # Input sequence_length: int scalar; default 120 time steps
        # Input sampling_rate: int scalar; default 6 rows between observations
        # Input batch_size: int scalar; default maximum batch size 256
        # Input start_index: int scalar; inclusive starting row
        # Input end_index: int scalar or None; exclusive input boundary
        # Input shuffle: bool scalar; no array shape
        # Input seed: int scalar; random seed
        if min(sequence_length, sampling_rate, batch_size) < 1:
            raise ValueError('Sequence length, sampling rate and batch size must be positive.')
        # end: int scalar; a partition or dataset boundary
        end = len(data) if end_index is None else end_index
        if not 0 <= start_index < end <= len(data):
            raise ValueError('Invalid dataset boundaries.')
        # self.data: shape (419593,14) for the stated 420451-row CSV after removing 858 trailing rows
        # self.targets: shape (419593,) for the stated CSV; one target per possible starting row
        self.data, self.targets = data, targets
        # self.offsets: shape (120,); values [0,6,...,714]
        self.offsets = np.arange(sequence_length) * sampling_rate
        # stop: int scalar; exclusive upper bound for window starts
        stop = min(end - self.offsets[-1], len(targets))
        # self.starts: shape (209511,) for training, (104398,) for validation, or (103542,) for test with the stated CSV and default splits
        self.starts = np.arange(start_index, stop)
        if not len(self.starts):
            raise ValueError('Split is too short for the requested windows.')
        # self.batch_size: int scalar; default maximum batch size 256
        # self.shuffle: bool scalar; no array shape
        self.batch_size, self.shuffle = batch_size, shuffle
        # self.rng: np.random.Generator object; no array shape
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (len(self.starts) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # starts: shape (209511,) for training, (104398,) for validation, or (103542,) for test with the stated CSV and default splits
        starts = self.starts.copy()
        if self.shuffle:
            self.rng.shuffle(starts)
        # j: int scalar; loop index / batch number
        for j in range(0, len(starts), self.batch_size):
            # s: shape (256,); starting rows selected for this batch; shorter in the final batch
            s = starts[j:j + self.batch_size]
            # Indices: (256,1) + (120,) -> (256,120); yields inputs (256,120,14), targets (256,)
            yield self.data[s[:, None] + self.offsets], self.targets[s]


class LSTMRegressor:
    def __init__(self, input_size, hidden_size=16, seed=42, dtype=np.float32):
        # Input input_size: int scalar; 14 weather features
        # Input hidden_size: int scalar; default 16 hidden units
        # Input seed: int scalar; random seed
        # Input dtype: NumPy type/dtype object; no array shape
        # rng: np.random.Generator object; no array shape
        rng = np.random.default_rng(seed)
        # h: int scalar 16; here h is the unit count, NOT the hidden-state array
        h = hidden_size
        # Glorot input kernel, orthogonal recurrent kernel, forget bias = 1.
        # limit: NumPy floating-point scalar; shape ()
        limit = np.sqrt(6 / (input_size + 4 * h))
        # wx: shape (14,64); input weights
        wx = rng.uniform(-limit, limit, (input_size, 4 * h))
        # q: shape (64,16); orthonormal columns from QR decomposition
        # _: shape (16,16); discarded QR triangular matrix
        q, _ = np.linalg.qr(rng.normal(size=(4 * h, h)))
        # b: shape (64,); combined gate biases
        b = np.zeros(4 * h)
        b[h:2*h] = 1
        # limit: NumPy floating-point scalar; shape ()
        limit = np.sqrt(6 / (h + 1))
        # self.params: dict: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
        self.params = {
            # Wx (14,64); Wh (16,64)
            'Wx': wx.astype(dtype), 'Wh': q.T.astype(dtype),
            # b (64,)
            'b': b.astype(dtype),
            # Wy (16,1)
            'Wy': rng.uniform(-limit, limit, (h, 1)).astype(dtype),
            # by (1,)
            'by': np.zeros(1, dtype=dtype),
        }
        # self.hidden_size: int scalar; default 16 hidden units
        self.hidden_size = h

    def forward(self, x, training=False):
        # Input x: shape (256,120,14): 256 sequences, 120 time steps, 14 features
        # Input training: bool scalar; no array shape
        # p: dict: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
        p = self.params
        # x: shape (256,120,14): 256 sequences, 120 time steps, 14 features
        x = np.asarray(x, dtype=p['Wx'].dtype)
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != p['Wx'].shape[0]:
            raise ValueError('Expected x shape (batch, timesteps, input_size).')
        # h: shape (256,16); hidden state
        h = np.zeros((len(x), self.hidden_size), dtype=x.dtype)
        # c: shape (256,16); cell state / memory
        c = np.zeros_like(h)
        # cache: list: 120 tuples when training, empty otherwise; each tuple contains 7 arrays of shape (256,16)
        cache = []
        # t: int scalar; time-step index 0 through 119
        for t in range(x.shape[1]):
            # z: shape (256,64); four groups of 16 preactivation values
            # x[:,t] (256,14) @ Wx (14,64) + h (256,16) @ Wh (16,64) + b (64,) -> (256,64)
            z = x[:, t] @ p['Wx'] + h @ p['Wh'] + p['b']
            # zi: shape (256,16); input gate preactivation
            # zf: shape (256,16); forget gate preactivation
            # zg: shape (256,16); candidate preactivation
            # zo: shape (256,16); output gate preactivation
            zi, zf, zg, zo = np.split(z, 4, axis=1)
            # i: shape (256,16); input gate
            # f: shape (256,16); forget gate
            # g: shape (256,16); candidate memory
            # o: shape (256,16); output gate
            i, f, g, o = sigmoid(zi), sigmoid(zf), np.tanh(zg), sigmoid(zo)
            # c_new: shape (256,16); updated cell state
            c_new = f * c + i * g
            # tc: shape (256,16); tanh of updated cell state
            tc = np.tanh(c_new)
            # h_new: shape (256,16); updated hidden state
            h_new = o * tc
            if training:
                cache.append((h, c, i, f, g, o, tc))
            # h: shape (256,16); hidden state
            # c: shape (256,16); cell state / memory
            h, c = h_new, c_new
        # prediction: shape (256,1); one predicted temperature per sequence
        prediction = h @ p['Wy'] + p['by']
        # Returns prediction (256,1) and either (inputs (256,120,14), hidden (256,16), cache) or None
        return prediction, (x, h, cache) if training else None

    def predict(self, x):
        # Input x: shape (256,120,14): 256 sequences, 120 time steps, 14 features
        # Returns (256,1) for a full default batch
        return self.forward(x)[0]

    def loss_and_gradients(self, x, y):
        # Input x: shape (256,120,14): 256 sequences, 120 time steps, 14 features
        # Input y: input (256,) or (256,1); reshaped inside this method to (256,1)
        # prediction: shape (256,1); one predicted temperature per sequence
        # x: shape (256,120,14): 256 sequences, 120 time steps, 14 features
        # last_h: shape (256,16); hidden state
        # cache: list: 120 tuples when training, empty otherwise; each tuple contains 7 arrays of shape (256,16)
        prediction, (x, last_h, cache) = self.forward(x, training=True)
        # y: shape (256,1); targets reshaped into a column
        y = np.asarray(y, dtype=prediction.dtype).reshape(-1, 1)
        if y.shape != prediction.shape:
            raise ValueError('Expected one scalar target per sequence.')
        # error: shape (256,1); prediction error / gradient with respect to prediction
        error = prediction - y
        # loss: scalar metric; NumPy scalar shape () or Python float without .shape
        loss = float(np.mean(error ** 2))
        # mae: scalar metric; NumPy scalar shape () or Python float without .shape
        mae = float(np.mean(np.abs(error)))
        # p: dict: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
        p = self.params
        # grads: dict: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
        # Comprehension variable name: str dictionary key; no array shape
        # Comprehension variable value: parameter ndarray: (14,64), (16,64), (64,), (16,1), or (1,), depending on the key
        grads = {name: np.zeros_like(value) for name, value in p.items()}
        # dy: shape (256,1); prediction error / gradient with respect to prediction
        dy = 2 * error / error.size
        # grads['Wy']: (16,256) @ (256,1) -> (16,1)
        grads['Wy'] = last_h.T @ dy
        # grads['by']: (1,)
        grads['by'] = dy.sum(axis=0)
        # dh: shape (256,16); hidden-state gradient
        dh = dy @ p['Wy'].T
        # dc: shape (256,16); gradient through the cell-state path
        dc = np.zeros_like(last_h)
        # Full backpropagation through time, including both state paths.
        # t: int scalar; time-step index 0 through 119
        for t in reversed(range(x.shape[1])):
            # prev_h: shape (256,16); previous hidden state
            # prev_c: shape (256,16); previous cell state
            # i: shape (256,16); input gate
            # f: shape (256,16); forget gate
            # g: shape (256,16); candidate memory
            # o: shape (256,16); output gate
            # tc: shape (256,16); tanh of updated cell state
            prev_h, prev_c, i, f, g, o, tc = cache[t]
            # do: shape (256,16); output-gate gradient
            do = dh * tc
            # dc_total: shape (256,16); combined cell-state gradient
            dc_total = dc + dh * o * (1 - tc ** 2)
            # df: shape (256,16); forget-gate gradient
            df = dc_total * prev_c
            # di: shape (256,16); input-gate gradient
            di = dc_total * g
            # dg: shape (256,16); candidate gradient
            dg = dc_total * i
            # dz: shape (256,64); concatenated preactivation gradients
            dz = np.concatenate((di * i * (1-i), df * f * (1-f),
                                 dg * (1-g*g), do * o * (1-o)), axis=1)
            # grads['Wx']: (14,256) @ (256,64) -> (14,64); accumulated over time
            grads['Wx'] += x[:, t].T @ dz
            # grads['Wh']: (16,256) @ (256,64) -> (16,64); accumulated over time
            grads['Wh'] += prev_h.T @ dz
            # grads['b']: (64,); sum over sequences, then accumulate over time
            grads['b'] += dz.sum(axis=0)
            # dh: shape (256,16); hidden-state gradient
            dh = dz @ p['Wh'].T
            # dc: shape (256,16); gradient through the cell-state path
            dc = dc_total * f
        # Returns two scalars and a dictionary of parameter-shaped gradients
        return loss, mae, grads

    def save(self, filename, **metadata):
        # Input filename: str or Path; no array shape
        # Input metadata: dict: mean/std have shape (14,); saved sequence_length/sampling_rate/delay are integers; may be None in fit()
        # f: binary file object; no array shape
        with open(filename, 'wb') as f:
            np.savez(f, **self.params, **metadata)

    @classmethod
    def load(cls, filename):
        # Input filename: str or Path; no array shape
        # z: NpzFile container; parameter shapes: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
        # Comprehension variable name: str dictionary key; no array shape
        with np.load(filename, allow_pickle=False) as z:
            # model: LSTMRegressor object; no single array shape
            model = cls(z['Wx'].shape[0], z['Wh'].shape[0], dtype=z['Wx'].dtype)
            # model.params: dict: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
            # Comprehension variable name: str dictionary key; no array shape
            model.params = {name: z[name].copy() for name in model.params}
        return model


class Adam:
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-7):
        # Input params: dict: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
        # Input learning_rate: float scalar; no array shape
        # Input beta1: float scalar; no array shape
        # Input beta2: float scalar; no array shape
        # Input epsilon: float scalar; no array shape
        # self.lr: float scalar; no array shape
        # self.b1: float scalar; no array shape
        # self.b2: float scalar; no array shape
        # self.eps: float scalar; no array shape
        self.lr, self.b1, self.b2, self.eps = learning_rate, beta1, beta2, epsilon
        # self.m: dict of moment arrays: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
        # Comprehension variable k: str dictionary key; no array shape
        # Comprehension variable v: parameter ndarray: (14,64), (16,64), (64,), (16,1), or (1,), depending on the key
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        # self.v: dict of moment arrays: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
        # Comprehension variable k: str dictionary key; no array shape
        # Comprehension variable v: parameter ndarray: (14,64), (16,64), (64,), (16,1), or (1,), depending on the key
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        # self.t: int scalar; optimizer update count
        self.t = 0

    def step(self, params, grads):
        # Input params: dict: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
        # Input grads: dict: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
        self.t += 1
        # Keras-style Adam: epsilon is applied to the uncorrected second moment.
        # alpha: NumPy floating-point scalar; shape ()
        alpha = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        # k: str dictionary key; no array shape
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1-self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1-self.b2) * grads[k] ** 2
            params[k] -= alpha * self.m[k] / (np.sqrt(self.v[k]) + self.eps)


def evaluate(model, dataset):
    # Input model: LSTMRegressor object; no single array shape
    # Input dataset: WindowDataset object; yields input (256,120,14) and target (256,) for a full default batch
    # squared: scalar error accumulator; no batch axis
    # absolute: scalar error accumulator; no batch axis
    # count: int scalar; accumulated sample count
    squared, absolute, count = 0., 0., 0
    # x: shape (256,120,14): 256 sequences, 120 time steps, 14 features
    # y: shape (256,) for a full batch
    for x, y in dataset:
        # error: shape (256,); prediction is flattened before subtraction
        error = model.predict(x).ravel().astype(np.float64) - y
        squared += np.sum(error ** 2)
        absolute += np.sum(np.abs(error))
        count += len(y)
    if not count:
        raise ValueError('Cannot evaluate an empty dataset.')
    # Returns two scalar metrics: MSE and MAE
    return squared / count, absolute / count


def fit(model, train_dataset, val_dataset, epochs=10,
        checkpoint='jena_lstm_numpy.npz', learning_rate=0.001, metadata=None):
    # Input model: LSTMRegressor object; no single array shape
    # Input train_dataset: WindowDataset object; yields input (256,120,14) and target (256,) for a full default batch
    # Input val_dataset: WindowDataset object; yields input (256,120,14) and target (256,) for a full default batch
    # Input epochs: int scalar; default 10
    # Input checkpoint: str or Path; no array shape
    # Input learning_rate: float scalar; no array shape
    # Input metadata: dict: mean/std have shape (14,); saved sequence_length/sampling_rate/delay are integers; may be None in fit()
    # optimizer: Adam object; no array shape
    optimizer = Adam(model.params, learning_rate)
    # history: dict[str,list[float]]: 4 metrics, each with 10 entries after the default training run
    # Comprehension variable k: str dictionary key; no array shape
    history = {k: [] for k in ('loss', 'mae', 'val_loss', 'val_mae')}
    # best: scalar; best validation MSE
    best = np.inf
    # epoch: int scalar; loop index / batch number
    # Comprehension variable g: gradient array: (14,64), (16,64), (64,), (16,1), or (1,)
    for epoch in range(epochs):
        # begin: float scalar; start time
        begin = time.perf_counter()
        # squared: scalar error accumulator; no batch axis
        # absolute: scalar error accumulator; no batch axis
        # count: int scalar; accumulated sample count
        squared, absolute, count = 0., 0., 0
        # step: int scalar; loop index / batch number
        # x: shape (256,120,14): 256 sequences, 120 time steps, 14 features
        # y: shape (256,) for a full batch
        # Comprehension variable g: gradient array: (14,64), (16,64), (64,), (16,1), or (1,)
        for step, (x, y) in enumerate(train_dataset, 1):
            # loss: scalar metric; NumPy scalar shape () or Python float without .shape
            # mae: scalar metric; NumPy scalar shape () or Python float without .shape
            # grads: dict: Wx (14,64), Wh (16,64), b (64,), Wy (16,1), by (1,)
            loss, mae, grads = model.loss_and_gradients(x, y)
            # Generator variable g: gradient array: (14,64), (16,64), (64,), (16,1), or (1,)
            if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in grads.values()):
                raise FloatingPointError('Non-finite loss or gradients; check data/learning rate.')
            optimizer.step(model.params, grads)
            squared += loss * len(y)
            absolute += mae * len(y)
            count += len(y)
            if step % 100 == 0:
                print(f'  Epoch {epoch+1}: batch {step}/{len(train_dataset)}', flush=True)
        if not count:
            raise ValueError('Cannot train on an empty dataset.')
        # val_loss: scalar metric; NumPy scalar shape () or Python float without .shape
        # val_mae: scalar metric; NumPy scalar shape () or Python float without .shape
        val_loss, val_mae = evaluate(model, val_dataset)
        if not np.isfinite(val_loss):
            raise FloatingPointError('Non-finite validation loss.')
        # values: tuple of 4 scalar metrics; not an ndarray
        values = (squared/count, absolute/count, val_loss, val_mae)
        # key: str dictionary key; no array shape
        # value: scalar metric value
        for key, value in zip(history, values):
            history[key].append(value)
        # saved: bool scalar
        saved = val_loss < best
        if saved:
            # best: scalar; best validation MSE
            best = val_loss
            model.save(checkpoint, **(metadata or {}))
        print(f'Epoch {epoch+1}/{epochs} - loss: {values[0]:.4f} - mae: {values[1]:.4f}'
              f' - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}'
              f' - {time.perf_counter()-begin:.1f}s' + (' - saved best' if saved else ''), flush=True)
    return history


def load_weather(path):
    # Input path: Path object; no array shape
    # f: text file object; no array shape
    # Comprehension variable row: list[str] with 15 entries: timestamp plus 14 features
    # Comprehension variable v: str numeric field from the CSV
    with open(path, newline='', encoding='utf-8-sig') as f:
        # reader: csv.reader iterator; no array shape
        reader = csv.reader(f)
        # header: list[str] with 15 entries: timestamp plus 14 features
        header = next(reader)
        # data: shape (420451,14) for the stated CSV; timestamp column removed
        # Comprehension variable row: list[str] with 15 entries: timestamp plus 14 features
        # Comprehension variable v: str numeric field from the CSV
        data = np.array([[float(v) for v in row[1:]] for row in reader if row], dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 14 or header[2] != 'T (degC)':
        raise ValueError('Expected the original 14-feature Jena climate CSV.')
    if not np.isfinite(data).all():
        raise ValueError('CSV contains non-finite values.')
    return data


def self_test():
    # rng: np.random.Generator object; no array shape
    rng = np.random.default_rng(10)
    # model: LSTMRegressor object with 2 input features and 3 hidden units
    model = LSTMRegressor(2, 3, dtype=np.float64)
    # x: shape (2,4,2); gradient check: 2 sequences, 4 steps, 2 features
    x = rng.normal(size=(2, 4, 2))
    # y: shape (2,); gradient-check targets
    y = rng.normal(size=2)
    # _: discarded scalar result or integer loop counter
    # grads: dict: Wx (2,12), Wh (3,12), b (12,), Wy (3,1), by (1,)
    _, _, grads = model.loss_and_gradients(x, y)
    # max_error: scalar; largest gradient-check error
    max_error = 0.
    # name: str dictionary key; no array shape
    # p: parameter array: Wx (2,12), Wh (3,12), b (12,), Wy (3,1), by (1,)
    for name, p in model.params.items():
        # index: tuple of integer indices: length 2 for matrices, 1 for bias vectors
        for index in np.ndindex(p.shape):
            # old: NumPy numeric scalar; shape ()
            old = p[index]
            p[index] = old + 1e-5
            # plus: NumPy numeric scalar; shape ()
            plus = np.mean((model.predict(x).ravel() - y) ** 2)
            p[index] = old - 1e-5
            # minus: NumPy numeric scalar; shape ()
            minus = np.mean((model.predict(x).ravel() - y) ** 2)
            p[index] = old
            # numerical: NumPy numeric scalar; shape ()
            numerical = (plus - minus) / 2e-5
            # analytic: NumPy numeric scalar; shape ()
            analytic = grads[name][index]
            # max_error: scalar; largest gradient-check error
            max_error = max(max_error, abs(numerical - analytic))
            np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-7)
    # Learning test requires information from early timesteps.
    # x: shape (16,5,2); synthetic learning input
    x = rng.normal(size=(16, 5, 2))
    # y: shape (16,); synthetic learning targets
    y = x[:, 0, 0] + 0.5 * x[:, 1, 1]
    # initial: NumPy numeric scalar; shape ()
    initial = np.mean((model.predict(x).ravel()-y)**2)
    # optimizer: Adam object; no array shape
    optimizer = Adam(model.params, learning_rate=0.02)
    # _: discarded scalar result or integer loop counter
    for _ in range(300):
        # _: discarded scalar result or integer loop counter
        # grads: dict: Wx (2,12), Wh (3,12), b (12,), Wy (3,1), by (1,)
        _, _, grads = model.loss_and_gradients(x, y)
        optimizer.step(model.params, grads)
    # final: NumPy numeric scalar; shape ()
    final = np.mean((model.predict(x).ravel()-y)**2)
    assert final < initial * 0.1, (initial, final)
    # tmp: str temporary-directory path; no array shape
    with tempfile.TemporaryDirectory() as tmp:
        # path: Path object; no array shape
        path = Path(tmp) / 'model.npz'
        model.save(path)
        np.testing.assert_array_equal(model.predict(x), LSTMRegressor.load(path).predict(x))
    # data: shape (20,2); synthetic data for window-indexing checks
    data = np.arange(40).reshape(20, 2)
    # ds: WindowDataset object: 6 windows, each input (3,2) with a scalar target
    ds = WindowDataset(data, np.arange(20)+100, sequence_length=3,
                       sampling_rate=2, batch_size=4, start_index=2, end_index=12)
    # batches: list of 2 test batches: inputs (4,3,2) / targets (4,) and inputs (2,3,2) / targets (2,)
    # Six test windows: batch 1 inputs (4,3,2), targets (4,); batch 2 inputs (2,3,2), targets (2,)
    batches = list(ds)
    np.testing.assert_array_equal(batches[0][0][0], data[[2, 4, 6]])
    # Comprehension variable b: tuple (inputs,targets): shapes (4,3,2)/(4,) or (2,3,2)/(2,)
    np.testing.assert_array_equal(np.concatenate([b[1] for b in batches]), np.arange(102, 108))
    print(f'PASS: all parameter gradients; max absolute error {max_error:.2e}')
    print(f'PASS: learns early-timestep targets; MSE {initial:.4f} -> {final:.6f}')
    print('PASS: checkpoint round-trip and window indexing / partial batches')


def main():
    # parser: argparse.ArgumentParser object; no array shape
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='jena_climate_2009_2016.csv')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--checkpoint', default='jena_lstm_numpy.npz')
    parser.add_argument('--strict-splits', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    # args: argparse.Namespace: csv/checkpoint are strings; epochs/batch_size are integers; download/strict_splits/self_test are booleans
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    # path: Path object; no array shape
    path = Path(args.csv)
    if not path.exists() and args.download:
        # url: str download URL; no array shape
        url = 'https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip'
        print('Downloading Jena weather data...', flush=True)
        # tmp: str temporary-directory path; no array shape
        with tempfile.TemporaryDirectory() as tmp:
            # archive: Path object; no array shape
            archive = Path(tmp) / 'jena.zip'
            urllib.request.urlretrieve(url, archive)
            # z: zipfile.ZipFile object; no array shape
            with zipfile.ZipFile(archive) as z:
                path.write_bytes(z.read('jena_climate_2009_2016.csv'))
    if not path.exists():
        parser.error('CSV not found; supply --csv PATH or use --download.')
    # raw_data: shape (420451,14) for the stated CSV; standardization preserves this shape
    raw_data = load_weather(path)
    # temperature: shape (420451,); original temperatures in degrees Celsius
    temperature = raw_data[:, 1].copy()  # Keep targets in degrees Celsius.
    # n_train: int scalar; 210225 for 420451 records
    n_train = int(0.5 * len(raw_data))
    # n_val: int scalar; 105112 for 420451 records
    n_val = int(0.25 * len(raw_data))
    # mean: shape (14,); one training-set statistic per weather feature
    mean = raw_data[:n_train].mean(axis=0, dtype=np.float64).astype(np.float32)
    # std: shape (14,); one training-set statistic per weather feature
    std = raw_data[:n_train].std(axis=0, dtype=np.float64).astype(np.float32)
    # std: shape (14,); one training-set statistic per weather feature
    std = np.where(std == 0, 1, std)
    # raw_data: shape (420451,14) for the stated CSV; standardization preserves this shape
    raw_data = (raw_data - mean) / std
    # sampling_rate: int scalar; default 6 rows between observations
    # sequence_length: int scalar; default 120 time steps
    sampling_rate, sequence_length = 6, 120
    # delay: int scalar; 858 rows by default
    delay = sampling_rate * (sequence_length + 24 - 1)
    # span: int scalar; 714 rows from the first to the last input observation
    span = sampling_rate * (sequence_length - 1)
    # data: shape (419593,14) for the stated 420451-row CSV after removing 858 trailing rows
    # targets: shape (419593,) for the stated CSV; one target per possible starting row
    data, targets = raw_data[:-delay], temperature[delay:]
    # boundaries: tuple of 3 integer pairs: (0,210225), (210225,315337), (315337,420451)
    boundaries = ((0, n_train), (n_train, n_train+n_val),
                  (n_train+n_val, len(raw_data)))
    # datasets: list of 3 WindowDataset objects after construction
    datasets = []
    # j: int scalar; loop index / batch number
    # start: int scalar; a partition or dataset boundary
    # end: int scalar; a partition or dataset boundary
    for j, (start, end) in enumerate(boundaries):
        # end_index is exclusive; strict mode also bounds the future labels.
        # end_index: int scalar or None; exclusive input boundary
        end_index = min(len(data), end - delay + span if args.strict_splits else end)
        datasets.append(WindowDataset(data, targets, sequence_length, sampling_rate,
                        args.batch_size, start, end_index, shuffle=(j == 0)))
    # train_dataset: WindowDataset object; yields input (256,120,14) and target (256,) for a full default batch
    # val_dataset: WindowDataset object; yields input (256,120,14) and target (256,) for a full default batch
    # test_dataset: WindowDataset object; yields input (256,120,14) and target (256,) for a full default batch
    train_dataset, val_dataset, test_dataset = datasets
    # Comprehension variable ds: WindowDataset object; yields input (256,120,14) and target (256,) for a full default batch
    print('Windows (train/validation/test):', *(len(ds.starts) for ds in datasets), flush=True)
    # model: LSTMRegressor object; no single array shape
    model = LSTMRegressor(raw_data.shape[-1], hidden_size=16)
    # history: dict[str,list[float]]: 4 metrics, each with 10 entries after the default training run
    history = fit(model, train_dataset, val_dataset, epochs=args.epochs,
                  checkpoint=args.checkpoint,
                  metadata={'mean': mean, 'std': std, 'sequence_length': sequence_length,
                            'sampling_rate': sampling_rate, 'delay': delay})
    # model: LSTMRegressor object; no single array shape
    model = LSTMRegressor.load(args.checkpoint)
    # test_loss: scalar metric; NumPy scalar shape () or Python float without .shape
    # test_mae: scalar metric; NumPy scalar shape () or Python float without .shape
    test_loss, test_mae = evaluate(model, test_dataset)
    print(f'Test MSE: {test_loss:.4f}')
    print(f'Test MAE: {test_mae:.2f} °C')
    # history is a dict with loss, mae, val_loss and val_mae lists.
    return model, history


if __name__ == '__main__':
    main()
