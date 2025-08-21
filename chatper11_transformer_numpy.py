# imdb dataset: dataset: https:/ /ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
import os
import re
import math
import json
import random
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------
# Data utilities (tokenize IMDB)
# ----------------------------

def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"<.*?>", " ", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_vocab(texts: List[str], vocab_size: int) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(clean_text(t).split())
    # Reserve 0 for PAD, 1 for OOV
    most_common = [(w, c) for w, c in counter.most_common(vocab_size - 2)]
    stoi = {w: i + 2 for i, (w, _) in enumerate(most_common)}
    stoi["<PAD>"] = 0
    stoi["<OOV>"] = 1
    return stoi


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi.get(w, 1) for w in clean_text(text).split()]


def pad_batch(batch: List[List[int]], maxlen: int) -> np.ndarray:
    x = np.zeros((len(batch), maxlen), dtype=np.int64)
    for i, seq in enumerate(batch):
        seq = seq[:maxlen]
        x[i, : len(seq)] = np.array(seq, dtype=np.int64)
    return x


def load_imdb(path: str) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    for split in ["train", "test"]:
        for label in ["pos", "neg"]:
            d = os.path.join(path, split, label)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.endswith(".txt"):
                    continue
                with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(1 if label == "pos" else 0)
    return texts, labels


# ----------------------------
# Layers (NumPy, manual autodiff)
# ----------------------------

class Parameter:
    def __init__(self, value: np.ndarray, name: str = ""):
        self.value = value
        self.grad = np.zeros_like(value)
        self.name = name

    def zero_grad(self):
        self.grad[...] = 0.0


class Embedding:
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W = Parameter(rng.normal(0, 0.02, size=(vocab_size, embed_dim)).astype(np.float32), "emb/W")
        self.cache = None

    def forward(self, x_ids: np.ndarray) -> np.ndarray:
        # x_ids: (B, T)
        out = self.W.value[x_ids]  # (B, T, D)
        self.cache = x_ids
        return out

    def backward(self, dout: np.ndarray):
        x_ids = self.cache
        # Accumulate gradients per index
        np.add.at(self.W.grad, x_ids, dout)

    @property
    def params(self):
        return [self.W]


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = Parameter(np.ones((dim,), dtype=np.float32), "ln/gamma")
        self.beta = Parameter(np.zeros((dim,), dtype=np.float32), "ln/beta")
        self.eps = eps
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (..., D)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma.value * xhat + self.beta.value
        self.cache = (xhat, var, mean, x)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        xhat, var, mean, x = self.cache
        N = x.shape[-1]
        dgamma = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
        dbeta = dout.sum(axis=tuple(range(dout.ndim - 1)))
        self.gamma.grad += dgamma
        self.beta.grad += dbeta
        # Backprop for LayerNorm
        std_inv = 1.0 / np.sqrt(var + 1e-5)
        dxhat = dout * self.gamma.value
        dx = (1.0 / N) * std_inv * (
            N * dxhat - dxhat.sum(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
        )
        return dx

    @property
    def params(self):
        return [self.gamma, self.beta]


class Dropout:
    def __init__(self, p: float, seed: int = 123):
        self.p = p
        self.seed = seed
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p <= 0.0:
            self.mask = None
            return x
        rng = np.random.default_rng(self.seed)
        self.mask = (rng.uniform(size=x.shape) >= self.p).astype(np.float32)
        return x * self.mask / (1.0 - self.p)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return dout
        return dout * self.mask / (1.0 - self.p)

    @property
    def params(self):
        return []


class Dense:
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, seed: int = 0, name_prefix: str = "dense"):
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6 / (in_dim + out_dim))
        self.W = Parameter(rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32), f"{name_prefix}/W")
        self.b = Parameter(np.zeros((out_dim,), dtype=np.float32), f"{name_prefix}/b") if bias else None
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W.value
        if self.b is not None:
            out = out + self.b.value
        self.cache = x
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache
        self.W.grad += x.reshape(-1, x.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
        if self.b is not None:
            self.b.grad += dout.sum(axis=tuple(range(dout.ndim - 1)))
        dx = dout @ self.W.value.T
        return dx

    @property
    def params(self):
        return [p for p in [self.W, self.b] if p is not None]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def drelu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class FeedForward:
    def __init__(self, dim: int, hidden: int, seed: int = 0):
        self.lin1 = Dense(dim, hidden, name_prefix="ff/lin1", seed=seed)
        self.lin2 = Dense(hidden, dim, name_prefix="ff/lin2", seed=seed + 1)
        self.cache_a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = self.lin1.forward(x)
        h = relu(a)
        self.cache_a = a
        out = self.lin2.forward(h)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dh = self.lin2.backward(dout)
        da = dh * drelu(self.cache_a)
        dx = self.lin1.backward(da)
        return dx

    @property
    def params(self):
        return self.lin1.params + self.lin2.params


class MultiHeadSelfAttention:
    def __init__(self, dim: int, num_heads: int, seed: int = 0):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.Wq = Dense(dim, dim, name_prefix="attn/Wq", seed=seed)
        self.Wk = Dense(dim, dim, name_prefix="attn/Wk", seed=seed + 1)
        self.Wv = Dense(dim, dim, name_prefix="attn/Wv", seed=seed + 2)
        self.Wo = Dense(dim, dim, name_prefix="attn/Wo", seed=seed + 3)
        self.cache = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        # x: (B, T, D) -> (B, H, T, Hd)
        B, T, D = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        # x: (B, H, T, Hd) -> (B, T, D)
        B, H, T, Hd = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(B, T, H * Hd)
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (B, T, D)
        Q = self._split_heads(self.Wq.forward(x))  # (B,H,T,Hd)
        K = self._split_heads(self.Wk.forward(x))  # (B,H,T,Hd)
        V = self._split_heads(self.Wv.forward(x))  # (B,H,T,Hd)
        scale = 1.0 / math.sqrt(self.head_dim)
        # Scores: (B,H,T,T)
        scores = (Q @ K.transpose(0, 1, 3, 2)) * scale
        # Softmax along last axis
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        out_heads = attn @ V  # (B,H,T,Hd)
        out = self._merge_heads(out_heads)  # (B,T,D)
        out = self.Wo.forward(out)
        self.cache = (x, Q, K, V, attn, out_heads)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, Q, K, V, attn, out_heads = self.cache
        # Back through Wo
        dmerge = self.Wo.backward(dout)  # (B,T,D)
        # Split heads grad
        B, T, D = dmerge.shape
        dheads = dmerge.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B,H,T,Hd)
        # attn @ V -> out_heads
        dattn = dheads @ V.transpose(0, 1, 3, 2)  # (B,H,T,T)
        dV = attn.transpose(0, 1, 3, 2) @ dheads  # (B,H,T,Hd)
        # softmax backward on scores
        # attn shape (B,H,T,T); dattn same
        dscores = np.empty_like(attn)
        for b in range(B):
            for h in range(self.num_heads):
                A = attn[b, h]  # (T,T)
                dA = dattn[b, h]  # (T,T)
                # Jacobian-vector product for softmax row-wise
                # For each query position t, softmax over keys axis (-1)
                # Use: dS = A * (dA - sum(dA*A))
                s = (dA * A).sum(axis=-1, keepdims=True)
                dscores[b, h] = A * (dA - s)
        scale = 1.0 / math.sqrt(self.head_dim)
        dscores *= scale
        # scores = Q @ K^T
        dQ = dscores @ K  # (B,H,T,Hd)
        dK = dscores.transpose(0, 1, 3, 2) @ Q  # (B,H,T,Hd)
        # Back to input projections
        # Merge heads grads back to (B,T,D)
        def merge(xh):
            return xh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dQm = merge(dQ)
        dKm = merge(dK)
        dVm = merge(dV)
        dx_q = self.Wq.backward(dQm)
        dx_k = self.Wk.backward(dKm)
        dx_v = self.Wv.backward(dVm)
        dx = dx_q + dx_k + dx_v
        return dx

    @property
    def params(self):
        return self.Wq.params + self.Wk.params + self.Wv.params + self.Wo.params


class GlobalMaxPool1D:
    def __init__(self):
        self.cache_idx = None
        self.cache_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (B, T, D) -> (B, D)
        self.cache_shape = x.shape
        idx = x.argmax(axis=1)  # (B, D) indices along T
        self.cache_idx = idx
        B, T, D = x.shape
        out = x[np.arange(B)[:, None], idx, np.arange(D)[None, :]]  # fancy index
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        B, T, D = self.cache_shape
        dx = np.zeros((B, T, D), dtype=np.float32)
        idx = self.cache_idx
        dx[np.arange(B)[:, None], idx, np.arange(D)[None, :]] = dout
        return dx

    @property
    def params(self):
        return []


class TransformerEncoderBlock:
    def __init__(self, dim: int, hidden: int, num_heads: int, dropout_p: float = 0.0, seed: int = 0):
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, seed=seed)
        self.drop1 = Dropout(dropout_p, seed=seed + 11)
        self.ln2 = LayerNorm(dim)
        self.ff = FeedForward(dim, hidden, seed=seed + 20)
        self.drop2 = Dropout(dropout_p, seed=seed + 21)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.drop1.training = training
        self.drop2.training = training
        # Pre-norm Transformer
        h = self.ln1.forward(x)
        h = self.attn.forward(h)
        h = self.drop1.forward(h)
        x = x + h
        h2 = self.ln2.forward(x)
        h2 = self.ff.forward(h2)
        h2 = self.drop2.forward(h2)
        out = x + h2
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Corresponds to forward residuals order
        dh2 = dout
        dh2 = self.drop2.backward(dh2)
        dh2 = self.ff.backward(dh2)
        dx2 = self.ln2.backward(dh2)
        dx = dout + dx2  # gradient flows through residual add

        dh1 = dx
        dh1 = self.drop1.backward(dh1)
        dh1 = self.attn.backward(dh1)
        dx1 = self.ln1.backward(dh1)
        dx_total = dx1 + dout  # through first residual
        return dx_total

    @property
    def params(self):
        return self.ln1.params + self.attn.params + self.ln2.params + self.ff.params


# ----------------------------
# Model
# ----------------------------

class NumPyTransformerClassifier:
    def __init__(self, vocab_size=20000, embed_dim=256, num_heads=2, dense_dim=32, dropout_p=0.5, seed: int = 0):
        self.embed = Embedding(vocab_size, embed_dim, seed=seed)
        self.encoder = TransformerEncoderBlock(embed_dim, dense_dim, num_heads, dropout_p=dropout_p, seed=seed)
        self.pool = GlobalMaxPool1D()
        self.drop = Dropout(dropout_p, seed=seed + 100)
        self.out = Dense(embed_dim, 1, name_prefix="out", seed=seed + 101)
        self.training = True

    @property
    def params(self):
        ps = []
        for layer in [self.embed, self.encoder, self.out]:
            ps += layer.params
        return ps

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def forward(self, x_ids: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training
        h = self.embed.forward(x_ids)
        h = self.encoder.forward(h, training=training)
        h = self.pool.forward(h)
        self.drop.training = training
        h = self.drop.forward(h)
        logits = self.out.forward(h)  # (B,1)
        probs = sigmoid(logits)
        self._cache_logits = logits
        self._cache_probs = probs
        self._cache_labels = None
        return probs

    def backward(self, y_true: np.ndarray):
        # y_true: (B,)
        probs = self._cache_probs.reshape(-1, 1)
        y = y_true.reshape(-1, 1).astype(np.float32)
        # BCE derivative wrt logits: dL/dz = (p - y)
        dlogits = (probs - y) / y.shape[0]  # average over batch
        dh = self.out.backward(dlogits)
        dh = self.drop.backward(dh)
        dh = self.pool.backward(dh)
        dh = self.encoder.backward(dh)
        self.embed.backward(dh)

    def loss_and_acc(self, y_true: np.ndarray) -> Tuple[float, float]:
        probs = self._cache_probs.reshape(-1)
        y = y_true.astype(np.float32)
        eps = 1e-7
        bce = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        preds = (probs >= 0.5).astype(np.int32)
        acc = (preds == y_true).mean()
        return float(bce), float(acc)


# ----------------------------
# Optimizer: RMSprop
# ----------------------------

class RMSprop:
    def __init__(self, params: List[Parameter], lr=1e-3, rho=0.9, eps=1e-8):
        self.params = params
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.cache = {id(p): np.zeros_like(p.value) for p in params}

    def step(self):
        for p in self.params:
            s = self.cache[id(p)]
            s[...] = self.rho * s + (1 - self.rho) * (p.grad ** 2)
            p.value[...] = p.value - self.lr * p.grad / (np.sqrt(s) + self.eps)


# ----------------------------
# Training utilities
# ----------------------------

def iterate_minibatches(X_ids: List[List[int]], y: np.ndarray, batch_size: int, maxlen: int, shuffle: bool = True):
    idxs = np.arange(len(X_ids))
    if shuffle:
        np.random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch_idx = idxs[start:start + batch_size]
        batch = [X_ids[i] for i in batch_idx]
        Xb = pad_batch(batch, maxlen=maxlen)
        yb = y[batch_idx]
        yield Xb, yb


def train_model(
    model: NumPyTransformerClassifier,
    X_train_ids: List[List[int]],
    y_train: np.ndarray,
    X_val_ids: List[List[int]],
    y_val: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
    maxlen: int = 256,
    lr: float = 1e-3,
    save_path: str = "transformer_encoder_np.npz",
):
    opt = RMSprop(model.params, lr=lr)
    best_val = 0.0
    for ep in range(1, epochs + 1):
        # Train
        model.zero_grad()
        losses = []
        accs = []
        for Xb, yb in iterate_minibatches(X_train_ids, y_train, batch_size, maxlen, shuffle=True):
            model.zero_grad()
            probs = model.forward(Xb, training=True)
            loss, acc = model.loss_and_acc(yb)
            losses.append(loss)
            accs.append(acc)
            model.backward(yb)
            opt.step()
        train_loss = float(np.mean(losses))
        train_acc = float(np.mean(accs))
        # Validate
        val_losses = []
        val_accs = []
        for Xb, yb in iterate_minibatches(X_val_ids, y_val, batch_size, maxlen, shuffle=False):
            _ = model.forward(Xb, training=False)
            loss, acc = model.loss_and_acc(yb)
            val_losses.append(loss)
            val_accs.append(acc)
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0
        print(f"Epoch {ep:02d}: train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            save_weights(model, save_path)
            print(f"  \u2191 Saved best to {save_path} (val_acc={best_val:.3f})")


def evaluate_model(model: NumPyTransformerClassifier, X_ids: List[List[int]], y: np.ndarray, batch_size: int = 64, maxlen: int = 256) -> Tuple[float, float]:
    losses, accs = [], []
    for Xb, yb in iterate_minibatches(X_ids, y, batch_size, maxlen, shuffle=False):
        _ = model.forward(Xb, training=False)
        loss, acc = model.loss_and_acc(yb)
        losses.append(loss)
        accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))


# ----------------------------
# Save / Load
# ----------------------------

def save_weights(model: NumPyTransformerClassifier, path: str):
    data = {}
    for p in model.params:
        data[p.name] = p.value
    np.savez(path, **data)


def load_weights(model: NumPyTransformerClassifier, path: str):
    zz = np.load(path)
    name_to_param = {p.name: p for p in model.params}
    for k in zz.files:
        if k in name_to_param:
            name_to_param[k].value[...] = zz[k]
        else:
            print(f"[warn] extra weight in file: {k}")


# ----------------------------
# Example main (end-to-end)
# ----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--imdb_path", type=str, required=False, default="./aclImdb")
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dense_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weights", type=str, default="transformer_encoder_np.npz")
    args = parser.parse_args()

    print("Loading IMDB...")
    texts, labels = load_imdb(args.imdb_path)
    labels = np.array(labels, dtype=np.int32)

    # Simple split: 40k train, 5k val, 5k test (IMDB has 50k total)
    idx = np.arange(len(texts))
    np.random.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = labels[idx]

    stoi = build_vocab(texts, args.vocab_size)
    ids_all = [encode(t, stoi) for t in texts]

    n = len(ids_all)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    X_train, y_train = ids_all[:n_train], labels[:n_train]
    X_val, y_val = ids_all[n_train:n_train + n_val], labels[n_train:n_train + n_val]
    X_test, y_test = ids_all[n_train + n_val :], labels[n_train + n_val :]

    print(f"Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

    model = NumPyTransformerClassifier(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dense_dim=args.dense_dim,
        dropout_p=args.dropout,
    )

    train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        lr=args.lr,
        save_path=args.weights,
    )

    print("Loading best weights and evaluating on test set...")
    load_weights(model, args.weights)
    test_loss, test_acc = evaluate_model(model, X_test, y_test, batch_size=args.batch_size, maxlen=args.maxlen)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.3f}")
