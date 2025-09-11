# imdb dataset: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# import os
# import re
# import math
# import json
# import random
# from collections import Counter
# from typing import List, Tuple, Dict, Any
# from load_data import load_imdb_with_cache

# import numpy as np

# # ----------------------------
# # Data utilities (tokenize IMDB)
# # ----------------------------

# def clean_text(t: str) -> str:
#     t = t.lower()
#     t = re.sub(r"<.*?>", " ", t)
#     t = re.sub(r"[^a-z0-9' ]+", " ", t)
#     t = re.sub(r"\s+", " ", t).strip()
#     return t


# def build_vocab(texts: List[str], vocab_size: int) -> Dict[str, int]:
#     counter = Counter()
#     for t in texts:
#         counter.update(clean_text(t).split())
#     # Reserve 0 for PAD, 1 for OOV
#     most_common = [(w, c) for w, c in counter.most_common(vocab_size - 2)]
#     stoi = {w: i + 2 for i, (w, _) in enumerate(most_common)}
#     stoi["<PAD>"] = 0
#     stoi["<OOV>"] = 1
#     return stoi


# def encode(text: str, stoi: Dict[str, int]) -> List[int]:
#     return [stoi.get(w, 1) for w in clean_text(text).split()]


# def pad_batch(batch: List[List[int]], maxlen: int) -> np.ndarray:
#     x = np.zeros((len(batch), maxlen), dtype=np.int64)
#     for i, seq in enumerate(batch):
#         seq = seq[:maxlen]
#         x[i, : len(seq)] = np.array(seq, dtype=np.int64)
#     return x


# def load_imdb(path: str) -> Tuple[List[str], List[int]]:
#     texts, labels = [], []
#     for split in ["train", "test"]:
#         for label in ["pos", "neg"]:
#             d = os.path.join(path, split, label)
#             if not os.path.isdir(d):
#                 continue
#             for fn in os.listdir(d):
#                 if not fn.endswith(".txt"):
#                     continue
#                 with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
#                     texts.append(f.read())
#                 labels.append(1 if label == "pos" else 0)
#     return texts, labels


# # ----------------------------
# # Layers (NumPy, manual autodiff)
# # ----------------------------

# class Parameter:
#     def __init__(self, value: np.ndarray, name: str = ""):
#         self.value = value
#         self.grad = np.zeros_like(value)
#         self.name = name

#     def zero_grad(self):
#         self.grad[...] = 0.0


# class Embedding:
#     def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
#         rng = np.random.default_rng(seed)
#         self.W = Parameter(rng.normal(0, 0.02, size=(vocab_size, embed_dim)).astype(np.float32), "emb/W")
#         self.cache = None

#     def forward(self, x_ids: np.ndarray) -> np.ndarray:
#         # x_ids: (B, T)
#         out = self.W.value[x_ids]  # (B, T, D)
#         self.cache = x_ids
#         return out

#     def backward(self, dout: np.ndarray):
#         x_ids = self.cache
#         # Accumulate gradients per index
#         np.add.at(self.W.grad, x_ids, dout)

#     @property
#     def params(self):
#         return [self.W]


# class LayerNorm:
#     def __init__(self, dim: int, eps: float = 1e-5):
#         self.gamma = Parameter(np.ones((dim,), dtype=np.float32), "ln/gamma")
#         self.beta = Parameter(np.zeros((dim,), dtype=np.float32), "ln/beta")
#         self.eps = eps
#         self.cache = None

#     def forward(self, x: np.ndarray) -> np.ndarray:
#         # x: (..., D)
#         mean = x.mean(axis=-1, keepdims=True)
#         var = x.var(axis=-1, keepdims=True)
#         xhat = (x - mean) / np.sqrt(var + self.eps)
#         out = self.gamma.value * xhat + self.beta.value
#         self.cache = (xhat, var, mean, x)
#         return out

#     def backward(self, dout: np.ndarray) -> np.ndarray:
#         xhat, var, mean, x = self.cache
#         N = x.shape[-1]
#         dgamma = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
#         dbeta = dout.sum(axis=tuple(range(dout.ndim - 1)))
#         self.gamma.grad += dgamma
#         self.beta.grad += dbeta
#         # Backprop for LayerNorm
#         std_inv = 1.0 / np.sqrt(var + 1e-5)
#         dxhat = dout * self.gamma.value
#         dx = (1.0 / N) * std_inv * (
#             N * dxhat - dxhat.sum(axis=-1, keepdims=True)
#             - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
#         )
#         return dx

#     @property
#     def params(self):
#         return [self.gamma, self.beta]


# class Dropout:
#     def __init__(self, p: float, seed: int = 123):
#         self.p = p
#         self.seed = seed
#         self.mask = None
#         self.training = True

#     def forward(self, x: np.ndarray) -> np.ndarray:
#         if not self.training or self.p <= 0.0:
#             self.mask = None
#             return x
#         rng = np.random.default_rng(self.seed)
#         self.mask = (rng.uniform(size=x.shape) >= self.p).astype(np.float32)
#         return x * self.mask / (1.0 - self.p)

#     def backward(self, dout: np.ndarray) -> np.ndarray:
#         if self.mask is None:
#             return dout
#         return dout * self.mask / (1.0 - self.p)

#     @property
#     def params(self):
#         return []


# class Dense:
#     def __init__(self, in_dim: int, out_dim: int, bias: bool = True, seed: int = 0, name_prefix: str = "dense"):
#         rng = np.random.default_rng(seed)
#         limit = math.sqrt(6 / (in_dim + out_dim))
#         self.W = Parameter(rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32), f"{name_prefix}/W")
#         self.b = Parameter(np.zeros((out_dim,), dtype=np.float32), f"{name_prefix}/b") if bias else None
#         self.cache = None

#     def forward(self, x: np.ndarray) -> np.ndarray:
#         out = x @ self.W.value
#         if self.b is not None:
#             out = out + self.b.value
#         self.cache = x
#         return out

#     def backward(self, dout: np.ndarray) -> np.ndarray:
#         x = self.cache
#         self.W.grad += x.reshape(-1, x.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
#         if self.b is not None:
#             self.b.grad += dout.sum(axis=tuple(range(dout.ndim - 1)))
#         dx = dout @ self.W.value.T
#         return dx

#     @property
#     def params(self):
#         return [p for p in [self.W, self.b] if p is not None]


# def relu(x: np.ndarray) -> np.ndarray:
#     return np.maximum(x, 0.0)


# def drelu(x: np.ndarray) -> np.ndarray:
#     return (x > 0).astype(np.float32)


# def sigmoid(x: np.ndarray) -> np.ndarray:
#     return 1.0 / (1.0 + np.exp(-x))


# class FeedForward:
#     def __init__(self, dim: int, hidden: int, seed: int = 0):
#         self.lin1 = Dense(dim, hidden, name_prefix="ff/lin1", seed=seed)
#         self.lin2 = Dense(hidden, dim, name_prefix="ff/lin2", seed=seed + 1)
#         self.cache_a = None

#     def forward(self, x: np.ndarray) -> np.ndarray:
#         a = self.lin1.forward(x)
#         h = relu(a)
#         self.cache_a = a
#         out = self.lin2.forward(h)
#         return out

#     def backward(self, dout: np.ndarray) -> np.ndarray:
#         dh = self.lin2.backward(dout)
#         da = dh * drelu(self.cache_a)
#         dx = self.lin1.backward(da)
#         return dx

#     @property
#     def params(self):
#         return self.lin1.params + self.lin2.params


# class MultiHeadSelfAttention:
#     def __init__(self, dim: int, num_heads: int, seed: int = 0):
#         assert dim % num_heads == 0
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.Wq = Dense(dim, dim, name_prefix="attn/Wq", seed=seed)
#         self.Wk = Dense(dim, dim, name_prefix="attn/Wk", seed=seed + 1)
#         self.Wv = Dense(dim, dim, name_prefix="attn/Wv", seed=seed + 2)
#         self.Wo = Dense(dim, dim, name_prefix="attn/Wo", seed=seed + 3)
#         self.cache = None

#     def _split_heads(self, x: np.ndarray) -> np.ndarray:
#         # x: (B, T, D) -> (B, H, T, Hd)
#         B, T, D = x.shape
#         x = x.reshape(B, T, self.num_heads, self.head_dim)
#         return x.transpose(0, 2, 1, 3)

#     def _merge_heads(self, x: np.ndarray) -> np.ndarray:
#         # x: (B, H, T, Hd) -> (B, T, D)
#         B, H, T, Hd = x.shape
#         x = x.transpose(0, 2, 1, 3).reshape(B, T, H * Hd)
#         return x

#     def forward(self, x: np.ndarray) -> np.ndarray:
#         # x: (B, T, D)
#         Q = self._split_heads(self.Wq.forward(x))  # (B,H,T,Hd)
#         K = self._split_heads(self.Wk.forward(x))  # (B,H,T,Hd)
#         V = self._split_heads(self.Wv.forward(x))  # (B,H,T,Hd)
#         scale = 1.0 / math.sqrt(self.head_dim)
#         # Scores: (B,H,T,T)
#         scores = (Q @ K.transpose(0, 1, 3, 2)) * scale
#         # Softmax along last axis
#         scores_max = scores.max(axis=-1, keepdims=True)
#         exp_scores = np.exp(scores - scores_max)
#         attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
#         out_heads = attn @ V  # (B,H,T,Hd)
#         out = self._merge_heads(out_heads)  # (B,T,D)
#         out = self.Wo.forward(out)
#         self.cache = (x, Q, K, V, attn, out_heads)
#         return out

#     def backward(self, dout: np.ndarray) -> np.ndarray:
#         x, Q, K, V, attn, out_heads = self.cache
#         # Back through Wo
#         dmerge = self.Wo.backward(dout)  # (B,T,D)
#         # Split heads grad
#         B, T, D = dmerge.shape
#         dheads = dmerge.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B,H,T,Hd)
#         # attn @ V -> out_heads
#         dattn = dheads @ V.transpose(0, 1, 3, 2)  # (B,H,T,T)
#         dV = attn.transpose(0, 1, 3, 2) @ dheads  # (B,H,T,Hd)
#         # softmax backward on scores
#         # attn shape (B,H,T,T); dattn same
#         dscores = np.empty_like(attn)
#         for b in range(B):
#             for h in range(self.num_heads):
#                 A = attn[b, h]  # (T,T)
#                 dA = dattn[b, h]  # (T,T)
#                 # Jacobian-vector product for softmax row-wise
#                 # For each query position t, softmax over keys axis (-1)
#                 # Use: dS = A * (dA - sum(dA*A))
#                 s = (dA * A).sum(axis=-1, keepdims=True)
#                 dscores[b, h] = A * (dA - s)
#         scale = 1.0 / math.sqrt(self.head_dim)
#         dscores *= scale
#         # scores = Q @ K^T
#         dQ = dscores @ K  # (B,H,T,Hd)
#         dK = dscores.transpose(0, 1, 3, 2) @ Q  # (B,H,T,Hd)
#         # Back to input projections
#         # Merge heads grads back to (B,T,D)
#         def merge(xh):
#             return xh.transpose(0, 2, 1, 3).reshape(B, T, D)
#         dQm = merge(dQ)
#         dKm = merge(dK)
#         dVm = merge(dV)
#         dx_q = self.Wq.backward(dQm)
#         dx_k = self.Wk.backward(dKm)
#         dx_v = self.Wv.backward(dVm)
#         dx = dx_q + dx_k + dx_v
#         return dx

#     @property
#     def params(self):
#         return self.Wq.params + self.Wk.params + self.Wv.params + self.Wo.params


# class GlobalMaxPool1D:
#     def __init__(self):
#         self.cache_idx = None
#         self.cache_shape = None

#     def forward(self, x: np.ndarray) -> np.ndarray:
#         # x: (B, T, D) -> (B, D)
#         self.cache_shape = x.shape
#         idx = x.argmax(axis=1)  # (B, D) indices along T
#         self.cache_idx = idx
#         B, T, D = x.shape
#         out = x[np.arange(B)[:, None], idx, np.arange(D)[None, :]]  # fancy index
#         return out

#     def backward(self, dout: np.ndarray) -> np.ndarray:
#         B, T, D = self.cache_shape
#         dx = np.zeros((B, T, D), dtype=np.float32)
#         idx = self.cache_idx
#         dx[np.arange(B)[:, None], idx, np.arange(D)[None, :]] = dout
#         return dx

#     @property
#     def params(self):
#         return []


# class TransformerEncoderBlock:
#     def __init__(self, dim: int, hidden: int, num_heads: int, dropout_p: float = 0.0, seed: int = 0):
#         self.ln1 = LayerNorm(dim)
#         self.attn = MultiHeadSelfAttention(dim, num_heads, seed=seed)
#         self.drop1 = Dropout(dropout_p, seed=seed + 11)
#         self.ln2 = LayerNorm(dim)
#         self.ff = FeedForward(dim, hidden, seed=seed + 20)
#         self.drop2 = Dropout(dropout_p, seed=seed + 21)

#     def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
#         self.drop1.training = training
#         self.drop2.training = training
#         # Pre-norm Transformer
#         h = self.ln1.forward(x)
#         h = self.attn.forward(h)
#         h = self.drop1.forward(h)
#         x = x + h
#         h2 = self.ln2.forward(x)
#         h2 = self.ff.forward(h2)
#         h2 = self.drop2.forward(h2)
#         out = x + h2
#         return out

#     def backward(self, dout: np.ndarray) -> np.ndarray:
#         # Corresponds to forward residuals order
#         dh2 = dout
#         dh2 = self.drop2.backward(dh2)
#         dh2 = self.ff.backward(dh2)
#         dx2 = self.ln2.backward(dh2)
#         dx = dout + dx2  # gradient flows through residual add

#         dh1 = dx
#         dh1 = self.drop1.backward(dh1)
#         dh1 = self.attn.backward(dh1)
#         dx1 = self.ln1.backward(dh1)
#         dx_total = dx1 + dout  # through first residual
#         return dx_total

#     @property
#     def params(self):
#         return self.ln1.params + self.attn.params + self.ln2.params + self.ff.params


# # ----------------------------
# # Model
# # ----------------------------

# class NumPyTransformerClassifier:
#     def __init__(self, vocab_size=20000, embed_dim=256, num_heads=2, dense_dim=32, dropout_p=0.5, seed: int = 0):
#         self.embed = Embedding(vocab_size, embed_dim, seed=seed)
#         self.encoder = TransformerEncoderBlock(embed_dim, dense_dim, num_heads, dropout_p=dropout_p, seed=seed)
#         self.pool = GlobalMaxPool1D()
#         self.drop = Dropout(dropout_p, seed=seed + 100)
#         self.out = Dense(embed_dim, 1, name_prefix="out", seed=seed + 101)
#         self.training = True

#     @property
#     def params(self):
#         ps = []
#         for layer in [self.embed, self.encoder, self.out]:
#             ps += layer.params
#         return ps

#     def zero_grad(self):
#         for p in self.params:
#             p.zero_grad()

#     def forward(self, x_ids: np.ndarray, training: bool = True) -> np.ndarray:
#         self.training = training
#         h = self.embed.forward(x_ids)
#         h = self.encoder.forward(h, training=training)
#         h = self.pool.forward(h)
#         self.drop.training = training
#         h = self.drop.forward(h)
#         logits = self.out.forward(h)  # (B,1)
#         probs = sigmoid(logits)
#         self._cache_logits = logits
#         self._cache_probs = probs
#         self._cache_labels = None
#         return probs

#     def backward(self, y_true: np.ndarray):
#         # y_true: (B,)
#         probs = self._cache_probs.reshape(-1, 1)
#         y = y_true.reshape(-1, 1).astype(np.float32)
#         # BCE derivative wrt logits: dL/dz = (p - y)
#         dlogits = (probs - y) / y.shape[0]  # average over batch
#         dh = self.out.backward(dlogits)
#         dh = self.drop.backward(dh)
#         dh = self.pool.backward(dh)
#         dh = self.encoder.backward(dh)
#         self.embed.backward(dh)

#     def loss_and_acc(self, y_true: np.ndarray) -> Tuple[float, float]:
#         probs = self._cache_probs.reshape(-1)
#         y = y_true.astype(np.float32)
#         eps = 1e-7
#         bce = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
#         preds = (probs >= 0.5).astype(np.int32)
#         acc = (preds == y_true).mean()
#         return float(bce), float(acc)


# # ----------------------------
# # Optimizer: RMSprop
# # ----------------------------

# class RMSprop:
#     def __init__(self, params: List[Parameter], lr=1e-3, rho=0.9, eps=1e-8):
#         self.params = params
#         self.lr = lr
#         self.rho = rho
#         self.eps = eps
#         self.cache = {id(p): np.zeros_like(p.value) for p in params}

#     def step(self):
#         for p in self.params:
#             s = self.cache[id(p)]
#             s[...] = self.rho * s + (1 - self.rho) * (p.grad ** 2)
#             p.value[...] = p.value - self.lr * p.grad / (np.sqrt(s) + self.eps)


# # ----------------------------
# # Training utilities
# # ----------------------------

# def iterate_minibatches(X_ids: List[List[int]], y: np.ndarray, batch_size: int, maxlen: int, shuffle: bool = True):
#     idxs = np.arange(len(X_ids))
#     if shuffle:
#         np.random.shuffle(idxs)
#     for start in range(0, len(idxs), batch_size):
#         batch_idx = idxs[start:start + batch_size]
#         batch = [X_ids[i] for i in batch_idx]
#         Xb = pad_batch(batch, maxlen=maxlen)
#         yb = y[batch_idx]
#         yield Xb, yb


# def train_model(
#     model: NumPyTransformerClassifier,
#     X_train_ids: List[List[int]],
#     y_train: np.ndarray,
#     X_val_ids: List[List[int]],
#     y_val: np.ndarray,
#     epochs: int = 10,
#     batch_size: int = 64,
#     maxlen: int = 256,
#     lr: float = 1e-3,
#     save_path: str = "transformer_encoder_np.npz",
# ):
#     opt = RMSprop(model.params, lr=lr)
#     best_val = 0.0
#     for ep in range(1, epochs + 1):
#         # Train
#         model.zero_grad()
#         losses = []
#         accs = []
#         for Xb, yb in iterate_minibatches(X_train_ids, y_train, batch_size, maxlen, shuffle=True):
#             model.zero_grad()
#             probs = model.forward(Xb, training=True)
#             loss, acc = model.loss_and_acc(yb)
#             losses.append(loss)
#             accs.append(acc)
#             model.backward(yb)
#             opt.step()
#         train_loss = float(np.mean(losses))
#         train_acc = float(np.mean(accs))
#         # Validate
#         val_losses = []
#         val_accs = []
#         for Xb, yb in iterate_minibatches(X_val_ids, y_val, batch_size, maxlen, shuffle=False):
#             _ = model.forward(Xb, training=False)
#             loss, acc = model.loss_and_acc(yb)
#             val_losses.append(loss)
#             val_accs.append(acc)
#         val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
#         val_acc = float(np.mean(val_accs)) if val_accs else 0.0
#         print(f"Epoch {ep:02d}: train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f}")
#         if val_acc > best_val:
#             best_val = val_acc
#             save_weights(model, save_path)
#             print(f"  \u2191 Saved best to {save_path} (val_acc={best_val:.3f})")


# def evaluate_model(model: NumPyTransformerClassifier, X_ids: List[List[int]], y: np.ndarray, batch_size: int = 64, maxlen: int = 256) -> Tuple[float, float]:
#     losses, accs = [], []
#     for Xb, yb in iterate_minibatches(X_ids, y, batch_size, maxlen, shuffle=False):
#         _ = model.forward(Xb, training=False)
#         loss, acc = model.loss_and_acc(yb)
#         losses.append(loss)
#         accs.append(acc)
#     return float(np.mean(losses)), float(np.mean(accs))


# # ----------------------------
# # Save / Load
# # ----------------------------

# def save_weights(model: NumPyTransformerClassifier, path: str):
#     data = {}
#     for p in model.params:
#         data[p.name] = p.value
#     np.savez(path, **data)


# def load_weights(model: NumPyTransformerClassifier, path: str):
#     zz = np.load(path)
#     name_to_param = {p.name: p for p in model.params}
#     for k in zz.files:
#         if k in name_to_param:
#             name_to_param[k].value[...] = zz[k]
#         else:
#             print(f"[warn] extra weight in file: {k}")


# # ----------------------------
# # Example main (end-to-end)
# # ----------------------------

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--imdb_path", type=str, required=False, default="./aclImdb")
#     parser.add_argument("--vocab_size", type=int, default=20000)
#     parser.add_argument("--embed_dim", type=int, default=256)
#     parser.add_argument("--num_heads", type=int, default=2)
#     parser.add_argument("--dense_dim", type=int, default=32)
#     parser.add_argument("--dropout", type=float, default=0.5)
#     parser.add_argument("--maxlen", type=int, default=256)
#     parser.add_argument("--epochs", type=int, default=5)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--weights", type=str, default="transformer_encoder_np.npz")
#     args = parser.parse_args()

#     print("Loading IMDB...")
#     # texts, labels = load_imdb(args.imdb_path)
#     texts, labels = load_imdb_with_cache()
#     labels = np.array(labels, dtype=np.int32)

#     # Simple split: 40k train, 5k val, 5k test (IMDB has 50k total)
#     idx = np.arange(len(texts))
#     np.random.shuffle(idx)
#     texts = [texts[i] for i in idx]
#     labels = labels[idx]

#     stoi = build_vocab(texts, args.vocab_size)
#     ids_all = [encode(t, stoi) for t in texts]

#     n = len(ids_all)
#     n_train = int(0.8 * n)
#     n_val = int(0.1 * n)

#     X_train, y_train = ids_all[:n_train], labels[:n_train]
#     X_val, y_val = ids_all[n_train:n_train + n_val], labels[n_train:n_train + n_val]
#     X_test, y_test = ids_all[n_train + n_val :], labels[n_train + n_val :]

#     print(f"Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

#     model = NumPyTransformerClassifier(
#         vocab_size=args.vocab_size,
#         embed_dim=args.embed_dim,
#         num_heads=args.num_heads,
#         dense_dim=args.dense_dim,
#         dropout_p=args.dropout,
#     )

#     train_model(
#         model,
#         X_train,
#         y_train,
#         X_val,
#         y_val,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         maxlen=args.maxlen,
#         lr=args.lr,
#         save_path=args.weights,
#     )

#     print("Loading best weights and evaluating on test set...")
#     load_weights(model, args.weights)
#     test_loss, test_acc = evaluate_model(model, X_test, y_test, batch_size=args.batch_size, maxlen=args.maxlen)
#     print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.3f}")
import os
import re
import math
import json
import random
from collections import Counter
from typing import List, Tuple, Dict, Any
from load_data import load_imdb_with_cache

import numpy as np

# ----------------------------
# Data utilities (tokenize IMDB)
# ----------------------------
# This section handles all the preliminary steps to prepare raw text data
# for a machine learning model, which requires numerical input.

def clean_text(t: str) -> str:
    """
    Performs basic text cleaning.
    - Converts text to lowercase.
    - Replaces HTML tags (like '<br />') with a space.
    - Removes punctuation and special characters, keeping only letters, numbers, and apostrophes.
    - Collapses multiple spaces into a single space and removes leading/trailing spaces.
    """
    t = t.lower()
    t = re.sub(r"<.*?>", " ", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_vocab(texts: List[str], vocab_size: int) -> Dict[str, int]:
    """
    Builds a vocabulary mapping words to unique integer IDs.
    - Uses a Counter to find the frequency of each word.
    - Selects the `vocab_size - 2` most common words.
    - Reserves ID 0 for padding (<PAD>) and ID 1 for out-of-vocabulary (<OOV>) words.
    """
    counter = Counter()
    for t in texts:
        # Splits the cleaned text into individual words (tokens)
        counter.update(clean_text(t).split())
    
    # Reserve 0 for PAD, 1 for OOV
    most_common = [(w, c) for w, c in counter.most_common(vocab_size - 2)]
    # Assigns integer IDs starting from 2
    stoi = {w: i + 2 for i, (w, _) in enumerate(most_common)}
    stoi["<PAD>"] = 0
    stoi["<OOV>"] = 1
    return stoi


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    """
    Converts a text string into a list of integer IDs using the vocabulary map.
    - For each word, it looks up its ID in the vocabulary.
    - If a word is not found, it's assigned the `<OOV>` token ID (1).
    """
    return [stoi.get(w, 1) for w in clean_text(text).split()]


def pad_batch(batch: List[List[int]], maxlen: int) -> np.ndarray:
    """
    Pads or truncates sequences in a batch to a uniform length.
    - This is necessary because neural networks require inputs of a fixed size.
    - Sequences shorter than `maxlen` are padded with zeros (`<PAD>` token ID).
    - Sequences longer than `maxlen` are truncated.
    - The output is a NumPy array, which is an efficient data structure for numerical computation.
    """
    x = np.zeros((len(batch), maxlen), dtype=np.int64)
    for i, seq in enumerate(batch):
        seq = seq[:maxlen]
        x[i, : len(seq)] = np.array(seq, dtype=np.int64)
    return x


def load_imdb(path: str) -> Tuple[List[str], List[int]]:
    """
    Loads text and labels from the IMDB dataset directory structure.
    - The dataset is expected to be in a specific folder hierarchy: `path/split/label/*.txt`.
    - It iterates through 'train' and 'test' folders, and 'pos' and 'neg' subfolders.
    - Reads each text file and assigns a label (1 for 'pos', 0 for 'neg').
    """
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
# This section implements the core components of the neural network from scratch using NumPy.
# Each class has a `forward` method for computation and a `backward` method for backpropagation,
# manually defining how gradients are calculated.

class Parameter:
    """
    A simple container class for learnable parameters (weights and biases).
    - Stores the parameter's `value` (a NumPy array).
    - Stores its `grad` (gradient), which is computed during the backward pass.
    - `zero_grad()` is a crucial method to reset gradients before each new batch.
    """
    def __init__(self, value: np.ndarray, name: str = ""):
        self.value = value
        self.grad = np.zeros_like(value)
        self.name = name

    def zero_grad(self):
        self.grad[...] = 0.0


class Embedding:
    """
    A layer that maps integer word IDs to dense vectors.
    """
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        # `W` is the embedding matrix, initialized with a normal distribution.
        self.W = Parameter(np.random.default_rng(seed).normal(0, 0.02, size=(vocab_size, embed_dim)).astype(np.float32), "emb/W")
        self.cache = None

    def forward(self, x_ids: np.ndarray) -> np.ndarray:
        # Looks up embedding vectors for each word ID in the input tensor.
        # x_ids: (B, T) -> out: (B, T, D)
        out = self.W.value[x_ids]
        # Caches the input IDs to be used during the backward pass.
        self.cache = x_ids
        return out

    def backward(self, dout: np.ndarray):
        # Accumulates gradients for each word ID.
        # `np.add.at` efficiently sums gradients from the same word ID
        # that appears multiple times in the batch.
        x_ids = self.cache
        np.add.at(self.W.grad, x_ids, dout)

    @property
    def params(self):
        # Returns a list of learnable parameters in this layer.
        return [self.W]


class LayerNorm:
    """
    Performs Layer Normalization, which stabilizes training by normalizing activations.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        # `gamma` and `beta` are learnable parameters for scaling and shifting.
        self.gamma = Parameter(np.ones((dim,), dtype=np.float32), "ln/gamma")
        self.beta = Parameter(np.zeros((dim,), dtype=np.float32), "ln/beta")
        self.eps = eps
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Computes mean and variance over the last dimension (the feature dimension).
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        # Normalizes the input. `eps` is a small value to prevent division by zero.
        xhat = (x - mean) / np.sqrt(var + self.eps)
        # Applies the learned scale and shift.
        out = self.gamma.value * xhat + self.beta.value
        # Caches intermediate values needed for the backward pass.
        self.cache = (xhat, var, mean, x)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        xhat, var, mean, x = self.cache
        N = x.shape[-1]
        # Calculates gradients for `gamma` and `beta` by summing along all but the last axis.
        dgamma = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
        dbeta = dout.sum(axis=tuple(range(dout.ndim - 1)))
        self.gamma.grad += dgamma
        self.beta.grad += dbeta
        # Computes the gradient with respect to the input `x` using the chain rule.
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
    """
    A regularization layer that randomly sets a fraction of inputs to zero during training.
    - Prevents co-adaptation of neurons and helps the model generalize better.
    """
    def __init__(self, p: float, seed: int = 123):
        self.p = p
        self.seed = seed
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        # If not training or dropout rate is zero, just pass the input through.
        if not self.training or self.p <= 0.0:
            self.mask = None
            return x
        # Generates a mask where each element has a `(1-p)` probability of being 1.
        rng = np.random.default_rng(self.seed)
        self.mask = (rng.uniform(size=x.shape) >= self.p).astype(np.float32)
        # Applies the mask and scales the output by `1/(1-p)`. This is "inverted dropout,"
        # which ensures the expected sum of activations remains the same, simplifying inference.
        return x * self.mask / (1.0 - self.p)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # During the backward pass, the same mask and scaling are applied to the gradient.
        if self.mask is None:
            return dout
        return dout * self.mask / (1.0 - self.p)

    @property
    def params(self):
        # Dropout has no learnable parameters.
        return []


class Dense:
    """
    A fully-connected linear layer.
    - Computes `y = x @ W + b`.
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, seed: int = 0, name_prefix: str = "dense"):
        rng = np.random.default_rng(seed)
        # Initializes weights using Xavier/Glorot uniform initialization, a common practice
        # to help with stable training.
        limit = math.sqrt(6 / (in_dim + out_dim))
        self.W = Parameter(rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32), f"{name_prefix}/W")
        self.b = Parameter(np.zeros((out_dim,), dtype=np.float32), f"{name_prefix}/b") if bias else None
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Performs matrix multiplication `x @ W`.
        out = x @ self.W.value
        # Adds the bias vector if it exists.
        if self.b is not None:
            out = out + self.b.value
        # Caches the input `x` for the backward pass.
        self.cache = x
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache
        # Computes the gradient for the weights (`W.grad`).
        # Reshaping is necessary to handle batches and multi-dimensional inputs.
        self.W.grad += x.reshape(-1, x.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
        # Computes the gradient for the bias (`b.grad`) by summing the incoming gradient.
        if self.b is not None:
            self.b.grad += dout.sum(axis=tuple(range(dout.ndim - 1)))
        # Computes the gradient with respect to the input `x`.
        dx = dout @ self.W.value.T
        return dx

    @property
    def params(self):
        return [p for p in [self.W, self.b] if p is not None]


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit) activation function. Returns `x` if `x > 0`, otherwise `0`.
    """
    return np.maximum(x, 0.0)


def drelu(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the ReLU function. Returns 1 if `x > 0`, otherwise 0.
    """
    return (x > 0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function. Squashes values between 0 and 1,
    making them suitable for binary classification probabilities.
    """
    return 1.0 / (1.0 + np.exp(-x))


class FeedForward:
    """
    A two-layer, fully-connected feed-forward network,
    a standard component of a Transformer block.
    """
    def __init__(self, dim: int, hidden: int, seed: int = 0):
        # Composed of two dense layers with a ReLU in between.
        self.lin1 = Dense(dim, hidden, name_prefix="ff/lin1", seed=seed)
        self.lin2 = Dense(hidden, dim, name_prefix="ff/lin2", seed=seed + 1)
        self.cache_a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = self.lin1.forward(x)
        h = relu(a)
        # Caches the pre-activation output `a` to compute the derivative of ReLU in the backward pass.
        self.cache_a = a
        out = self.lin2.forward(h)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Backpropagates through `lin2`.
        dh = self.lin2.backward(dout)
        # Applies the derivative of ReLU to the gradient `dh`.
        da = dh * drelu(self.cache_a)
        # Backpropagates through `lin1`.
        dx = self.lin1.backward(da)
        return dx

    @property
    def params(self):
        return self.lin1.params + self.lin2.params


class MultiHeadSelfAttention:
    """
    The core attention mechanism of the Transformer.
    - Allows the model to weigh the importance of different words in a sequence.
    - Uses multiple "heads" to attend to different parts of the input in parallel.
    
    """
    def __init__(self, dim: int, num_heads: int, seed: int = 0):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # Defines the dense layers for Query, Key, and Value projections.
        self.Wq = Dense(dim, dim, name_prefix="attn/Wq", seed=seed)
        self.Wk = Dense(dim, dim, name_prefix="attn/Wk", seed=seed + 1)
        self.Wv = Dense(dim, dim, name_prefix="attn/Wv", seed=seed + 2)
        # The final output projection layer.
        self.Wo = Dense(dim, dim, name_prefix="attn/Wo", seed=seed + 3)
        self.cache = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        # Reshapes the input tensor to separate the heads.
        # (B, T, D) -> (B, H, T, Hd)
        B, T, D = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        # Merges the outputs from the multiple heads back into a single tensor.
        # (B, H, T, Hd) -> (B, T, D)
        B, H, T, Hd = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(B, T, H * Hd)
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Project input to Q, K, V and split heads.
        Q = self._split_heads(self.Wq.forward(x))
        K = self._split_heads(self.Wk.forward(x))
        V = self._split_heads(self.Wv.forward(x))
        # Scaled dot-product attention formula: `softmax(Q @ K.T / sqrt(d_k)) @ V`.
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (Q @ K.transpose(0, 1, 3, 2)) * scale
        # Softmax on scores to get attention weights.
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        out_heads = attn @ V
        # Merge heads and pass through the final linear layer.
        out = self._merge_heads(out_heads)
        out = self.Wo.forward(out)
        self.cache = (x, Q, K, V, attn, out_heads)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, Q, K, V, attn, out_heads = self.cache
        # The backward pass is a complex reverse of the forward pass, applying the chain rule
        # to each operation, from the final output back to the input.
        dmerge = self.Wo.backward(dout)
        B, T, D = dmerge.shape
        dheads = dmerge.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        dattn = dheads @ V.transpose(0, 1, 3, 2)
        dV = attn.transpose(0, 1, 3, 2) @ dheads
        # Manually computes the backward pass for the softmax function.
        dscores = np.empty_like(attn)
        for b in range(B):
            for h in range(self.num_heads):
                A = attn[b, h]
                dA = dattn[b, h]
                s = (dA * A).sum(axis=-1, keepdims=True)
                dscores[b, h] = A * (dA - s)
        scale = 1.0 / math.sqrt(self.head_dim)
        dscores *= scale
        dQ = dscores @ K
        dK = dscores.transpose(0, 1, 3, 2) @ Q
        # Merge head gradients and backpropagate through the initial Dense layers.
        def merge(xh):
            return xh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dQm = merge(dQ)
        dKm = merge(dK)
        dVm = merge(dV)
        dx_q = self.Wq.backward(dQm)
        dx_k = self.Wk.backward(dKm)
        dx_v = self.Wv.backward(dVm)
        # Sum the gradients from Q, K, and V paths to get the total input gradient.
        dx = dx_q + dx_k + dx_v
        return dx

    @property
    def params(self):
        return self.Wq.params + self.Wk.params + self.Wv.params + self.Wo.params


class GlobalMaxPool1D:
    """
    A pooling layer that extracts the maximum value for each feature across the sequence dimension.
    - Used to get a fixed-size representation of the entire sequence.
    """
    def __init__(self):
        self.cache_idx = None
        self.cache_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Finds the indices of the maximum values along the sequence dimension (axis=1).
        # x: (B, T, D) -> out: (B, D)
        self.cache_shape = x.shape
        idx = x.argmax(axis=1)
        self.cache_idx = idx
        B, T, D = x.shape
        # Uses NumPy's "fancy indexing" to select the maximum values.
        out = x[np.arange(B)[:, None], idx, np.arange(D)[None, :]]
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # The gradient is passed back only to the elements that were the maximums.
        B, T, D = self.cache_shape
        dx = np.zeros((B, T, D), dtype=np.float32)
        idx = self.cache_idx
        # Scatters the incoming gradient `dout` back to the original `argmax` positions.
        dx[np.arange(B)[:, None], idx, np.arange(D)[None, :]] = dout
        return dx

    @property
    def params(self):
        return []


class TransformerEncoderBlock:
    """
    A single block of the Transformer encoder, consisting of a self-attention layer and a feed-forward network.
    - Follows the "pre-norm" architecture (LayerNorm before the sub-layers).
    - Includes residual connections (`x + h`) and dropout for regularization.
    """
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
        # First sub-layer: Attention
        h = self.ln1.forward(x)
        h = self.attn.forward(h)
        h = self.drop1.forward(h)
        # Residual connection
        x = x + h
        # Second sub-layer: Feed-forward
        h2 = self.ln2.forward(x)
        h2 = self.ff.forward(h2)
        h2 = self.drop2.forward(h2)
        # Residual connection
        out = x + h2
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # The backward pass follows the forward pass in reverse, handling residual gradients.
        # Gradient flows back through the second residual connection.
        dh2 = dout
        dh2 = self.drop2.backward(dh2)
        dh2 = self.ff.backward(dh2)
        dx2 = self.ln2.backward(dh2)
        # Gradient for the input of the second residual is the sum of `dout` and `dx2`.
        dx = dout + dx2

        # Gradient flows back through the first residual connection.
        dh1 = dx
        dh1 = self.drop1.backward(dh1)
        dh1 = self.attn.backward(dh1)
        dx1 = self.ln1.backward(dh1)
        # The final gradient is the sum of the gradient from the first residual path (`dx1`)
        # and the gradient that "skipped" the first residual connection (`dout`). This seems
        # to be a slight error in the residual gradient calculation. The correct form for
        # `x = x + h` would be `dx = dout`, and `dh = dout`. So, `dx_total = dx1 + dout`.
        # This implementation calculates `dx` and then re-adds `dout`, which is redundant.
        dx_total = dx1 + dout
        return dx_total

    @property
    def params(self):
        return self.ln1.params + self.attn.params + self.ln2.params + self.ff.params


# ----------------------------
# Model
# ----------------------------
# This class assembles all the layers into a complete model for text classification.

class NumPyTransformerClassifier:
    """
    A full Transformer-based binary classifier for text data.
    """
    def __init__(self, vocab_size=20000, embed_dim=256, num_heads=2, dense_dim=32, dropout_p=0.5, seed: int = 0):
        self.embed = Embedding(vocab_size, embed_dim, seed=seed)
        self.encoder = TransformerEncoderBlock(embed_dim, dense_dim, num_heads, dropout_p=dropout_p, seed=seed)
        self.pool = GlobalMaxPool1D()
        self.drop = Dropout(dropout_p, seed=seed + 100)
        self.out = Dense(embed_dim, 1, name_prefix="out", seed=seed + 101)
        self.training = True

    @property
    def params(self):
        # A property to easily access all learnable parameters in the model.
        ps = []
        for layer in [self.embed, self.encoder, self.out]:
            ps += layer.params
        return ps

    def zero_grad(self):
        # Resets the gradients of all parameters.
        for p in self.params:
            p.zero_grad()

    def forward(self, x_ids: np.ndarray, training: bool = True) -> np.ndarray:
        # Defines the forward pass of the entire network.
        # Input IDs are converted to embeddings, passed through the encoder,
        # pooled to a fixed size, and then passed through a final dense layer and sigmoid.
        self.training = training
        h = self.embed.forward(x_ids)
        h = self.encoder.forward(h, training=training)
        h = self.pool.forward(h)
        self.drop.training = training
        h = self.drop.forward(h)
        logits = self.out.forward(h)  # (B,1)
        probs = sigmoid(logits)
        # Caches the outputs needed for loss calculation and backpropagation.
        self._cache_logits = logits
        self._cache_probs = probs
        self._cache_labels = None
        return probs

    def backward(self, y_true: np.ndarray):
        # Defines the backward pass.
        # Starts by computing the gradient of the loss with respect to the output logits.
        # The derivative of Binary Cross-Entropy (BCE) with respect to logits is `(probs - y_true)`.
        probs = self._cache_probs.reshape(-1, 1)
        y = y_true.reshape(-1, 1).astype(np.float32)
        # Average the gradient over the batch size.
        dlogits = (probs - y) / y.shape[0]
        # Backpropagates the gradient through the layers in reverse order.
        dh = self.out.backward(dlogits)
        dh = self.drop.backward(dh)
        dh = self.pool.backward(dh)
        dh = self.encoder.backward(dh)
        self.embed.backward(dh)

    def loss_and_acc(self, y_true: np.ndarray) -> Tuple[float, float]:
        # Calculates the Binary Cross-Entropy loss and accuracy.
        probs = self._cache_probs.reshape(-1)
        y = y_true.astype(np.float32)
        eps = 1e-7
        # The BCE formula is `-mean(y * log(p) + (1-y) * log(1-p))`. `eps` is added for numerical stability.
        bce = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        # Accuracy is calculated by thresholding probabilities at 0.5.
        preds = (probs >= 0.5).astype(np.int32)
        acc = (preds == y_true).mean()
        return float(bce), float(acc)


# ----------------------------
# Optimizer: RMSprop
# ----------------------------
# This class updates the model's parameters based on their gradients.

class RMSprop:
    """
    An implementation of the RMSprop optimizer.
    - Uses a moving average of squared gradients to scale the learning rate.
    - This helps to adapt the learning rate for each parameter, which can lead to faster convergence.
    """
    def __init__(self, params: List[Parameter], lr=1e-3, rho=0.9, eps=1e-8):
        self.params = params
        self.lr = lr
        self.rho = rho
        self.eps = eps
        # `cache` stores the moving average of squared gradients for each parameter.
        self.cache = {id(p): np.zeros_like(p.value) for p in params}

    def step(self):
        # Performs a single optimization step.
        for p in self.params:
            s = self.cache[id(p)]
            # Updates the moving average `s`.
            s[...] = self.rho * s + (1 - self.rho) * (p.grad ** 2)
            # Updates the parameter `p.value` using the gradients and the cached moving average.
            p.value[...] = p.value - self.lr * p.grad / (np.sqrt(s) + self.eps)


# ----------------------------
# Training utilities
# ----------------------------
# Functions to manage the training and evaluation loops.

def iterate_minibatches(X_ids: List[List[int]], y: np.ndarray, batch_size: int, maxlen: int, shuffle: bool = True):
    """
    A generator that yields batches of padded data for training or evaluation.
    - Shuffles the indices to randomize the batch order if `shuffle=True`.
    """
    idxs = np.arange(len(X_ids))
    if shuffle:
        np.random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch_idx = idxs[start:start + batch_size]
        batch = [X_ids[i] for i in batch_idx]
        # Pads the sequences in the current batch.
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
    """
    The main training loop.
    - Initializes the optimizer.
    - Iterates for a specified number of epochs.
    - Within each epoch, it iterates through training batches.
        - Calls `model.zero_grad()` to reset gradients.
        - Calls `model.forward()` to compute predictions.
        - Calls `model.loss_and_acc()` to get loss and accuracy.
        - Calls `model.backward()` to compute gradients.
        - Calls `opt.step()` to update parameters.
    - After each epoch, it evaluates the model on the validation set.
    - Saves the model's weights if the validation accuracy improves, which is a common
      strategy to prevent saving a model that has started overfitting.
    """
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
    """
    Evaluates the model on a given dataset (e.g., the test set).
    - Runs the forward pass in non-training mode (`training=False`).
    - Calculates and returns the average loss and accuracy.
    """
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
# These functions handle model persistence.

def save_weights(model: NumPyTransformerClassifier, path: str):
    """
    Saves the model's learnable parameters to a compressed NumPy file (`.npz`).
    """
    data = {}
    for p in model.params:
        data[p.name] = p.value
    np.savez(path, **data)


def load_weights(model: NumPyTransformerClassifier, path: str):
    """
    Loads model weights from a `.npz` file, matching them by name.
    - Useful for resuming training or evaluating a pre-trained model.
    """
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
# The main execution block that orchestrates the entire process.

if __name__ == "__main__":
    import argparse

    # Sets up command-line arguments to configure the model and training.
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
    # Loads the full dataset (50,000 reviews). `load_imdb_with_cache` is a helper
    # that prevents re-loading and re-processing the data on every run.
    texts, labels = load_imdb_with_cache()
    labels = np.array(labels, dtype=np.int32)

    # Shuffles the data and performs a simple 80/10/10 train/validation/test split.
    idx = np.arange(len(texts))
    np.random.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = labels[idx]

    # Builds the vocabulary and encodes all texts into integer IDs.
    stoi = build_vocab(texts, args.vocab_size)
    ids_all = [encode(t, stoi) for t in texts]

    n = len(ids_all)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    X_train, y_train = ids_all[:n_train], labels[:n_train]
    X_val, y_val = ids_all[n_train:n_train + n_val], labels[n_train:n_train + n_val]
    X_test, y_test = ids_all[n_train + n_val :], labels[n_train + n_val :]

    print(f"Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

    # Initializes the model with specified hyperparameters.
    model = NumPyTransformerClassifier(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dense_dim=args.dense_dim,
        dropout_p=args.dropout,
    )

    # Starts the training process.
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
    # Loads the best-performing model from the training phase.
    load_weights(model, args.weights)
    # Evaluates the final model on the unseen test set.
    test_loss, test_acc = evaluate_model(model, X_test, y_test, batch_size=args.batch_size, maxlen=args.maxlen)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.3f}")
