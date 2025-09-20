# dataset: http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
import os
import re
import math
import json
import random
from collections import Counter
from typing import List, Tuple, Dict, Any
import time

import numpy as np

# =============================================================
# Data utilities (spa-eng)
# =============================================================

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<OOV>": 1,
    "[start]": 2,
    "[end]": 3,
}

def clean_text_en(t: str) -> str:
    t = t.lower()
    t = re.sub(r"<.*?>", " ", t)
    # allow [] so [start]/[end] are preserved
    t = re.sub(r"[^a-z0-9'\[\] ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_text_es(t: str) -> str:
    t = t.lower()
    t = re.sub(r"<.*?>", " ", t)
    # allow [] so [start]/[end] are preserved
    t = re.sub(r"[^a-z0-9áéíóúüñ¿¡'\[\] ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_vocab(texts: List[str], vocab_size: int, is_target: bool = False) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        ct = clean_text_es(t) if is_target else clean_text_en(t)
        counter.update(ct.split())

    # Remove any specials if they appear in the corpus (they do, since you injected [start]/[end])
    for tok in SPECIAL_TOKENS:
        counter.pop(tok, None)

    stoi = {tok: idx for tok, idx in SPECIAL_TOKENS.items()}

    # Fill remaining slots contiguously
    next_id = len(SPECIAL_TOKENS)
    for w, _ in counter.most_common():
        if next_id >= vocab_size:
            break
        stoi[w] = next_id
        next_id += 1

    # Sanity check: ids must be 0..len(stoi)-1 with no holes
    assert max(stoi.values()) == len(stoi) - 1, "Vocab ids not contiguous"
    return stoi

def encode(text: str, stoi: Dict[str, int], is_target: bool = False) -> List[int]:
    t = clean_text_es(text) if is_target else clean_text_en(text)
    return [stoi.get(w, SPECIAL_TOKENS["<OOV>"]) for w in t.split()]


def pad_batch(batch: List[List[int]], maxlen: int) -> np.ndarray:
    x = np.zeros((len(batch), maxlen), dtype=np.int64)
    for i, seq in enumerate(batch):
        seq = seq[:maxlen]
        x[i, : len(seq)] = np.array(seq, dtype=np.int64)
    return x


def load_spa_eng(path: str) -> List[Tuple[str, str]]:
    """path points to folder containing spa.txt (unzipped spa-eng.zip)."""
    txt = os.path.join(path, "spa.txt")
    pairs = []
    with open(txt, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            eng, spa, *_ = line.split("\t")
            pairs.append((eng, spa))
    return pairs


# =============================================================
# Core autograd-lite tensors (Parameters only)
# =============================================================

class Parameter:
    def __init__(self, value: np.ndarray, name: str = ""):
        self.value = value
        self.grad = np.zeros_like(value)
        self.name = name

    def zero_grad(self):
        self.grad[...] = 0.0


# =============================================================
# Layers
# =============================================================

class Embedding:
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 0, name: str = "emb"):
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 0.02, size=(vocab_size, embed_dim)).astype(np.float32)
        self.W = Parameter(W, f"{name}/W")
        self.cache_ids = None

    def forward(self, x_ids: np.ndarray) -> np.ndarray:
        out = self.W.value[x_ids]
        self.cache_ids = x_ids
        return out

    def backward(self, dout: np.ndarray):
        np.add.at(self.W.grad, self.cache_ids, dout)

    @property
    def params(self):
        return [self.W]


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5, name: str = "ln"):
        self.gamma = Parameter(np.ones((dim,), dtype=np.float32), f"{name}/gamma")
        self.beta = Parameter(np.zeros((dim,), dtype=np.float32), f"{name}/beta")
        self.eps = eps
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma.value * xhat + self.beta.value
        self.cache = (xhat, var, x)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        xhat, var, x = self.cache
        N = x.shape[-1]
        reduce_axes = tuple(range(dout.ndim - 1))
        self.gamma.grad += (dout * xhat).sum(axis=reduce_axes)
        self.beta.grad += dout.sum(axis=reduce_axes)
        std_inv = 1.0 / np.sqrt(var + self.eps)
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
    def __init__(self, p: float, seed: int = 0):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p <= 0:
            self.mask = None
            return x
        self.mask = (self.rng.uniform(size=x.shape) >= self.p).astype(np.float32)
        return x * self.mask / (1.0 - self.p)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return dout
        return dout * self.mask / (1.0 - self.p)

    @property
    def params(self):
        return []


class Dense:
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, seed: int = 0, name: str = "dense"):
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6 / (in_dim + out_dim))
        self.W = Parameter(rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32), f"{name}/W")
        self.b = Parameter(np.zeros((out_dim,), dtype=np.float32), f"{name}/b") if bias else None
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


class FeedForward:
    def __init__(self, dim: int, hidden: int, seed: int = 0, name: str = "ff"):
        self.lin1 = Dense(dim, hidden, name=f"{name}/lin1", seed=seed)
        self.lin2 = Dense(hidden, dim, name=f"{name}/lin2", seed=seed + 1)
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


# --------- Attention (general: can do self- or cross-attn) ---------

class MultiHeadAttention:
    def __init__(self, dim_q: int, dim_kv: int, num_heads: int, seed: int = 0, name: str = "attn"):
        assert dim_q % num_heads == 0
        assert dim_kv % num_heads == 0
        self.num_heads = num_heads # num_heads=8
        self.head_q = dim_q // num_heads # head_q = dim_kv = 256 // 8 = 32
        self.head_kv = dim_kv // num_heads # head_kv = dim_kv = 256 // 8 = 32
        self.Wq = Dense(dim_q, dim_q, name=f"{name}/Wq", seed=seed) # dim_q=256
        self.Wk = Dense(dim_kv, dim_kv, name=f"{name}/Wk", seed=seed + 1) # dim_kv=256
        self.Wv = Dense(dim_kv, dim_kv, name=f"{name}/Wv", seed=seed + 2) # dim_kv=256
        self.Wo = Dense(dim_q, dim_q, name=f"{name}/Wo", seed=seed + 3) # dim_q=256
        self.cache = None

    def _split(self, x: np.ndarray, dim_per_head: int) -> np.ndarray:
        B, T, D = x.shape # x.shape=(64,20,256)
        x = x.reshape(B, T, self.num_heads, dim_per_head) # self.num_heads = 8, dim_per_head = 32
        return x.transpose(0, 2, 1, 3)  # (B,H,T,Hd)

    def _merge(self, x: np.ndarray) -> np.ndarray:
        B, H, T, Hd = x.shape # x.shape=(64,8,20,32)
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Hd)

    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                mask: np.ndarray = None) -> np.ndarray:
        # q,k,v: (B,T*,D)
        Q = self._split(self.Wq.forward(q), self.head_q) #q.shape=(64,20,256), self.head_q=32 -> Q.shape=(64,8,20,32)
        K = self._split(self.Wk.forward(k), self.head_kv)
        V = self._split(self.Wv.forward(v), self.head_kv)
        scale = 1.0 / math.sqrt(self.head_kv) # scale = 1/sqrt(32)=0.1767766952966369
        scores = (Q @ K.transpose(0, 1, 3, 2)) * scale  # (B,H,Tq,Tk) Q.shape=(64,8,20,32), K.shape=(64,8,20,32) -> scores.shape=(64,8,20,20)
        if mask is not None:
            # mask: True means masked. Broadcast to (B,1,Tq,Tk)
            scores = np.where(mask, -1e9, scores)
        # softmax last axis
        scores_max = scores.max(axis=-1, keepdims=True) # score.shape=(64,8,20,20) -> scores_max.shape=(64,8,20,1)
        exp_scores = np.exp(scores - scores_max) #exp_scores.shape=(64,8,20,20)
        denom = exp_scores.sum(axis=-1, keepdims=True) + 1e-12 #denom.shape=(64,8,20,1)
        A = exp_scores / denom  # (B,H,Tq,Tk) # A.shape=(64,8,20,20)
        out_h = A @ V  # (B,H,Tq,Hd) # A.shape=(64,8,20,20), V.shape=(64,8,20,32) -> out_h.shape=(64,8,20,32)
        out = self._merge(out_h)  # (B,Tq,D)
        out = self.Wo.forward(out) # out.shape=(64,20,256)
        self.cache = (q, k, v, Q, K, V, A, out_h, mask)
        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q, k, v, Q, K, V, A, out_h, mask = self.cache
        dmerge = self.Wo.backward(dout)  # dout.shape=(64,20,256) (B,Tq,D)
        B, Tq, D = dmerge.shape # B=64, Tq=20, D=256
        H = self.num_heads # H=8
        Hd = D // H # Hd=32
        dh = dmerge.reshape(B, Tq, H, Hd).transpose(0, 2, 1, 3)  # (B,H,Tq,Hd) # dh.shape=(64,8,20,32)
        # A @ V = out_h
        dA = dh @ V.transpose(0, 1, 3, 2)   # (B,H,Tq,Tk) # dA.shape=(64,8,20,20)
        dV = A.transpose(0, 1, 3, 2) @ dh   # (B,H,Tk,Hd) # dV.shape=(64,8,20,32)
        # softmax backward per row
        dS = np.empty_like(A) # dS.shape=(64,8,20,20)
        for b in range(B): # B=64
            for h in range(H):
                for t in range(Tq):
                    a = A[b, h, t]               # (Tk,)
                    da = dA[b, h, t]
                    s = (da * a).sum()
                    dS[b, h, t] = a * (da - s)
        # scale
        dS *= 1.0 / math.sqrt(Hd) # dS.shape=(64,8,20,20)
        # S = Q @ K^T
        dQ = dS @ K # dS.shape = (64,8,20,20) , K.shape = (64,8,20,32) -> dQ.shape = (64,8,20,32)
        dK = dS.transpose(0, 1, 3, 2) @ Q # dS.shape = (64,8,20,20) , Q.shape = (64,8,20,32) -> dK.shape = (64,8,20,32)
        # merge heads back
        def merge(xh):
            return xh.transpose(0, 2, 1, 3).reshape(B, Tq if xh.shape[2] == Tq else xh.shape[2], H * Hd)
        dQm = merge(dQ)  # (B,Tq,D) dQ.shape=(64,8,20,32) -> dQm.shape=(64,20,256)
        dKm = merge(dK)  # (B,Tk,D)
        dVm = merge(dV)  # (B,Tk,D)
        dq = self.Wq.backward(dQm)
        dk = self.Wk.backward(dKm)
        dv = self.Wv.backward(dVm)
        return dq, dk, dv # dq.shape=(64,20,256) dk.shape=(64,20,256) dv.shape=(64,20,256)

    @property
    def params(self):
        return self.Wq.params + self.Wk.params + self.Wv.params + self.Wo.params


# --------- Encoder / Decoder blocks ---------

class TransformerEncoderBlock:
    def __init__(self, dim: int, hidden: int, num_heads: int, dropout_p: float = 0.1, seed: int = 0, name: str = "encblk"):
        self.ln1 = LayerNorm(dim, name=f"{name}/ln1") # dim=256
        self.attn = MultiHeadAttention(dim, dim, num_heads, seed=seed, name=f"{name}/attn") # dim=256 num_heads=8
        self.drop1 = Dropout(dropout_p, seed=seed + 11)
        self.ln2 = LayerNorm(dim, name=f"{name}/ln2")
        self.ff = FeedForward(dim, hidden, seed=seed + 20, name=f"{name}/ff")
        self.drop2 = Dropout(dropout_p, seed=seed + 21)

    def forward(self, x: np.ndarray, training: bool, src_pad: np.ndarray) -> np.ndarray:
        self.drop1.training = training
        self.drop2.training = training
        h = self.ln1.forward(x) # x.shape=(64,20,256) -> h.shape=(64,20,256)
        # mask keys only: broadcast to (B,1,Tq,Tk) with Tq=Tk
        B, T, _ = x.shape
        key_mask = src_pad[:, None, None, :]  # (B,1,1,T) src_pad.shape=(64,20)
        h = self.attn.forward(h, h, h, mask=key_mask) # h.shape=(64,20,256)
        h = self.drop1.forward(h) # h.shape=(64,20,256)
        x = x + h # x.shape=(64,20,256)
        h2 = self.ln2.forward(x) # x.shape=(64,20,256) -> h2.shape=(64,20,256)
        h2 = self.ff.forward(h2)
        h2 = self.drop2.forward(h2)
        out = x + h2 # out.shape=(64,20,256)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dh2 = self.drop2.backward(dout) # dout.shape=(64,20,256) dh2.shape=(64,20,256)
        dh2 = self.ff.backward(dh2)
        dx2 = self.ln2.backward(dh2)
        dx = dout + dx2
        dh1 = self.drop1.backward(dx)
        dq, dk, dv = self.attn.backward(dh1)
        #dx1 = self.ln1.backward(dq)  # attn is pre-norm; gradient path via q only
        dx1 = self.ln1.backward(dq + dk + dv)  # FIX: sum grads from Q,K,V
        # residual: add gradient that flowed around attn (dx)
        return dx1 + dx # dx1.shape=(64,20,256) dx.shape=(64,20,256)

    @property
    def params(self):
        return self.ln1.params + self.attn.params + self.ln2.params + self.ff.params


class TransformerDecoderBlock:
    def __init__(self, dim: int, hidden: int, num_heads: int, dropout_p: float = 0.1, seed: int = 0, name: str = "decblk"):
        self.ln1 = LayerNorm(dim, name=f"{name}/ln1") # dim=256 hidden=1024 num_heads=8
        self.self_attn = MultiHeadAttention(dim, dim, num_heads, seed=seed, name=f"{name}/self") 
        self.drop1 = Dropout(dropout_p, seed=seed + 11)
        self.ln2 = LayerNorm(dim, name=f"{name}/ln2")
        self.cross_attn = MultiHeadAttention(dim, dim, num_heads, seed=seed + 1, name=f"{name}/cross")
        self.drop2 = Dropout(dropout_p, seed=seed + 21)
        self.ln3 = LayerNorm(dim, name=f"{name}/ln3")
        self.ff = FeedForward(dim, hidden, seed=seed + 31, name=f"{name}/ff")
        self.drop3 = Dropout(dropout_p, seed=seed + 41)
        self._cache_shapes = None

    @staticmethod
    def build_causal_mask(B: int, T: int) -> np.ndarray:
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)  # True above diagonal -> masked, T=20, causal.shape=(20,20)
        return np.broadcast_to(causal, (B, 1, T, T))

    def forward(self, y: np.ndarray, enc_out: np.ndarray, training: bool, dec_pad: np.ndarray, enc_pad: np.ndarray) -> np.ndarray:
        self.drop1.training = training
        self.drop2.training = training
        self.drop3.training = training
        B, T, _ = y.shape
        # Self-attention with causal + key padding
        h = self.ln1.forward(y)
        self_mask = self.build_causal_mask(B, T) | dec_pad[:, None, None, :]  # (B,1,T,T) self.mask.shape=(64,1,20,20), dec_pad.shape=(64,20)
        h = self.self_attn.forward(h, h, h, mask=self_mask) # h.shape=(64,20,256), self_mask.shape=(64,1,20,20)
        h = self.drop1.forward(h)
        y = y + h # y.shape=(64,20,256)
        # Cross-attention (mask keys using encoder padding)
        h2 = self.ln2.forward(y) # y.shape=(64,20,256) -> h2.shape=(64,20,256)
        cross_mask = enc_pad[:, None, None, :]  # enc_pad.shape=(64,20), cross_mask.shape=(64,1,1,20)
        h2 = self.cross_attn.forward(h2, enc_out, enc_out, mask=cross_mask) # h2.shape=(64,20,256), enc_out.shape=(64,20,256)
        h2 = self.drop2.forward(h2) # h2.shape=(64,20,256)
        y = y + h2 # y.shape=(64,20,256)
        h3 = self.ln3.forward(y) # y.shape=(64,20,256) -> h3.shape=(64,20,256)
        h3 = self.ff.forward(h3)
        h3 = self.drop3.forward(h3)
        out = y + h3
        # cache for backprop tie-ups
        self._cache_shapes = (self_mask, cross_mask)
        return out # out.shape=(64,20,256)

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # returns gradients wrt decoder input and encoder output (to route upstream)
        dh3 = self.drop3.backward(dout) # dout.shape=(64,20,256), dh3.shape=(64,20,256)
        dh3 = self.ff.backward(dh3)
        dy3 = self.ln3.backward(dh3)
        dy = dout + dy3 # dout.shape=(64,20,256) dy3.shape=(64,20,256), dy.shape=(64,20,256)
        dh2 = self.drop2.backward(dy)
        dq_cross, dk_cross, dv_cross = self.cross_attn.backward(dh2) # dh2.shape=(64,20,256), dq_cross.shape=(64,20,256), dk_cross.shape=(64,20,256), dv_cross.shape=(64,20,256)
        dy2 = self.ln2.backward(dq_cross) # dq_cross.shape=(64,20,256), dy2.shape=(64,20,256)
        dy = dy + dy2
        dh1 = self.drop1.backward(dy)
        dq_self, dk_self, dv_self = self.self_attn.backward(dh1) # dh1.shape=(64,20,256), dq_self.shape=(64,20,256), dk_self.shape=(64,20,256), dv_self.shape=(64,20,256)
        dy1 = self.ln1.backward(dq_self + dk_self + dv_self)  # FIX dy1.shape=(64,20,256)
        dy_total = dy1 + dy # dy_total.shape=(64,20,256)
        # encoder receives dk_cross + dv_cross via its output (k,v are projections of enc_out)
        denc = dk_cross + dv_cross
        return dy_total, denc

    @property
    def params(self):
        return (
            self.ln1.params + self.self_attn.params +
            self.ln2.params + self.cross_attn.params +
            self.ln3.params + self.ff.params
        )

# =============================================================
# Full Seq2Seq Transformer
# =============================================================

class Seq2SeqTransformer:
    def __init__(self, src_vocab: int, tgt_vocab: int, embed_dim: int = 256, num_heads: int = 8, ff_hidden: int = 1024,
                 num_enc_layers: int = 1, num_dec_layers: int = 1, dropout_p: float = 0.1, max_len: int = 128, seed: int = 0):
        self.embed_src = Embedding(src_vocab, embed_dim, seed=seed, name="src_emb") # src_vocab = 11912, embed_dim=256
        self.embed_tgt = Embedding(tgt_vocab, embed_dim, seed=seed + 1, name="tgt_emb") # tgt_vocab = 15000, embed_dim=256
        rng = np.random.default_rng(seed + 7)
        self.pos_src = Parameter(rng.normal(0, 0.02, size=(max_len, embed_dim)).astype(np.float32), "pos/src") # max_len=20, embed_dim=256
        self.pos_tgt = Parameter(rng.normal(0, 0.02, size=(max_len, embed_dim)).astype(np.float32), "pos/tgt")
        self.enc_layers = [
            TransformerEncoderBlock(embed_dim, ff_hidden, num_heads, dropout_p=dropout_p, seed=seed + i * 100, name=f"enc{i}")
            for i in range(num_enc_layers) # embed_dim=256, ff_hidden=1024, num_heads=8, dropout_p=0.1, num_enc_layers=1
        ]
        self.dec_layers = [
            TransformerDecoderBlock(embed_dim, ff_hidden, num_heads, dropout_p=dropout_p, seed=seed + 1000 + i * 100, name=f"dec{i}")
            for i in range(num_dec_layers) # embed_dim=256, ff_hidden=1024, num_heads=8, dropout_p=0.1, num_dec_layers=1
        ]
        self.proj = Dense(embed_dim, tgt_vocab, name="proj", seed=seed + 9999) # embed_dim=256, tgt_vocab=15000
        self.dropout = Dropout(dropout_p, seed=seed + 4242) # dropout_p=0.1
        self.max_len = max_len # 20
        self.training = True
        # caches for backward across layer stacks
        self._cache_enc_inputs = None
        self._cache_dec_inputs = None

    @property
    def params(self):
        ps = []
        for m in [self.embed_src, self.embed_tgt, *self.enc_layers, *self.dec_layers, self.proj]:
            ps += m.params if hasattr(m, 'params') else []
        ps += [self.pos_src, self.pos_tgt]
        return ps

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def forward(self, src_ids: np.ndarray, tgt_in_ids: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training
        B, Ts = src_ids.shape # src_ids.shape = (32, 20), B=32, Ts=20
        _, Tt = tgt_in_ids.shape # tgt_in_ids.shape = (32, 20), Tt=20
        # Embeddings + positions
        hs = self.embed_src.forward(src_ids) + self.pos_src.value[:Ts][None, :, :] # self.pos_src.value.shape=(128, 256), hs.shape = (32, 20, 256)
        ht = self.embed_tgt.forward(tgt_in_ids) + self.pos_tgt.value[:Tt][None, :, :] # self.pos_tgt.value.shape=(128, 256), ht.shape = (32, 20, 256)
        # masks
        src_pad = (src_ids == SPECIAL_TOKENS["<PAD>"]) # src_pad.shape = (32, 20)
        tgt_pad = (tgt_in_ids == SPECIAL_TOKENS["<PAD>"]) # tgt_pad.shape = (32, 20)
        # Encoder
        for layer in self.enc_layers: # only 1 layer
            hs = layer.forward(hs, training=training, src_pad=src_pad) # hs.shape = (64,20,256) -> (64,20,256), src_pad.shape = (64,20)
        # Decoder
        for layer in self.dec_layers: # only 1 layer
            ht = layer.forward(ht, enc_out=hs, training=training, dec_pad=tgt_pad, enc_pad=src_pad) # ht.shape = (64,20,256) -> (64,20,256), tgt_pad.shape = (64,20), src_pad.shape = (64,20)
        # Final proj to vocab (apply dropout on decoder outputs)
        self.dropout.training = training
        h = self.dropout.forward(ht) # ht.shape = (64,20,256) -> h.shape = (64,20,256)
        logits = self.proj.forward(h)  # h.shape= (64,20,256) -> logits.shape = (64,20,20004)
        # cache for backward routing
        self._cache_enc_inputs = (src_pad,)
        self._cache_dec_inputs = (tgt_pad,)
        return logits

    def backward(self, dlogits: np.ndarray):
        # dlogits: (B,T,V)
        dh = self.proj.backward(dlogits) # dlogits.shape = (64,20,15000) -> dh.shape = (64,20,256)
        dh = self.dropout.backward(dh) # dh.shape = (64,20,256)
        # back through decoder stack
        denc_accum = 0
        for layer in reversed(self.dec_layers): # only 1 layer
            dh, denc = layer.backward(dh) # dh.shape = (64,20,256) -> dh.shape = (64,20,256), denc.shape = (64,20,256)
            denc_accum = denc_accum + denc # denc_accum.shape = (64,20,256)
        # back through encoder stack (propagate denc_accum)
        grad = denc_accum # grad.shape = (64,20,256)
        for layer in reversed(self.enc_layers): # only 1 layer
            grad = layer.backward(grad) # grad.shape = (64,20,256) -> grad.shape = (64,20,256)
        # back into embeddings
        self.embed_tgt.backward(dh) # dh.shape = (64,20,256)
        self.embed_src.backward(grad) # grad.shape = (64,20,256)

    # ------------- Loss / metrics -------------

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp(logits - m)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

    def loss_and_acc(self, logits: np.ndarray, tgt_out_ids: np.ndarray, pad_id: int = 0) -> Tuple[float, float, np.ndarray]:
        # logits: (B,T,V) ; tgt_out_ids: (B,T)
        probs = self.softmax(logits) # logits.shape = (64,20,15000), probs.shape = (64,20,15000)
        B, T, V = probs.shape # B=64, T=20, V=15000
        onehot = np.zeros_like(probs)
        # mask out-of-range ids
        tgt = np.clip(tgt_out_ids, 0, V - 1) # tgt_out_ids.shape = (64,20), tgt.shape = (64,20)
        onehot[np.arange(B)[:, None], np.arange(T)[None, :], tgt] = 1.0
        # cross-entropy with mask
        pad_mask = (tgt_out_ids == pad_id) # tgt_out_ids.shape = (64,20), pad_mask.shape = (64,20)
        # avoid log(0)
        logp = np.log(probs + 1e-12)
        nll = - (onehot * logp).sum(axis=-1) # onehot.shape = (64,20,15000), logp.shape = (64,20,15000) , nll.shape = (64,20)
        n_tokens = np.maximum(1, (~pad_mask).sum()) #n_tokens = 427
        loss = float(nll[~pad_mask].sum() / n_tokens)
        # accuracy
        preds = probs.argmax(axis=-1) # probs.shape = (64,20,15000), preds.shape = (64,20)
        acc = float((preds[~pad_mask] == tgt[~pad_mask]).mean()) if n_tokens > 0 else 0.0
        # gradient wrt logits
        dlogits = (probs - onehot) / n_tokens
        dlogits[pad_mask] = 0.0 # dlogits.shape = (64,20,15000)
        return loss, acc, dlogits

    # ------------- Greedy decoding -------------
    def greedy_decode(self, src_ids, stoi_tgt, itos_tgt, max_len=20):
        self.dropout.training = False
        B, Ts = src_ids.shape # src_ids.shape = (64, 20), B=64, Ts=20
        # encode once
        hs = self.embed_src.forward(src_ids) + self.pos_src.value[:Ts][None, :, :] # self._pos..value.shape= (20,256), hs.shape = (64,20,256)
        src_pad = (src_ids == SPECIAL_TOKENS["<PAD>"]) #src._pad.shape= (64,20)
        for layer in self.enc_layers: # only 1 layer
            hs = layer.forward(hs, training=False, src_pad=src_pad) # hs.shape = (64,20,256), src_pad.shape = (64,20)

        start_id = SPECIAL_TOKENS["[start]"] # start_id = 2
        end_id = SPECIAL_TOKENS["[end]"] # end_id = 3  

        y = np.full((B, 1), start_id, dtype=np.int64) # B.shape = (64,1)
        finished = np.zeros((B,), dtype=bool) # finished.shape= (64,)
        decoded_ids = [[] for _ in range(B)] # len(decoded_ids) = 64

        for _ in range(max_len): # max_len=20
            ht = self.embed_tgt.forward(y) + self.pos_tgt.value[: y.shape[1]][None, :, :] # y.shape = (64,1), ht.shape = (64,1,256)
            dec_pad = (y == SPECIAL_TOKENS["<PAD>"]) # dec_pad.shape = (64,1)
            h = ht # h.shape = (64,1,256)
            for layer in self.dec_layers: # only 1 layer
                h = layer.forward(h, enc_out=hs, training=False, dec_pad=dec_pad, enc_pad=src_pad) # h.shape= (64,1,256), hs.shape= (64,20,256), dec_pad.shape = (64,1), src_pad.shape = (64,20)
            logits = self.proj.forward(h) # logits.shape = (64,1,15000)
            probs = self.softmax(logits[:, -1:, :])  # probs.shape= (64,1,15000)  (B,1,V)
            tok = probs.argmax(axis=-1).astype(np.int64)  # (B,1) tok.shape = (64,1)

            # don’t advance finished sequences
            tok[finished, 0] = end_id
            y = np.concatenate([y, tok], axis=1) #y.shape = (64,2)

            for i in range(B): # B=64
                if not finished[i]:
                    tid = int(tok[i, 0])
                    decoded_ids[i].append(tid)
                    if tid == end_id:
                        finished[i] = True
            if finished.all():
                break

        # map to words, cut at end_id
        outs = []
        for seq in decoded_ids:  # len(decoded_ids) = 64
            words = []
            for idx in seq:
                if idx == end_id:
                    break
                words.append(itos_tgt.get(idx, "<OOV>"))
            outs.append(words)
        return outs # len (outs) = 64, outs[0] = ['un', 'hombre', 'con', 'un', 'sombrero', '.', '[end]']

# =============================================================
# Optimizer
# =============================================================

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


# =============================================================
# Dataset & training helpers
# =============================================================

def make_translation_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out = []
    for en, es in pairs:
        es2 = f"[start] {es} [end]"
        out.append((en, es2))
    return out


def split_pairs(pairs: List[Tuple[str, str]], val_frac=0.15) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    random.shuffle(pairs)
    n = len(pairs)
    n_val = int(val_frac * n)
    n_train = n - 2 * n_val
    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]
    return train, val, test


def texts_from_pairs(pairs: List[Tuple[str, str]]):
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return xs, ys


def encode_translation(pairs: List[Tuple[str, str]], stoi_src: Dict[str, int], stoi_tgt: Dict[str, int]) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    X_src, Y_tgt = texts_from_pairs(pairs)
    src_ids = [encode(x, stoi_src, is_target=False) for x in X_src]
    tgt_ids_full = [encode(y, stoi_tgt, is_target=True) for y in Y_tgt]
    # shift for teacher forcing
    tgt_in = [seq[:-1] for seq in tgt_ids_full]
    tgt_out = [seq[1:] for seq in tgt_ids_full]
    return src_ids, tgt_in, tgt_out


def iterate_minibatches_trans(
    src_ids_all: List[List[int]], tgt_in_all: List[List[int]], tgt_out_all: List[List[int]],
    batch_size: int, src_maxlen: int, tgt_maxlen: int, shuffle: bool = True,
):
    n = len(src_ids_all)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        bidx = idx[start:start + batch_size]
        src_batch = [src_ids_all[i] for i in bidx]
        tgt_in_batch = [tgt_in_all[i] for i in bidx]
        tgt_out_batch = [tgt_out_all[i] for i in bidx]
        Xs = pad_batch(src_batch, src_maxlen)
        Yi = pad_batch(tgt_in_batch, tgt_maxlen)
        Yo = pad_batch(tgt_out_batch, tgt_maxlen)
        yield Xs, Yi, Yo


# =============================================================
# Save / Load
# =============================================================

def save_weights(model: Seq2SeqTransformer, path: str):
    data = {}
    for p in model.params:
        data[p.name] = p.value
    np.savez(path, **data)


def load_weights(model: Seq2SeqTransformer, path: str):
    zz = np.load(path)
    name_to_param = {p.name: p for p in model.params}
    for k in zz.files:
        if k in name_to_param:
            name_to_param[k].value[...] = zz[k]
        else:
            print(f"[warn] extra weight in file: {k}")


# =============================================================
# Main training loop
# =============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--spa_dir", type=str, default="./spa-eng")
    parser.add_argument("--vocab_src", type=int, default=15000)
    parser.add_argument("--vocab_tgt", type=int, default=15000)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_hidden", type=int, default=1024)
    parser.add_argument("--enc_layers", type=int, default=1)
    parser.add_argument("--dec_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--src_maxlen", type=int, default=20)
    parser.add_argument("--tgt_maxlen", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weights", type=str, default="transformer_seq2seq_np.npz")
    args = parser.parse_args()

    print("Loading spa-eng pairs...")
    all_pairs = load_spa_eng(args.spa_dir)
    all_pairs = make_translation_pairs(all_pairs) # 'Go.' -> '[start] Ve. [end]'

    train_pairs, val_pairs, test_pairs = split_pairs(all_pairs, val_frac=0.15)
    print(f"Train {len(train_pairs)} | Val {len(val_pairs)} | Test {len(test_pairs)}")

    src_texts_train, tgt_texts_train = texts_from_pairs(train_pairs)
    stoi_src = build_vocab(src_texts_train, args.vocab_src, is_target=False)
    stoi_tgt = build_vocab(tgt_texts_train, args.vocab_tgt, is_target=True)
    itos_tgt = {i: w for w, i in stoi_tgt.items()}

    Xs_tr, Yi_tr, Yo_tr = encode_translation(train_pairs, stoi_src, stoi_tgt)
    Xs_va, Yi_va, Yo_va = encode_translation(val_pairs, stoi_src, stoi_tgt)

    model = Seq2SeqTransformer(
        src_vocab=len(stoi_src), tgt_vocab=len(stoi_tgt),
        embed_dim=args.embed_dim, num_heads=args.num_heads, ff_hidden=args.ff_hidden,
        num_enc_layers=args.enc_layers, num_dec_layers=args.dec_layers,
        dropout_p=args.dropout, max_len=max(args.src_maxlen, args.tgt_maxlen), seed=0,
    )

    opt = RMSprop(model.params, lr=args.lr)

    best_val = 1e9
    for ep in range(1, args.epochs + 1):
        # ---------------- Train ----------------
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Epoch {ep:02d} starting at {start_time}")
        losses = []
        accs = []
        for Xs, Yi, Yo in iterate_minibatches_trans(Xs_tr, Yi_tr, Yo_tr, args.batch_size, args.src_maxlen, args.tgt_maxlen, shuffle=True):
            model.zero_grad()
            logits = model.forward(Xs, Yi, training=True) #Xs.shape=(64,20) Yi.shape=(64,20) logits.shape=(64,20,15000)
            loss, acc, dlogits = model.loss_and_acc(logits, Yo, pad_id=SPECIAL_TOKENS["<PAD>"]) # Yo.shape=(64,20) loss=6.908 acc=0.0 dlogits.shape=(64,20,15000)
            losses.append(loss)
            accs.append(acc)
            model.backward(dlogits) # dlogits.shape=(64,20,15000)
            opt.step()
        tr_loss = float(np.mean(losses)) if losses else float('nan')
        tr_acc = float(np.mean(accs)) if accs else 0.0

        # ---------------- Val ----------------
        v_losses = []
        v_accs = []
        for Xs, Yi, Yo in iterate_minibatches_trans(Xs_va, Yi_va, Yo_va, args.batch_size, args.src_maxlen, args.tgt_maxlen, shuffle=False):
            logits = model.forward(Xs, Yi, training=False)
            loss, acc, _ = model.loss_and_acc(logits, Yo, pad_id=SPECIAL_TOKENS["<PAD>"])
            v_losses.append(loss)
            v_accs.append(acc)
        va_loss = float(np.mean(v_losses)) if v_losses else float('nan')
        va_acc = float(np.mean(v_accs)) if v_accs else 0.0

        print(f"Epoch {ep:02d}: train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            save_weights(model, args.weights)
            print(f"  ↑ Saved best to {args.weights} (val_loss={best_val:.4f})")

    # ---------------- Test a few samples ----------------
    print("Loading best weights and decoding samples...")
    load_weights(model, args.weights)
    Xs_te, Yi_te, Yo_te = encode_translation(test_pairs[:64], stoi_src, stoi_tgt)
    Xs = pad_batch(Xs_te, args.src_maxlen)
    dec_words = model.greedy_decode(Xs, stoi_tgt, itos_tgt, max_len=args.tgt_maxlen)
    for i in range(min(10, len(test_pairs))):
        print("-")
        print("EN:", test_pairs[i][0])
        print("ES:", " ".join(dec_words[i]))
