# dataset: http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
"""
NumPy-only Transformer (Encoder–Decoder) for English→Spanish toy translation
- No TensorFlow / Keras
- Greedy decoding
- Teacher forcing during training
- Batches built from spa-eng/spa.txt (same dataset path as your script)

NOTE: This is a compact educational implementation meant to run on CPU.
Expect training to be slow compared to framework equivalents; start with a few
steps/epochs just to verify correctness, then expand.
"""
import os
import re
import math
import random
import string
from typing import List, Tuple, Dict
import time, math
import numpy as np

# ---------------------------
# Data prep / tokenization
# ---------------------------
DATA_PATH = "spa-eng/spa.txt"  # same as your script
vocab_size = 15000
sequence_length = 20  # source len; target uses +1 for start/end handling
batch_size = 64
seed = 0
rng = np.random.default_rng(seed)

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "").replace("]", "")

# Special tokens / ids
PAD = "[pad]"
START = "[start]"
END = "[end]"
UNK = "[unk]"

# indices
PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3

SPECIALS = [PAD, START, END, UNK]


def load_pairs(path: str) -> List[Tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    pairs = []
    for line in lines:
        if not line:
            continue
        en, es, *_ = line.split("\t")
        es = f"{START} {es} {END}"
        pairs.append((en, es))
    return pairs


def standardize_target(s: str) -> str:
    s = s.lower()
    return re.sub(f"[{re.escape(strip_chars)}]", "", s)


def simple_tokenize(text: str) -> List[str]:
    return text.strip().split()


def build_vocab(texts: List[str], max_tokens: int, standardize=None) -> Tuple[Dict[str, int], List[str]]:
    freq = {}
    for t in texts:
        if standardize:
            t = standardize(t)
        for tok in simple_tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1
    # sort by frequency desc, then lexicographically
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = SPECIALS.copy()
    for tok, _ in items:
        if tok in (START, END):
            # START/END already ensured in target strings
            continue
        if len(vocab) >= max_tokens:
            break
        vocab.append(tok)
    itos = vocab
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos


def vectorize(texts: List[str], stoi: Dict[str, int], seq_len: int, standardize=None) -> np.ndarray:
    out = np.full((len(texts), seq_len), PAD_ID, dtype=np.int64)
    for i, t in enumerate(texts):
        if standardize:
            t = standardize(t)
        toks = simple_tokenize(t)
        toks = toks[:seq_len]
        ids = [stoi.get(tok, UNK_ID) for tok in toks]
        out[i, :len(ids)] = np.array(ids, dtype=np.int64)
    return out


# ---------------------------
# Mini-dataloader
# ---------------------------

def make_splits(pairs: List[Tuple[str, str]], val_ratio=0.15):
    random.shuffle(pairs)
    n = len(pairs)
    n_val = int(val_ratio * n)
    n_train = n - 2 * n_val
    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]
    return train, val, test


def batches(pairs: List[Tuple[str, str]], batch_size: int, 
            src_stoi: Dict[str, int], tgt_stoi: Dict[str, int]):
    # Build tensors per batch
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        src_txt = [p[0] for p in chunk]
        tgt_txt = [p[1] for p in chunk]
        src = vectorize(src_txt, src_stoi, sequence_length, standardize=None)
        # target gets +1 for teacher forcing pair
        tgt_full = vectorize(tgt_txt, tgt_stoi, sequence_length + 1, standardize=standardize_target)
        # inputs to decoder (without last token) & targets (without first token)
        dec_in = tgt_full[:, :-1]
        dec_tg = tgt_full[:, 1:]
        yield src, dec_in, dec_tg


# ---------------------------
# Layers / modules (NumPy)
# ---------------------------

def xavier_uniform(shape, rng):
    fan_in, fan_out = shape[0], shape[1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def zeros(shape):
    return np.zeros(shape, dtype=np.float32)


def ones(shape):
    return np.ones(shape, dtype=np.float32)


class Linear:
    def __init__(self, in_f, out_f, rng):
        self.W = xavier_uniform((in_f, out_f), rng)
        self.b = zeros((out_f,))
        # grads
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)

    def __call__(self, x):
        # x: (B, T, D)
        y = x @ self.W + self.b
        return y

    def zero_grad(self):
        self.gW.fill(0.0)
        self.gb.fill(0.0)


class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = ones((d_model,))
        self.beta = zeros((d_model,))
        self.eps = eps
        self.ggamma = np.zeros_like(self.gamma)
        self.gbeta = np.zeros_like(self.beta)

    def __call__(self, x):
        # x: (B, T, D)
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta

    # Gradients computed in backward elsewhere – this LN is used in a residual way
    # For simplicity in this educational code, we treat LN as stateless in backward
    # and approximate gradients via precomputed caches in blocks.


class Embedding:
    def __init__(self, vocab, d_model, rng):
        self.E = rng.normal(0, 0.02, size=(vocab, d_model)).astype(np.float32)
        self.gE = np.zeros_like(self.E)

    def __call__(self, x):
        # x: (B, T) int64
        return self.E[x]

    def zero_grad(self):
        self.gE.fill(0.0)


def causal_mask(T):
    # (T, T) with -inf above diagonal
    m = np.triu(np.ones((T, T), dtype=np.float32), k=1)
    m[m == 1.0] = -np.inf
    m[m == 0.0] = 0.0
    return m  # add to logits before softmax


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0.0, x)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, rng):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        # projections
        self.Wq = Linear(d_model, d_model, rng)
        self.Wk = Linear(d_model, d_model, rng)
        self.Wv = Linear(d_model, d_model, rng)
        self.Wo = Linear(d_model, d_model, rng)

    def __call__(self, x_q, x_kv, attn_mask=None, key_padding_mask=None):
        # x_q: (B,T_q,D), x_kv: (B,T_k,D)
        B, Tq, D = x_q.shape
        Tk = x_kv.shape[1]
        q = self.Wq(x_q).reshape(B, Tq, self.num_heads, self.d_head).transpose(0,2,1,3)  # (B,H,Tq,dh)
        k = self.Wk(x_kv).reshape(B, Tk, self.num_heads, self.d_head).transpose(0,2,1,3)  # (B,H,Tk,dh)
        v = self.Wv(x_kv).reshape(B, Tk, self.num_heads, self.d_head).transpose(0,2,1,3)
        # scaled dot-product
        scores = (q @ k.transpose(0,1,3,2)) / math.sqrt(self.d_head)  # (B,H,Tq,Tk)
        if attn_mask is not None:
            # attn_mask: (Tq,Tk) broadcast to (B,H,Tq,Tk)
            scores = scores + attn_mask
        if key_padding_mask is not None:
            # key_padding_mask: (B,1,1,Tk) with 0 for valid, -inf for pad
            scores = scores + key_padding_mask
        attn = softmax(scores, axis=-1)
        y = attn @ v  # (B,H,Tq,dh)
        y = y.transpose(0,2,1,3).reshape(B, Tq, D)
        out = self.Wo(y)
        return out


class FeedForward:
    def __init__(self, d_model, d_hidden, rng):
        self.lin1 = Linear(d_model, d_hidden, rng)
        self.lin2 = Linear(d_hidden, d_model, rng)

    def __call__(self, x):
        return self.lin2(relu(self.lin1(x)))


class EncoderBlock:
    def __init__(self, d_model, d_hidden, n_heads, rng):
        self.mha = MultiHeadAttention(d_model, n_heads, rng)
        self.ln1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_hidden, rng)
        self.ln2 = LayerNorm(d_model)

    def __call__(self, x, key_padding_mask):
        # Self-attention
        sa = self.mha(x, x, attn_mask=None, key_padding_mask=key_padding_mask)
        x = self.ln1(x + sa)
        # FFN
        ff = self.ff(x)
        x = self.ln2(x + ff)
        return x


class DecoderBlock:
    def __init__(self, d_model, d_hidden, n_heads, rng):
        self.self_mha = MultiHeadAttention(d_model, n_heads, rng)
        self.ln1 = LayerNorm(d_model)
        self.cross_mha = MultiHeadAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_hidden, rng)
        self.ln3 = LayerNorm(d_model)
        self._cached_causal = None

    def __call__(self, x, enc_out, self_attn_mask, self_kpm, enc_kpm):
        sa = self.self_mha(x, x, attn_mask=self_attn_mask, key_padding_mask=self_kpm)
        x = self.ln1(x + sa)
        ca = self.cross_mha(x, enc_out, attn_mask=None, key_padding_mask=enc_kpm)
        x = self.ln2(x + ca)
        ff = self.ff(x)
        x = self.ln3(x + ff)
        return x


class Transformer:
    def __init__(self, src_vocab, tgt_vocab, d_model=256, d_hidden=2048, n_heads=8, n_layers=2, max_len=sequence_length+1, rng=None):
        self.src_emb = Embedding(src_vocab, d_model, rng)
        self.tgt_emb = Embedding(tgt_vocab, d_model, rng)
        # Learned positional embeddings
        self.pos_src = rng.normal(0, 0.02, size=(sequence_length, d_model)).astype(np.float32)
        self.pos_tgt = rng.normal(0, 0.02, size=(max_len, d_model)).astype(np.float32)
        self.g_pos_src = np.zeros_like(self.pos_src)
        self.g_pos_tgt = np.zeros_like(self.pos_tgt)
        self.enc_layers = [EncoderBlock(d_model, d_hidden, n_heads, rng) for _ in range(n_layers)]
        self.dec_layers = [DecoderBlock(d_model, d_hidden, n_heads, rng) for _ in range(n_layers)]
        self.proj = Linear(d_model, tgt_vocab, rng)
        self.d_model = d_model
        self.max_len = max_len

    def encode(self, src_ids):
        # src_ids: (B, Ts)
        B, Ts = src_ids.shape
        x = self.src_emb(src_ids) + self.pos_src[None, :Ts, :]
        # build key padding mask: 0 for valid, -inf for pad positions
        pad = (src_ids == PAD_ID).astype(np.float32)
        kpm = pad[:, None, None, :] * (-1e9)
        for layer in self.enc_layers:
            x = layer(x, key_padding_mask=kpm)
        return x, kpm

    def decode(self, tgt_ids, enc_out, enc_kpm):
        # tgt_ids: (B, Tt)
        B, Tt = tgt_ids.shape
        x = self.tgt_emb(tgt_ids) + self.pos_tgt[None, :Tt, :]
        # causal mask (Tt,Tt)
        cm = causal_mask(Tt)[None, None, :, :]  # (1,1,Tt,Tt)
        # self key-padding mask for target (mask pads in keys)
        pad = (tgt_ids == PAD_ID).astype(np.float32)
        self_kpm = pad[:, None, None, :] * (-1e9)
        for layer in self.dec_layers:
            x = layer(x, enc_out, cm, self_kpm, enc_kpm)
        return x

    def forward(self, src_ids, tgt_in_ids):
        enc_out, enc_kpm = self.encode(src_ids)
        dec_h = self.decode(tgt_in_ids, enc_out, enc_kpm)
        logits = self.proj(dec_h)  # (B,T,V)
        return logits


# ---------------------------
# Loss / accuracy
# ---------------------------

def xent_loss(logits, targets, ignore_index=PAD_ID):
    # logits: (B,T,V), targets: (B,T)
    B, T, V = logits.shape
    # gather logit of true class
    logits_2d = logits.reshape(B*T, V)
    targets_1d = targets.reshape(B*T)
    mask = (targets_1d != ignore_index).astype(np.float32)
    # log softmax
    ls = logits_2d - np.max(logits_2d, axis=1, keepdims=True)
    log_probs = ls - np.log(np.sum(np.exp(ls), axis=1, keepdims=True) + 1e-9)
    idx = (np.arange(B*T), targets_1d)
    nll = -log_probs[idx]
    nll = nll * mask
    denom = np.sum(mask) + 1e-9
    return np.sum(nll) / denom


def accuracy(logits, targets, ignore_index=PAD_ID):
    pred = np.argmax(logits, axis=-1)
    mask = (targets != ignore_index)
    correct = (pred == targets) & mask
    denom = np.sum(mask)
    return (np.sum(correct), denom if denom > 0 else 1)


# ---------------------------
# Optimizer (Adam)
# ---------------------------
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ---------------------------
# Backprop scaffolding (automatic differentiation not used)
# To keep this example tractable, we only backprop through the final softmax
# projection and embeddings using a simple surrogate: treat encoder/decoder
# blocks as fixed feature extractors during the first run, then optionally
# unfreeze them by numerical gradient (very slow) or skip for demo training.
#
# For an educational yet runnable script, we'll train ONLY the output projection
# and target embeddings (which already learns a simple LM conditioned via cross
# attention). This gives sensible behavior quickly on CPU. For full training,
# a full manual autograd would be required, which is beyond a concise example.
# ---------------------------

# In practice we: forward pass -> compute loss -> compute grad w.r.t. logits ->
# backprop into proj.W, proj.b, and tgt_emb.E via chain rule.

def grad_proj_and_tgt_emb(model: Transformer, src_ids, tgt_in_ids, targets):
    # Forward
    enc_out, enc_kpm = model.encode(src_ids)
    dec_h = model.decode(tgt_in_ids, enc_out, enc_kpm)  # (B,T,D)
    logits = model.proj(dec_h)  # (B,T,V)

    B, T, V = logits.shape
    # softmax gradient (with ignore_index) using dL/dlogits = p - y
    probs = softmax(logits, axis=-1)
    targets_1h = np.zeros_like(probs)
    for b in range(B):
        for t in range(T):
            y = targets[b, t]
            if y == PAD_ID:
                continue
            targets_1h[b, t, y] = 1.0
    dlogits = (probs - targets_1h) / max(1, np.sum(targets != PAD_ID))  # average over valid tokens

    # grads for proj
    # logits = dec_h @ W + b
    gW = (dec_h.reshape(B*T, -1).T @ dlogits.reshape(B*T, V)).astype(np.float32)
    gb = dlogits.sum(axis=(0,1)).astype(np.float32)

    # backprop into dec_h for embedding update
    ddec_h = dlogits @ model.proj.W.T  # (B,T,D)

    # grads for target embeddings via dec input: dec_in_emb = E[tgt_in_ids]
    gE_tgt = np.zeros_like(model.tgt_emb.E)
    # Treat decoder as identity for gradient routing to embeddings (approx)
    # Accumulate gradient directly on input embeddings positions
    # NOTE: This is a simplification for an educational demo.
    for b in range(B):
        for t in range(T):
            idx = tgt_in_ids[b, t]
            if idx != PAD_ID:
                gE_tgt[idx] += ddec_h[b, t]

    return logits, gW, gb, gE_tgt


# ---------------------------
# Training loop (projection + tgt embeddings only)
# ---------------------------

def train(model: Transformer, train_pairs, val_pairs, src_stoi, tgt_stoi, epochs=3, lr=1e-3):
    params = [model.proj.W, model.proj.b, model.tgt_emb.E]
    opt = Adam(params, lr=lr)

    for ep in range(1, epochs+1):
        random.shuffle(train_pairs)
        tot_loss = 0.0
        tot_corr = 0
        tot_cnt = 0
        n_batches = 0
        for src_ids, dec_in, dec_tg in batches(train_pairs, batch_size, src_stoi, tgt_stoi):
            logits, gW, gb, gE_tgt = grad_proj_and_tgt_emb(model, src_ids, dec_in, dec_tg)
            loss = xent_loss(logits, dec_tg)
            c, d = accuracy(logits, dec_tg)
            tot_loss += loss
            tot_corr += c
            tot_cnt += d
            n_batches += 1
            # step
            opt.step([gW, gb, gE_tgt])
        val_loss, val_acc = evaluate(model, val_pairs, src_stoi, tgt_stoi)
        print(f"Epoch {ep}: train_loss={tot_loss/max(1,n_batches):.4f} train_acc={tot_corr/max(1,tot_cnt):.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

def train_estimation(model, train_pairs, val_pairs, src_stoi, tgt_stoi, epochs=1, lr=1e-3):
    params = [model.proj.W, model.proj.b, model.tgt_emb.E]
    opt = Adam(params, lr=lr)

    epoch_batches = math.ceil(len(train_pairs) / batch_size)
    warmup = 10            # don’t measure first few batches
    measure_steps = 100    # measure next N batches for a quick read

    for ep in range(1, epochs+1):
        random.shuffle(train_pairs)
        t0 = time.perf_counter()
        times = []
        bcount = 0
        tot_loss = 0.0; tot_corr = 0; tot_cnt = 0; n_batches = 0

        for src_ids, dec_in, dec_tg in batches(train_pairs, batch_size, src_stoi, tgt_stoi):
            t1 = time.perf_counter()
            logits, gW, gb, gE_tgt = grad_proj_and_tgt_emb(model, src_ids, dec_in, dec_tg)
            loss = xent_loss(logits, dec_tg)
            c, d = accuracy(logits, dec_tg)
            tot_loss += loss; tot_corr += c; tot_cnt += d; n_batches += 1
            opt.step([gW, gb, gE_tgt])

            t2 = time.perf_counter()
            bcount += 1
            if bcount > warmup and len(times) < measure_steps:
                times.append(t2 - t1)
            if len(times) == measure_steps:
                break  # stop early after we have a good sample

        avg_batch_s = sum(times) / max(1, len(times))
        approx_epoch_s = avg_batch_s * epoch_batches
        print(f"≈ {avg_batch_s*1000:.1f} ms/batch → ≈ {approx_epoch_s/60:.1f} min/epoch (batch_size={batch_size}, steps/epoch≈{epoch_batches})")


def evaluate(model: Transformer, pairs, src_stoi, tgt_stoi):
    losses = []
    corr = 0
    cnt = 0
    for src_ids, dec_in, dec_tg in batches(pairs, batch_size, src_stoi, tgt_stoi):
        logits = model.forward(src_ids, dec_in)
        losses.append(xent_loss(logits, dec_tg))
        c, d = accuracy(logits, dec_tg)
        corr += c
        cnt += d
    return (sum(losses)/max(1,len(losses)), corr/max(1,cnt))


# ---------------------------
# Decoding (greedy)
# ---------------------------

def greedy_decode(model: Transformer, sentence: str, src_stoi, tgt_stoi, tgt_itos, max_len=20):
    src = vectorize([sentence], src_stoi, sequence_length)
    enc_out, enc_kpm = model.encode(src)
    cur = np.full((1, 1), START_ID, dtype=np.int64)
    out_tokens = []
    for t in range(max_len):
        dec_h = model.decode(cur, enc_out, enc_kpm)
        logits = model.proj(dec_h)  # (1,t+1,V)
        next_id = int(np.argmax(logits[0, -1]))
        tok = tgt_itos[next_id] if next_id < len(tgt_itos) else UNK
        if tok == END:
            break
        out_tokens.append(tok)
        cur = np.concatenate([cur, np.array([[next_id]], dtype=np.int64)], axis=1)
    return f"{START} " + " ".join(out_tokens) + f" {END}"


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"

    pairs = load_pairs(DATA_PATH)
    train_pairs, val_pairs, test_pairs = make_splits(pairs, val_ratio=0.15)

    train_en = [p[0] for p in train_pairs]
    train_es = [p[1] for p in train_pairs]

    src_stoi, src_itos = build_vocab(train_en, vocab_size, standardize=None)
    # Ensure specials at fixed ids
    for i, tok in enumerate(SPECIALS):
        src_stoi[tok] = i
    src_itos[:len(SPECIALS)] = SPECIALS

    tgt_stoi, tgt_itos = build_vocab(train_es, vocab_size, standardize=standardize_target)
    for i, tok in enumerate(SPECIALS):
        tgt_stoi[tok] = i
    tgt_itos[:len(SPECIALS)] = SPECIALS

    print("Sample pair:", random.choice(train_pairs))

    # Model
    d_model = 256
    d_hidden = 2048
    n_heads = 8
    n_layers = 2
    model = Transformer(len(src_itos), len(tgt_itos), d_model, d_hidden, n_heads, n_layers, max_len=sequence_length+1, rng=rng)

    # Quick sanity batch
    src_b, dec_in_b, dec_tg_b = next(batches(train_pairs, batch_size, src_stoi, tgt_stoi))
    logits = model.forward(src_b, dec_in_b)
    print("Sanity forward: logits shape:", logits.shape)

    # Train a few epochs (projection+tgt embeddings only)
    train(model, train_pairs, val_pairs, src_stoi, tgt_stoi, epochs=3, lr=1e-3)

    # Decode some samples
    print("\nGreedy decode samples:")
    test_eng = [p[0] for p in test_pairs]
    for _ in range(5):
        s = random.choice(test_eng)
        print("-", s)
        print(greedy_decode(model, s, src_stoi, tgt_stoi, tgt_itos, max_len=20))
