# test_seq2seq_transformer.py
import os
import tempfile
import numpy as np
import pytest

# CHANGE THIS to your module name if different:
# from seq2seq_np import Seq2SeqTransformer, SPECIAL_TOKENS, pad_batch, Parameter, RMSprop
from chapter11_numpy_transformer_translation \
    import Seq2SeqTransformer, SPECIAL_TOKENS, pad_batch, RMSprop, save_weights, load_weights  # if running in a single file/notebook

PAD = SPECIAL_TOKENS["<PAD>"]
OOV = SPECIAL_TOKENS["<OOV>"]
START = SPECIAL_TOKENS["[start]"]
END = SPECIAL_TOKENS["[end]"]

def tiny_model(src_vocab, tgt_vocab, max_len=8, seed=0):
    return Seq2SeqTransformer(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        embed_dim=32,
        num_heads=4,
        ff_hidden=64,
        num_enc_layers=1,
        num_dec_layers=1,
        dropout_p=0.1,
        max_len=max_len,
        seed=seed,
    )

def make_tiny_batch():
    # Tiny synthetic vocab (IDs are consistent with SPECIAL_TOKENS indices)
    # src: "hello world", tgt: "[start] hola mundo [end]"
    # We'll just craft ID sequences directly.
    # src vocab size 10, tgt vocab size 12 for headroom
    src_vocab = 10
    tgt_vocab = 12

    # batch of 3, length 5 (with padding)
    # Example src sentences (IDs >= 4 are normal tokens)
    Xs = np.array([
        [4, 5, PAD, PAD, PAD],   # "w1 w2"
        [4, 6, 7, PAD, PAD],     # "w1 w3 w4"
        [8, PAD, PAD, PAD, PAD], # "w5"
    ], dtype=np.int64)

    # For teacher forcing we need Yi (input) and Yo (output) for target seqs
    # Y full includes [start] ... [end]
    # We'll build 3 targets of different lengths (then pad)
    Yi = np.array([
        [START,  9, 10,  END, PAD],  # [start] t1 t2 [end]
        [START,  9,  END, PAD, PAD], # [start] t1 [end]
        [START,  END, PAD, PAD, PAD],# [start] [end]
    ], dtype=np.int64)

    Yo = np.array([
        [9, 10, END, PAD, PAD],  # shifted by 1
        [9, END, PAD, PAD, PAD],
        [END, PAD, PAD, PAD, PAD],
    ], dtype=np.int64)

    return src_vocab, tgt_vocab, Xs, Yi, Yo

def test_forward_shapes_and_masking():
    src_vocab, tgt_vocab, Xs, Yi, Yo = make_tiny_batch()
    model = tiny_model(src_vocab, tgt_vocab, max_len=max(Xs.shape[1], Yi.shape[1]), seed=123)

    logits = model.forward(Xs, Yi, training=True)
    assert logits.shape == (Xs.shape[0], Yi.shape[1], tgt_vocab)

    # Loss should ignore PAD positions; dlogits must be zero at PADs
    loss, acc, dlogits = model.loss_and_acc(logits, Yo, pad_id=PAD)
    assert dlogits.shape == logits.shape
    pad_mask = (Yo == PAD)
    assert np.all(dlogits[pad_mask] == 0.0)
    assert np.isfinite(loss), "Loss should be finite"

def test_backward_and_nonzero_grads():
    src_vocab, tgt_vocab, Xs, Yi, Yo = make_tiny_batch()
    model = tiny_model(src_vocab, tgt_vocab, max_len=max(Xs.shape[1], Yi.shape[1]), seed=0)

    logits = model.forward(Xs, Yi, training=True)
    loss, acc, dlogits = model.loss_and_acc(logits, Yo, pad_id=PAD)

    # Backprop
    model.zero_grad()
    model.backward(dlogits)

    # Check some representative params have nonzero grads and no NaNs
    touched = 0
    for p in model.params:
        assert np.all(np.isfinite(p.grad)), f"NaN/Inf in grad for {p.name}"
        if np.any(np.abs(p.grad) > 0):
            touched += 1
    assert touched > 0, "At least some parameters should receive gradients"

def test_one_optimizer_step_reduces_loss_a_bit():
    src_vocab, tgt_vocab, Xs, Yi, Yo = make_tiny_batch()
    model = tiny_model(src_vocab, tgt_vocab, max_len=max(Xs.shape[1], Yi.shape[1]), seed=42)
    opt = RMSprop(model.params, lr=5e-3)

    logits0 = model.forward(Xs, Yi, training=True)
    loss0, _, dlogits0 = model.loss_and_acc(logits0, Yo, pad_id=PAD)

    model.zero_grad()
    model.backward(dlogits0)
    opt.step()

    logits1 = model.forward(Xs, Yi, training=True)
    loss1, _, _ = model.loss_and_acc(logits1, Yo, pad_id=PAD)

    # With random init and tiny batch, loss should not increase wildly.
    # Often it will decrease; allow small tolerance in case of noise.
    assert loss1 <= loss0 + 0.05, f"Loss should not get much worse after one step: {loss0} -> {loss1}"

def test_save_and_load_roundtrip(tmp_path: pytest.TempPathFactory):
    src_vocab, tgt_vocab, Xs, Yi, Yo = make_tiny_batch()
    model = tiny_model(src_vocab, tgt_vocab, max_len=max(Xs.shape[1], Yi.shape[1]), seed=7)

    # Save
    #weights_path = os.path.join(tempfile.gettempdir(), "unit_test_transformer_weights.npz")
    weights_path = os.path.join(tempfile.gettempdir(), "transformer_seq2seq_np.npz")
    # Reuse helpers in your code:
    # from seq2seq_np import save_weights, load_weights
    # from __main__ import save_weights, load_weights  # adjust as needed
    # save_weights(model, weights_path)

    # Perturb a param
    before = model.params[0].value.copy()
    model.params[0].value += 123.456

    # Load and check restored
    load_weights(model, weights_path)
    after = model.params[0].value
    assert np.allclose(before, after), "Weights should be restored exactly after load()"

def test_greedy_decode_contract_no_end_token_in_output_words():
    # Build tiny vocab mappings for decoding
    src_vocab, tgt_vocab, Xs, Yi, Yo = make_tiny_batch()
    model = tiny_model(src_vocab, tgt_vocab, max_len=max(Xs.shape[1], Yi.shape[1]), seed=99)

    # Fake a tiny stoi/itos for target vocab so that [end] maps to the literal string "[end]"
    itos_tgt = {PAD: "<PAD>", OOV: "<OOV>", START: "[start]", END: "[end]"}
    # Fill the rest with dummy tokens
    for i in range(tgt_vocab):
        if i not in itos_tgt:
            itos_tgt[i] = f"t{i}"

    # Greedy decode uses only src_ids + internal model; just pass Xs as sources.
    outs = model.greedy_decode(Xs, stoi_tgt=None, itos_tgt=itos_tgt, max_len=5)
    assert len(outs) == Xs.shape[0]
    # Contract: decoded word lists should NOT include "[end]" because the function truncates there.
    for words in outs:
        assert "[end]" not in words, "greedy_decode should not include the [end] token in its returned words"
        assert len(words) <= 5

def test_determinism_with_fixed_seed():
    src_vocab, tgt_vocab, Xs, Yi, Yo = make_tiny_batch()
    m1 = tiny_model(src_vocab, tgt_vocab, max_len=max(Xs.shape[1], Yi.shape[1]), seed=1234)
    m2 = tiny_model(src_vocab, tgt_vocab, max_len=max(Xs.shape[1], Yi.shape[1]), seed=1234)

    logits1 = m1.forward(Xs, Yi, training=False)
    logits2 = m2.forward(Xs, Yi, training=False)

    assert np.allclose(logits1, logits2), "Models with same seed/inputs should be deterministic"

