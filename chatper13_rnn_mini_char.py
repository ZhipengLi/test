import numpy as np
from urllib.request import urlopen

# data I/O
url = "https://raw.githubusercontent.com/amephraim/nlp/refs/heads/master/texts/J.%20K.%20Rowling%20-%20Harry%20Potter%201%20-%20Sorcerer's%20Stone.txt"  # String (download URL).
data = urlopen(url).read().decode("utf-8")  # String of 439742 characters.
chars = sorted(set(data))  # List of 79 characters.
data_size, vocab_size = len(data), len(chars)  # Both are integer scalars.
print(f"data has {data_size} characters, {vocab_size} unique.")

char_to_ix = {ch: i for i, ch in enumerate(chars)}  # Dictionary: character -> scalar index; 79 entries.
ix_to_char = {i: ch for i, ch in enumerate(chars)}  # Dictionary: scalar index -> character; 79 entries.

# hyperparameters
hidden_size = 100  # Integer scalar; number of hidden state values.
seq_length = 25  # Integer scalar; number of time steps per training iteration.
learning_rate = 1e-1  # Float scalar.

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # Shape: (100, 79).
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Shape: (100, 100).
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # Shape: (79, 100).
bh = np.zeros((hidden_size, 1))  # Shape: (100, 1).
by = np.zeros((vocab_size, 1))  # Shape: (79, 1).

def lossFun(inputs, targets, hprev):  # inputs, targets: lists of 25 indices; hprev: (100, 1).
    xs, hs, ys, ps = {}, {}, {}, {}  # Dictionary value shapes: xs, ys, ps: (79, 1); hs: (100, 1).
    hs[-1] = np.copy(hprev)  # Shape: (100, 1).
    loss = 0  # Scalar; sum of the losses for all time steps.

    # forward pass
    for t in range(len(inputs)):  # t: integer scalar.
        xs[t] = np.zeros((vocab_size, 1))  # Shape: (79, 1).
        xs[t][inputs[t]] = 1  # Selected row: (1,); sets the active one-hot entry.
        hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t - 1] + bh)  # Shape: (100, 1).
        ys[t] = Why @ hs[t] + by  # Shape: (79, 1).
        exp_y = np.exp(ys[t] - np.max(ys[t]))  # Shape: (79, 1); np.max returns a scalar.
        ps[t] = exp_y / np.sum(exp_y)  # Shape: (79, 1); np.sum returns a scalar.
        loss += -np.log(ps[t][targets[t], 0] + 1e-12)  # Scalar; ps[t][targets[t], 0] is a scalar probability.

    # backward pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)  # Shapes: (100, 79), (100, 100), (79, 100).
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)  # Shapes: (100, 1), (79, 1).
    dhnext = np.zeros_like(hs[0])  # Shape: (100, 1).

    for t in reversed(range(len(inputs))):  # t: integer scalar.
        dy = np.copy(ps[t])  # Shape: (79, 1).
        dy[targets[t]] -= 1  # Selected row: (1,); dy remains (79, 1).
        dWhy += dy @ hs[t].T  # Shape: (79, 100).
        dby += dy  # Shape: (79, 1).
        dh = Why.T @ dy + dhnext  # Shape: (100, 1).
        dhraw = (1 - hs[t] ** 2) * dh  # Shape: (100, 1).
        dbh += dhraw  # Shape: (100, 1).
        dWxh += dhraw @ xs[t].T  # Shape: (100, 79).
        dWhh += dhraw @ hs[t - 1].T  # Shape: (100, 100).
        dhnext = Whh.T @ dhraw  # Shape: (100, 1).

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:  # dparam shapes in loop order: (100, 79), (100, 100), (79, 100), (100, 1), (79, 1).
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

def sample(h, seed_ix, n):  # h: (100, 1); seed_ix, n: integer scalars.
    x = np.zeros((vocab_size, 1))  # Shape: (79, 1).
    x[seed_ix] = 1  # Selected row: (1,); sets the active one-hot entry.
    ixes = []  # List of scalar indices; length grows from 0 to 200 in the current call.

    for _ in range(n):  # _: integer scalar (unused loop index).
        h = np.tanh(Wxh @ x + Whh @ h + bh)  # Shape: (100, 1).
        y = Why @ h + by  # Shape: (79, 1).
        exp_y = np.exp(y - np.max(y))  # Shape: (79, 1); np.max returns a scalar.
        p = exp_y / np.sum(exp_y)  # Shape: (79, 1).

        ix = np.random.choice(vocab_size, p=p.ravel())  # ix: integer scalar; p.ravel(): (79,).
        x = np.zeros((vocab_size, 1))  # Shape: (79, 1).
        x[ix] = 1  # Selected row: (1,); sets the active one-hot entry.
        ixes.append(ix)

    return ixes

# training
p = 0  # Integer scalar; current position in the source text.
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)  # Shapes: (100, 79), (100, 100), (79, 100).
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # Shapes: (100, 1), (79, 1).
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # Scalar.

for n in range(20000):  # n: integer scalar (training iteration index).
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # Shape: (100, 1).
        p = 0  # Integer scalar; current position in the source text.

    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]  # List of 25 scalar indices.
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]  # List of 25 scalar indices.

    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)  # List of 200 scalar indices.
        txt = "".join(ix_to_char[ix] for ix in sample_ix)  # String of 200 characters.
        print(f"\n----\n{txt}\n----")

    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)  # loss: scalar; dWxh: (100, 79); dWhh: (100, 100); dWhy: (79, 100); dbh: (100, 1); dby: (79, 1); hprev: (100, 1).
    smooth_loss = smooth_loss * 0.999 + loss * 0.001  # Scalar.

    if n % 100 == 0:
        print(f"iter {n}, loss: {smooth_loss:.4f}")

    for param, dparam, mem in zip(
        [Wxh, Whh, Why, bh, by],
        [dWxh, dWhh, dWhy, dbh, dby],
        [mWxh, mWhh, mWhy, mbh, mby],
    ):  # param, dparam, mem shapes in loop order: (100, 79), (100, 100), (79, 100), (100, 1), (79, 1).
        mem += dparam * dparam  # Shapes in loop order: (100, 79), (100, 100), (79, 100), (100, 1), (79, 1).
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # Shapes in loop order: (100, 79), (100, 100), (79, 100), (100, 1), (79, 1).

    p += seq_length  # Integer scalar.

print("Training finished.")
