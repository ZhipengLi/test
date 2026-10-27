import numpy as np
from urllib.request import urlopen

url = "https://raw.githubusercontent.com/amephraim/nlp/refs/heads/master/texts/J.%20K.%20Rowling%20-%20Harry%20Potter%201%20-%20Sorcerer's%20Stone.txt"  # String (download URL).
hidden_size = 100  # Scalar; numeric shape comments assume 100 hidden values and 79 unique characters.
seq_length = 25  # Scalar; number of time steps per training iteration.
learning_rate = 1e-1  # Scalar.


def sigmoid(x):  # x: (100, 1) for each gate.
    return np.exp(-np.logaddexp(0.0, -x))  # Shape: (100, 1); stable sigmoid without exponential overflow.


def lossFun(inputs, targets, hprev, cprev):  # inputs, targets: lists of 25 indices; hprev, cprev: (100, 1).
    xs, hs, cs, ps = {}, {}, {}, {}  # Dictionary value shapes: xs, ps: (79, 1); hs, cs: (100, 1).
    fs, ins, gs, os = {}, {}, {}, {}  # Dictionary value shapes: (100, 1) for each of the four gates.
    hs[-1] = np.copy(hprev)  # Shape: (100, 1); hidden state from the previous sequence.
    cs[-1] = np.copy(cprev)  # Shape: (100, 1); cell state from the previous sequence.
    loss = 0.0  # Scalar; sum of the losses for all time steps.

    for t in range(len(inputs)):  # Forward pass; t: integer scalar.
        xs[t] = np.zeros((vocab_size, 1))  # Shape: (79, 1).
        xs[t][inputs[t]] = 1  # Selected row: (1,); sets the active one-hot entry.
        gates = Wxh @ xs[t] + Whh @ hs[t - 1] + bh  # Shape: (400, 1); four stacked blocks of 100 values.
        fs[t] = sigmoid(gates[:hidden_size])  # Shape: (100, 1); forget gate controls retained cell information.
        ins[t] = sigmoid(gates[hidden_size:2 * hidden_size])  # Shape: (100, 1); input gate controls new cell information.
        gs[t] = np.tanh(gates[2 * hidden_size:3 * hidden_size])  # Shape: (100, 1); candidate values for the cell state.
        os[t] = sigmoid(gates[3 * hidden_size:])  # Shape: (100, 1); output gate controls exposed cell information.
        cs[t] = fs[t] * cs[t - 1] + ins[t] * gs[t]  # Shape: (100, 1); cell state, with element-wise multiplication.
        hs[t] = os[t] * np.tanh(cs[t])  # Shape: (100, 1); hidden state.
        y = Why @ hs[t] + by  # Shape: (79, 1).
        shifted_y = y - np.max(y)  # Shape: (79, 1); subtract the maximum for numerical stability.
        exp_y = np.exp(shifted_y)  # Shape: (79, 1).
        ps[t] = exp_y / np.sum(exp_y)  # Shape: (79, 1); next-character probabilities.
        loss += np.log(np.sum(exp_y)) - shifted_y[targets[t], 0]  # Scalar; stable negative log probability of the target.

    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)  # Shapes: (400, 79), (400, 100), (79, 100).
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)  # Shapes: (400, 1), (79, 1).
    dhnext = np.zeros_like(hprev)  # Shape: (100, 1); hidden-state gradient from the next time step.
    dcnext = np.zeros_like(cprev)  # Shape: (100, 1); cell-state gradient from the next time step.

    for t in reversed(range(len(inputs))):  # Backward pass; t: integer scalar.
        dy = np.copy(ps[t])  # Shape: (79, 1).
        dy[targets[t]] -= 1  # Selected row: (1,); dy remains (79, 1).
        dWhy += dy @ hs[t].T  # Shape: (79, 100).
        dby += dy  # Shape: (79, 1).
        dh = Why.T @ dy + dhnext  # Shape: (100, 1).
        tanh_c = np.tanh(cs[t])  # Shape: (100, 1).
        do = dh * tanh_c  # Shape: (100, 1); output-gate gradient.
        dc = dh * os[t] * (1 - tanh_c ** 2) + dcnext  # Shape: (100, 1); combined cell-state gradient.
        df = dc * cs[t - 1]  # Shape: (100, 1); forget-gate gradient.
        di = dc * gs[t]  # Shape: (100, 1); input-gate gradient.
        dg = dc * ins[t]  # Shape: (100, 1); candidate gradient.
        dgates = np.vstack((  # Shape: (400, 1); gradients before the sigmoid and tanh functions.
            df * fs[t] * (1 - fs[t]),  # Shape: (100, 1); forget block.
            di * ins[t] * (1 - ins[t]),  # Shape: (100, 1); input block.
            dg * (1 - gs[t] ** 2),  # Shape: (100, 1); candidate block.
            do * os[t] * (1 - os[t]),  # Shape: (100, 1); output block.
        ))
        dbh += dgates  # Shape: (400, 1).
        dWxh += dgates @ xs[t].T  # Shape: (400, 79).
        dWhh += dgates @ hs[t - 1].T  # Shape: (400, 100).
        dhnext = Whh.T @ dgates  # Shape: (100, 1); pass the hidden-state gradient backward.
        dcnext = dc * fs[t]  # Shape: (100, 1); pass the cell-state gradient backward.

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:  # Shapes in loop order: (400, 79), (400, 100), (79, 100), (400, 1), (79, 1).
        np.clip(dparam, -5, 5, out=dparam)  # Preserve shape; limit each gradient value to the range -5 to 5.

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1], cs[len(inputs) - 1]  # Final hidden and cell states: (100, 1) each.


def sample(h, c, seed_ix, n):  # h, c: (100, 1); seed_ix, n: integer scalars.
    x = np.zeros((vocab_size, 1))  # Shape: (79, 1).
    x[seed_ix] = 1  # Selected row: (1,); sets the active one-hot entry.
    ixes = []  # List of scalar indices; length grows from 0 to 200 in the training-loop call.

    for _ in range(n):  # Integer scalar (unused loop index).
        gates = Wxh @ x + Whh @ h + bh  # Shape: (400, 1).
        f = sigmoid(gates[:hidden_size])  # Shape: (100, 1); forget gate.
        i = sigmoid(gates[hidden_size:2 * hidden_size])  # Shape: (100, 1); input gate.
        g = np.tanh(gates[2 * hidden_size:3 * hidden_size])  # Shape: (100, 1); cell candidate.
        o = sigmoid(gates[3 * hidden_size:])  # Shape: (100, 1); output gate.
        c = f * c + i * g  # Shape: (100, 1); new cell state.
        h = o * np.tanh(c)  # Shape: (100, 1); new hidden state.
        y = Why @ h + by  # Shape: (79, 1).
        exp_y = np.exp(y - np.max(y))  # Shape: (79, 1).
        p = exp_y / np.sum(exp_y)  # Shape: (79, 1).
        ix = np.random.choice(vocab_size, p=p.ravel())  # ix: scalar; p.ravel(): (79,).
        x = np.zeros((vocab_size, 1))  # Shape: (79, 1).
        x[ix] = 1  # Selected row: (1,); use the selected character as the next input.
        ixes.append(ix)

    return ixes


if __name__ == "__main__":  # Download data and train only when this file runs as a script.
    data = urlopen(url).read().decode("utf-8")  # String of 439742 characters in the current source text.
    chars = sorted(set(data))  # List of 79 characters in the current source text.
    data_size, vocab_size = len(data), len(chars)  # Integer scalars; 439742 and 79 for the current source text.
    print(f"data has {data_size} characters, {vocab_size} unique.")
    char_to_ix = {ch: i for i, ch in enumerate(chars)}  # Dictionary: character -> scalar index; 79 entries.
    ix_to_char = {i: ch for i, ch in enumerate(chars)}  # Dictionary: scalar index -> character; 79 entries.

    Wxh = np.random.randn(4 * hidden_size, vocab_size) * 0.01  # Shape: (400, 79); forget, input, candidate, output blocks.
    Whh = np.random.randn(4 * hidden_size, hidden_size) * 0.01  # Shape: (400, 100); same block order as Wxh.
    Why = np.random.randn(vocab_size, hidden_size) * 0.01  # Shape: (79, 100).
    bh = np.zeros((4 * hidden_size, 1))  # Shape: (400, 1); four gate-bias blocks.
    bh[:hidden_size] = 1.0  # Shape: (100, 1); positive forget bias helps retain information initially.
    by = np.zeros((vocab_size, 1))  # Shape: (79, 1).

    p = 0  # Integer scalar; current position in the source text.
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)  # Shapes: (400, 79), (400, 100), (79, 100).
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # Shapes: (400, 1), (79, 1).
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # Scalar.

    for n in range(20000):  # n: integer scalar (training iteration index).
        if p + seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size, 1))  # Shape: (100, 1); reset the hidden state at the start of the text.
            cprev = np.zeros((hidden_size, 1))  # Shape: (100, 1); reset the cell state at the same time.
            p = 0  # Integer scalar.

        inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]  # List of 25 scalar indices.
        targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]  # List of 25 scalar indices.

        if n % 100 == 0:
            sample_ix = sample(hprev, cprev, inputs[0], 200)  # List of 200 scalar indices.
            txt = "".join(ix_to_char[ix] for ix in sample_ix)  # String of 200 characters.
            print(f"\n----\n{txt}\n----")

        loss, dWxh, dWhh, dWhy, dbh, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)  # loss: scalar; hprev, cprev: (100, 1).
        smooth_loss = smooth_loss * 0.999 + loss * 0.001  # Scalar.

        if n % 100 == 0:
            print(f"iter {n}, loss: {smooth_loss:.4f}")

        for param, dparam, mem in zip(
            [Wxh, Whh, Why, bh, by],
            [dWxh, dWhh, dWhy, dbh, dby],
            [mWxh, mWhh, mWhy, mbh, mby],
        ):  # Shapes in loop order: (400, 79), (400, 100), (79, 100), (400, 1), (79, 1).
            mem += dparam * dparam  # AdaGrad memory; shapes: (400, 79), (400, 100), (79, 100), (400, 1), (79, 1).
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # Parameter shapes: (400, 79), (400, 100), (79, 100), (400, 1), (79, 1).

        p += seq_length  # Integer scalar; carry both states into the next sequence, with gradients limited to 25 time steps.

    print("Training finished.")
