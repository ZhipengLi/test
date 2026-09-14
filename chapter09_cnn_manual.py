"""NumPy CNN: valid convolution -> ReLU -> 2x2 max pool -> dense.

Place beside your existing load_data.py and run: python cnn_fixed.py
Verify without MNIST: python cnn_fixed.py --self-test
Requires NumPy >= 1.20. Inputs: grayscale images; integer class labels.
"""
import argparse
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def prepare_images(images, pixel_scale):
    """Use 255 for raw MNIST [0,255], or 1 for already normalized [0,1]."""
    x = np.asarray(images, dtype=np.float64)
    if x.ndim == 2 and x.shape[1] == 784:
        x = x.reshape(-1, 28, 28)
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    elif x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3 or len(x) == 0:
        raise ValueError('Expected nonempty grayscale images (N,H,W).')
    if pixel_scale not in (1, 255):
        raise ValueError('pixel_scale must be 1 or 255.')
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > pixel_scale:
        raise ValueError('Images must be finite and within [0, pixel_scale].')
    return x / pixel_scale


def prepare_labels(labels, count, num_classes=10):
    y = np.asarray(labels)
    if y.shape != (count,) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('Expected one integer class label per image, shape (N,).')
    if np.any((y < 0) | (y >= num_classes)):
        raise ValueError('Class label out of range.')
    return y.astype(np.int64, copy=False)


def softmax_cross_entropy(logits, label):
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum()
    # Log-sum-exp: remains accurate even when the true probability underflows.
    loss = np.log(exp_logits.sum()) - shifted[label]
    grad = probs.copy()  # Do not corrupt probabilities used by callers.
    grad[label] -= 1
    return float(loss), probs, grad


class CNN:
    def __init__(self, image_shape=(28, 28), num_filters=8,
                 kernel_size=3, num_classes=10, seed=42):
        self.image_shape = tuple(image_shape)
        h, w = self.image_shape
        ch, cw = h - kernel_size + 1, w - kernel_size + 1
        if min(ch, cw) < 2:
            raise ValueError('Image too small for convolution and pooling.')
        rng = np.random.default_rng(seed)
        self.filters = rng.normal(size=(num_filters, kernel_size, kernel_size)) * np.sqrt(2 / kernel_size**2)
        self.conv_bias = np.zeros(num_filters)
        flat_size = num_filters * (ch // 2) * (cw // 2)
        self.dense_W = rng.normal(size=(flat_size, num_classes)) / np.sqrt(flat_size)
        self.dense_b = np.zeros(num_classes)

    def parameters(self):
        return {'filters': self.filters, 'conv_bias': self.conv_bias,
                'dense_W': self.dense_W, 'dense_b': self.dense_b}

    def forward(self, x):
        if x.shape != self.image_shape:
            raise ValueError(f'Expected image shape {self.image_shape}, got {x.shape}.')
        k = self.filters.shape[-1]
        patches = sliding_window_view(x, (k, k))  # (conv_h, conv_w, k, k)
        conv = np.einsum('ijab,fab->fij', patches, self.filters)
        conv += self.conv_bias[:, None, None]
        activated = np.maximum(conv, 0)
        c, h, w = activated.shape
        ph, pw = h // 2, w // 2
        # Each last axis contains one non-overlapping 2x2 pool window.
        windows = activated[:, :2*ph, :2*pw].reshape(c, ph, 2, pw, 2)
        windows = windows.transpose(0, 1, 3, 2, 4).reshape(c, ph, pw, 4)
        winners = windows.argmax(axis=-1)
        pooled = np.take_along_axis(windows, winners[..., None], axis=-1)[..., 0]
        flat = pooled.reshape(-1)
        logits = flat @ self.dense_W + self.dense_b
        return logits, (patches, conv, winners, flat)

    def loss_and_gradients(self, x, label):
        logits, (patches, conv, winners, flat) = self.forward(x)
        loss, probs, dlogits = softmax_cross_entropy(logits, label)
        # All gradients use the SAME weights as the forward pass.
        dflat = self.dense_W @ dlogits
        grads = {'dense_W': np.outer(flat, dlogits), 'dense_b': dlogits}
        dpool = dflat.reshape(winners.shape)
        # Route each pool gradient to exactly one saved maximum (first on ties).
        dwindows = np.zeros((*winners.shape, 4))
        np.put_along_axis(dwindows, winners[..., None], dpool[..., None], axis=-1)
        c, ph, pw = winners.shape
        drelu = np.zeros_like(conv)
        drelu[:, :2*ph, :2*pw] = dwindows.reshape(c, ph, pw, 2, 2).transpose(0, 1, 3, 2, 4).reshape(c, 2*ph, 2*pw)
        dconv = drelu * (conv > 0)
        grads['filters'] = np.einsum('fij,ijab->fab', dconv, patches)
        grads['conv_bias'] = dconv.sum(axis=(1, 2))
        # No input-image gradient is needed: this is the first trainable layer.
        return loss, probs, grads

    def train_step(self, x, label, learning_rate):
        loss, probs, grads = self.loss_and_gradients(x, label)
        for name, param in self.parameters().items():
            param -= learning_rate * grads[name]
        return loss, int(probs.argmax())

    def evaluate(self, images, labels):
        loss, correct = 0., 0
        for x, label in zip(images, labels):
            logits, _ = self.forward(x)
            sample_loss, probs, _ = softmax_cross_entropy(logits, label)
            loss += sample_loss
            correct += int(probs.argmax() == label)
        return loss / len(images), correct / len(images)


def self_test():
    rng = np.random.default_rng(7)
    # Odd convolution dimensions also test the ignored pool border.
    model = CNN((7, 7), num_filters=2, num_classes=3)
    x = rng.normal(size=(7, 7))
    _, _, grads = model.loss_and_gradients(x, 1)
    eps, max_error = 1e-5, 0.
    for name, param in model.parameters().items():
        numerical = np.empty_like(param)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + eps
            plus = softmax_cross_entropy(model.forward(x)[0], 1)[0]
            param[idx] = original - eps
            minus = softmax_cross_entropy(model.forward(x)[0], 1)[0]
            param[idx] = original
            numerical[idx] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(grads[name], numerical, rtol=1e-4, atol=1e-7)
        max_error = max(max_error, float(np.max(np.abs(grads[name] - numerical))))
    loss, probs, grad = softmax_cross_entropy(np.array([1000., -1000.]), 1)
    assert loss == 2000. and np.isfinite(grad).all() and probs[1] == 0
    # A tied positive pool should receive total gradient once, not four times.
    tied = CNN((4, 4), num_filters=1, kernel_size=1, num_classes=2)
    tied.filters[:] = 1
    tied.dense_W[:, 0], tied.dense_W[:, 1] = 0, 1
    _, _, tg = tied.loss_and_gradients(np.ones((4, 4)), 0)
    expected = 4 / (1 + np.exp(-4))
    np.testing.assert_allclose(tg['conv_bias'], [expected])
    # End-to-end optimization on synthetic vertical/horizontal patterns.
    images = np.zeros((12, 8, 8))
    labels = np.arange(12) % 2
    for i, label in enumerate(labels):
        if label == 0:
            images[i, :, 2:4] = 1
        else:
            images[i, 2:4, :] = 1
    images += rng.uniform(0, .05, images.shape)
    learner = CNN((8, 8), num_filters=3, num_classes=2)
    start = learner.evaluate(images, labels)[0]
    old_filters = learner.filters.copy()
    for _ in range(30):
        for i in rng.permutation(len(images)):
            learner.train_step(images[i], labels[i], .03)
    end, accuracy = learner.evaluate(images, labels)
    assert end < start * .2 and accuracy == 1.
    assert not np.array_equal(old_filters, learner.filters)
    print(f'Gradient check passed: max absolute error {max_error:.2e}')
    print('Extreme-logit and tied-pooling checks passed.')
    print(f'Synthetic learning passed: loss {start:.4f} -> {end:.4f}, accuracy {accuracy:.0%}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--pixel-scale', type=int, choices=(1, 255), default=255,
                        help='255 for raw pixels; 1 if your loader already returns [0,1].')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=.01)
    parser.add_argument('--train-limit', type=int, default=1000, help='0 uses all training images.')
    parser.add_argument('--test-limit', type=int, default=200, help='0 uses all test images.')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.epochs <= 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or min(args.train_limit, args.test_limit) < 0:
        parser.error('Use positive epochs/rate and nonnegative limits.')
    from load_data import load_mnist_with_cache
    (train_x, train_y), (test_x, test_y) = load_mnist_with_cache()
    rng = np.random.default_rng(42)
    def subset(images, labels, limit):
        images, labels = np.asarray(images), np.asarray(labels)
        if len(images) != len(labels):
            raise ValueError('Image and label counts differ.')
        indices = rng.permutation(len(images))
        if limit:
            indices = indices[:limit]
        images = prepare_images(images[indices], args.pixel_scale)
        return images, prepare_labels(labels[indices], len(images))
    train_x, train_y = subset(train_x, train_y, args.train_limit)
    test_x, test_y = subset(test_x, test_y, args.test_limit)
    if train_x.shape[1:] != test_x.shape[1:]:
        raise ValueError('Train/test image dimensions differ.')
    model = CNN(train_x.shape[1:])
    print(f'Training on {len(train_x)} images; testing on {len(test_x)} images.')
    for epoch in range(args.epochs):
        for i in rng.permutation(len(train_x)):
            model.train_step(train_x[i], train_y[i], args.learning_rate)
        # Measure with fixed end-of-epoch weights, not changing online weights.
        loss, accuracy = model.evaluate(train_x, train_y)
        print(f'Epoch {epoch + 1}/{args.epochs} - Train loss: {loss:.4f} - Train accuracy: {accuracy:.4f}')
    loss, accuracy = model.evaluate(test_x, test_y)
    print(f'Test loss: {loss:.4f} - Test accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()
