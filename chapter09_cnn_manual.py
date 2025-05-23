import numpy as np
from tensorflow.keras.datasets import mnist
import gzip
import os
import urllib.request

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)


# Function to download and load MNIST dataset
# def load_mnist():
#     base_url = 'http://yann.lecun.com/exdb/mnist/'
#     files = {
#         'train_images': 'train-images-idx3-ubyte.gz',
#         'train_labels': 'train-labels-idx1-ubyte.gz',
#         'test_images': 't10k-images-idx3-ubyte.gz',
#         'test_labels': 't10k-labels-idx1-ubyte.gz',
#     }
#     data = {}
#     for key, filename in files.items():
#         if not os.path.exists(filename):
#             print(f'Downloading {filename}...')
#             urllib.request.urlretrieve(base_url + filename, filename)
#         with gzip.open(filename, 'rb') as f:
#             if 'images' in key:
#                 data[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28) / 255.0
#             else:
#                 data[key] = np.frombuffer(f.read(), np.uint8, offset=8)
#     return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']

# Load data
# train_images, train_labels, test_images, test_labels = load_mnist()

# Use a subset for faster training (optional)
train_images = train_images[:1000]
train_labels = train_labels[:1000]
test_images = test_images[:200]
test_labels = test_labels[:200]

# Initialize weights
np.random.seed(42)
conv_filters = np.random.randn(8, 3, 3) * 0.1
conv_bias = np.zeros(8)
dense_W = np.random.randn(1352, 10) * 0.1
dense_b = np.zeros(10)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# Loss function
def cross_entropy(pred, label):
    return -np.log(pred[label] + 1e-7)

# Forward pass functions
def conv2d_forward(input, filters, bias):
    H, W = input.shape
    num_filters, FH, FW = filters.shape
    out_h = H - FH + 1
    out_w = W - FW + 1
    output = np.zeros((num_filters, out_h, out_w))
    for n in range(num_filters):
        for i in range(out_h):
            for j in range(out_w):
                region = input[i:i+FH, j:j+FW]
                output[n, i, j] = np.sum(region * filters[n]) + bias[n]
    return output

def maxpool2d(input):
    C, H, W = input.shape
    out_h = H // 2
    out_w = W // 2
    output = np.zeros((C, out_h, out_w))
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                region = input[c, i*2:i*2+2, j*2:j*2+2]
                output[c, i, j] = np.max(region)
    return output

def flatten(x):
    return x.flatten()

def dense_forward(x, W, b):
    return np.dot(x, W) + b

# Training parameters
epochs = 5
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    correct = 0
    total_loss = 0
    for i in range(len(train_images)):
        x = train_images[i]
        label = train_labels[i]

        # Forward pass
        x_conv = conv2d_forward(x, conv_filters, conv_bias)
        x_relu = relu(x_conv)
        x_pool = maxpool2d(x_relu)
        x_flat = flatten(x_pool)
        x_dense = dense_forward(x_flat, dense_W, dense_b)
        x_prob = softmax(x_dense)

        # Loss and accuracy
        loss = cross_entropy(x_prob, label)
        total_loss += loss
        pred = np.argmax(x_prob)
        if pred == label:
            correct += 1

        # Backward pass
        # Gradient of loss w.r.t. dense layer output
        grad_output = x_prob
        grad_output[label] -= 1

        # Gradients for dense layer
        grad_W = np.outer(x_flat, grad_output)
        grad_b = grad_output

        # Update dense layer weights
        dense_W -= learning_rate * grad_W
        dense_b -= learning_rate * grad_b

        # Note: Backpropagation through convolutional and pooling layers is not implemented in this example

    accuracy = correct / len(train_images)
    avg_loss = total_loss / len(train_images)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

# Evaluation on test set
correct = 0
for i in range(len(test_images)):
    x = test_images[i]
    label = test_labels[i]
    x_conv = conv2d_forward(x, conv_filters, conv_bias)
    x_relu = relu(x_conv)
    x_pool = maxpool2d(x_relu)
    x_flat = flatten(x_pool)
    x_dense = dense_forward(x_flat, dense_W, dense_b)
    x_prob = softmax(x_dense)
    pred = np.argmax(x_prob)
    if pred == label:
        correct += 1

test_accuracy = correct / len(test_images)
print(f"Test Accuracy: {test_accuracy:.4f}")
