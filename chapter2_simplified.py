import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)

# Preprocess
train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255

# Initialize weights and biases
def init_weights(shape):
    return np.random.randn(*shape) * 0.01

W1 = init_weights((784, 512))
b1 = np.zeros((512,))
W2 = init_weights((512, 10))
b2 = np.zeros((10,))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss
def sparse_categorical_crossentropy(predictions, labels):
    batch_size = predictions.shape[0]
    correct_logits = predictions[np.arange(batch_size), labels]
    loss = -np.mean(np.log(correct_logits + 1e-7))
    return loss

# Accuracy
def accuracy(predictions, labels):
    pred_labels = np.argmax(predictions, axis=1)
    return np.mean(pred_labels == labels)

# Training settings
epochs = 20
batch_size = 128
learning_rate = 0.001

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    indices = np.arange(train_images.shape[0])
    np.random.shuffle(indices)
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    for i in range(0, train_images.shape[0], batch_size):
        x_batch = train_images[i:i+batch_size]
        y_batch = train_labels[i:i+batch_size]

        # Forward
        z1 = np.dot(x_batch, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)

        # Loss
        loss = sparse_categorical_crossentropy(a2, y_batch)

        # Backward
        batch_size_actual = x_batch.shape[0]
        dz2 = a2
        dz2[np.arange(batch_size_actual), y_batch] -= 1
        dz2 /= batch_size_actual

        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(z1)

        dW1 = np.dot(x_batch.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # Update parameters
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Evaluate on training data
    z1 = np.dot(train_images, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    train_acc = accuracy(a2, train_labels)
    print(f"Training accuracy: {train_acc:.4f}")

# Evaluate on test data
z1 = np.dot(test_images, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = softmax(z2)
test_acc = accuracy(a2, test_labels)
print(f"test_acc: {test_acc:.4f}")
