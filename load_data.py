# mnist_cache.py
import os
import pickle
from tensorflow.keras.datasets import mnist

def load_mnist_with_cache(cache_file='mnist_data.pkl'):
    """
    Load MNIST data with caching.
    If cache exists, load from it. Otherwise, download and cache it.

    Returns:
        (train_data, train_labels), (test_data, test_labels)
    """
    if os.path.exists(cache_file):
        print("Loading data from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        print("Downloading data...")
        (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
        with open(cache_file, 'wb') as f:
            pickle.dump(((train_data, train_labels), (test_data, test_labels)), f)
        return (train_data, train_labels), (test_data, test_labels)
