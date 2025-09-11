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

def load_imdb_with_cache(cache_file='imdb_data.pkl'):
    """
    Load IMDB data with caching.
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
        texts, labels = [], []
        for split in ["train", "test"]:
            for label in ["pos", "neg"]:
                d = os.path.join("aclImdb", split, label)
                if not os.path.isdir(d):
                    continue
                for fn in os.listdir(d):
                    if not fn.endswith(".txt"):
                        continue
                    with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                        texts.append(f.read())
                    labels.append(1 if label == "pos" else 0)
        with open(cache_file, 'wb') as f:
            pickle.dump((texts, labels), f)
        return texts, labels

