import os
import pickle
from tensorflow.keras.datasets import reuters
import numpy as np 
from tensorflow import keras 
from tensorflow.keras import layers
import matplotlib.pyplot as plt

cache_file = 'reuters_data.pkl'

if os.path.exists(cache_file):
    print("Loading data from cache...")
    with open(cache_file, 'rb') as f:
        (train_data, train_labels), (test_data, test_labels) = pickle.load(f)
else:
    print("Downloading data...")
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    with open(cache_file, 'wb') as f:
        pickle.dump(((train_data, train_labels), (test_data, test_labels)), f)

print('len of train data:', len(train_data))
print('len of test data:', len(test_data))

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]]) 
print(f'decoded:{decoded_newswire}')

def vectorize_sequences(sequences, dimension=10000): 
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.           
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1. 
    return results

y_train = to_one_hot(train_labels)   
y_test = to_one_hot(test_labels)


model = keras.Sequential([
layers.Dense(64, activation="relu"),
layers.Dense(64, activation="relu"),
layers.Dense(46, activation="softmax")
])

model.compile(optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train,
partial_y_train,
epochs=20,
batch_size=512,
validation_data=(x_val, y_val))

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


model = keras.Sequential([
layers.Dense(64, activation="relu"),
layers.Dense(64, activation="relu"),
layers.Dense(46, activation="softmax")
])
model.compile(optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"])
model.fit(x_train, y_train,epochs=9,batch_size=512)
results = model.evaluate(x_test, y_test)

print("test loss, test acc:", results)

predictions = model.predict(x_test)
print('predictions[0].shape:', predictions[0].shape)
print('np.sum(predictions[0]):',np.sum(predictions[0]))
print('np.argmax(predictions[0]:',np.argmax(predictions[0]))