import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from load_data import load_mnist_with_cache

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []
    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))
    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses,
            label="Training loss for each batch")
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_at_epoch_{epoch}")
        self.per_batch_losses = []

def get_mnist_model():            
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model
(images, labels), (test_images, test_labels) = load_mnist_with_cache() # mnist.load_data()    
images = images.reshape((60000, 28 * 28)).astype("float32") / 255 
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255 
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]
model = get_mnist_model()
model.compile(optimizer="rmsprop",  
    loss="sparse_categorical_crossentropy",  
    metrics=["accuracy"])                    
#model.fit(train_images, train_labels,epochs=3,validation_data=(val_images, val_labels))  
model.fit(train_images, train_labels,
    epochs=10,
    callbacks=[LossHistory()],
    validation_data=(val_images, val_labels))

test_metrics = model.evaluate(test_images, test_labels)   
predictions = model.predict(test_images) 
print("test_metrics", test_metrics)
print("predictions", predictions)