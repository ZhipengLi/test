# https://ai.stanford.edu/~amaas/data/sentiment/
# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

import os, pathlib, shutil, random
from tensorflow import keras
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val" 
train_dir = base_dir / "train" 
for category in ("neg", "pos"):
    os.makedirs(val_dir / category)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)   
    num_val_samples = int(0.2 * len(files))  
    val_files = files[-num_val_samples:]  
for fname in val_files:    
    shutil.move(train_dir / category / fname, val_dir / category / fname)

batch_size = 32 
train_ds = keras.utils.text_dataset_from_directory(  
    "aclImdb/train", batch_size=batch_size
)
val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)
test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

text_vectorization = TextVectorization(
    max_tokens=20000,         
    output_mode="multi_hot",         
)

text_only_train_ds = train_ds.map(lambda x, y: x)   
text_vectorization.adapt(text_only_train_ds)   
binary_1gram_train_ds = train_ds.map(          
lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)                      
binary_1gram_val_ds = val_ds.map(              
lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)                      
binary_1gram_test_ds = test_ds.map(            
lambda x, y: (text_vectorization(x), y), num_parallel_calls=4) 

def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",
          loss="binary_crossentropy",
          metrics=["accuracy"])
    return model

model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_1gram.keras",
    save_best_only=True)
 ]

model.fit(binary_1gram_train_ds.cache(),  
    validation_data=binary_1gram_val_ds.cache(),  
    epochs=10,
    callbacks=callbacks)
model = keras.models.load_model("binary_1gram.keras") 
print(f"Test acc: {model.evaluate(binary_1gram_test_ds)[1]:.3f}")
