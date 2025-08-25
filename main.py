import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import struct
from time_log import time_log_module as tlm
import os, time

model_path = "model.keras"
model_exist = os.path.exists(model_path)

# ===== Functions =======
def rd(path): # to read .idx1-ubyte/.idx3-ubyte files
    with open(path, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape=tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
# =======================

# Building the model 
if model_exist:
    print(f"{tlm()} Loading model...")
    model = load_model(model_path)
    t = input(f"{tlm()} An existing model was found un succesfully loaded.\nDo you want to train it or no [y/n]?\n>>> ")
    train = t == "y" or t == "Y"
else:
    print(f"{tlm()} Building model...")
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

# Loading dataset (MNIST)
print(f"{tlm()} Loading dataset")
X_train = rd('data/train-images.idx3-ubyte')
y_train = rd('data/train-labels.idx1-ubyte')
X_test = rd('data/t10k-images.idx3-ubyte')
y_test = rd('data/t10k-labels.idx1-ubyte')

# Normalize images
X_train = X_train.astype('float32') / 255.0 
X_test = X_test.astype('float32') / 255.0

# Compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Trainning the model
if train:
    print(f"{tlm()} Starting the model training...")
    start = time.time()
    train = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
    end = time.time()
    temps_pour_train = end-start # secondes
    print(f"{tlm()} Model successfully trainned in {temps_pour_train} seconds.")
    model.save(model_path)
model.summary()

# Affichage des resultats du train
history = train.history
print(f"{tlm()} Final training accuracy: {history['accuracy'][-1]:.4f}")
print(f"{tlm()} Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
print(f"{tlm()} Final training loss: {history['loss'][-1]:.4f}")
print(f"{tlm()} Final validation loss: {history['val_loss'][-1]:.4f}")

# Test du model sur les donnees de test
print(f"{tlm()} Evaluating the model on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"{tlm()} Test loss: {test_loss:.4f}")
print(f"{tlm()} Test accuracy: {test_acc:.4f}")

# afficher quelques predictions
print(f"{tlm()} Making some predictions on test data...")
predictions = model.predict(X_test[:20])
for i in range(20):
    print(f"Image {i+1}: Predicted label: {np.argmax(predictions[i])}, True label: {y_test[i]}")