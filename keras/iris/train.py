#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Select Theano as backend for Keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
np.random.seed(1234)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load iris dataset
dataset = load_iris()
inputs = dataset["data"]
target_names = dataset["target_names"]
targets = dataset["target"]
targets_onehot = np_utils.to_categorical(targets, len(target_names))

print("Loaded iris dataset with {} entries.".format(inputs.shape[0]))
print("Example entry: {}".format(inputs[0]))

# Define model
model = Sequential()

model.add(Dense(
    8, # Number of nodes
    kernel_initializer="glorot_normal", # Initialization
    activation="relu", # Activation
    input_dim=inputs.shape[1])) # Shape of inputs (only needed for the first layer)

model.add(Dense(
    len(target_names),
    kernel_initializer="glorot_uniform",
    activation="softmax"))

model.summary()

# Set loss, optimizer and evaluation metrics
model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=0.10),
        metrics=["accuracy",])

# Set up preprocessing
preprocessing = StandardScaler()
preprocessing.fit(inputs)
inputs = preprocessing.transform(inputs)

# Train
history = model.fit(
        inputs,
        targets_onehot,
        batch_size=20,
        epochs=10)

# Calculate accuracy
predictions = model.predict(inputs)
predictions_argmax = np.argmax(predictions, axis=1)
accuracy = np.sum(predictions_argmax==targets)/float(inputs.shape[0])
print("Accuracy on full dataset: {}".format(accuracy))

# Save model and preprocessing object
model.save("weights.h5")

# Plot loss and accuracy over epochs
loss = history.history['loss']
acc = history.history['acc']
epochs = range(1, 1+len(loss))

fig, ax1 = plt.subplots()

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.plot(epochs, loss, 'rs', label="Loss")

ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy")
ax2.plot(epochs, acc, 'bo', label="Accuracy")

plt.savefig("iris_epochs.png")
