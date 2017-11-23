#!/usr/bin/env python

import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Select TensorFlow as backend for Keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
np.random.seed(1234)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load toy calorimeter dataset
dataset = np.loadtxt("toy_calorimeter.csv", skiprows=1)
inputs = dataset[:,0:-1]
targets = dataset[:,-1].reshape(-1, 1)

print("Loaded calorimeter dataset with {} entries.".format(inputs.shape[0]))
print("Example entry: {}".format(inputs[0]))

# Define model
model = Sequential()

model.add(Dense(
    100, # Number of nodes
    kernel_initializer="glorot_normal", # Initialization
    activation="tanh", # Activation
    input_dim=inputs.shape[1])) # Shape of inputs (only needed for the first layer)

model.add(Dense(
    1,
    kernel_initializer="glorot_uniform",
    activation="linear"))

model.summary()

# Set loss, optimizer and evaluation metrics
model.compile(
        loss="mean_squared_error",
        optimizer=Adam(),
        metrics=[])

# Split dataset in training and testing
inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.5)

# Set up preprocessing
preprocessing_input = StandardScaler()
preprocessing_input.fit(inputs_train)

preprocessing_target = StandardScaler()
preprocessing_target.fit(targets_train)

# Train
history = model.fit(
        preprocessing_input.transform(inputs),
        preprocessing_target.transform(targets),
        batch_size=100,
        epochs=10,
        validation_split=0.25)

# Calculate network response on train dataset
predictions = preprocessing_target.inverse_transform(
        model.predict(preprocessing_input.transform(inputs_test)))

# Save model and preprocessing objects
pickle.dump([preprocessing_input, preprocessing_target],
        open("calorimeter_preprocessing.pickle", "wb"))
model.save("calorimeter_model.h5")

# Plot example distribution of one of the inputs
plt.figure()
plt.xlabel("Energy deposit in first calorimeter layer")
plt.ylabel("Count")
plt.hist(inputs[:,0], bins=50, histtype="step", lw=3)
plt.savefig("calorimeter_layer.png")

# Plot predictions vs ground truth
plt.figure()
plt.xlabel("Energy deposit")
plt.ylabel("Count")
plt.hist(predictions, bins=50, histtype="step", lw=3, alpha=0.8,
        label="Predictions")
plt.hist(targets_test, bins=50, histtype="step", lw=3, alpha=0.8,
        label="Ground truth")
plt.legend()
plt.savefig("calorimeter_predictions.png")

# Plot loss and accuracy over epochs
loss = history.history['loss']
epochs = range(1, 1+len(loss))

plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, loss, 'rs', label="Loss")
plt.savefig("calorimeter_epochs.png")
