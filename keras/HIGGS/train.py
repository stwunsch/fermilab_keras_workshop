#!/usr/bin/env python

import pickle
import h5py

# Select TensorFlow as backend for Keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
np.random.seed(1234)

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load HIGGS dataset
file_ = h5py.File("HIGGS.h5")
inputs = np.array(file_["features"])
targets = np.array(file_["targets"])

# Shallow neural network
model_shallow = Sequential()
model_shallow.add(Dense(1000, kernel_initializer="glorot_normal", activation="tanh",
    input_dim=inputs.shape[1]))
model_shallow.add(Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid"))

# "Deep" neural network
model_deep = Sequential()
model_deep.add(Dense(300, kernel_initializer="glorot_normal", activation="relu",
    input_dim=inputs.shape[1]))
model_deep.add(Dense(300, kernel_initializer="glorot_normal", activation="relu"))
model_deep.add(Dense(300, kernel_initializer="glorot_normal", activation="relu"))
model_deep.add(Dense(300, kernel_initializer="glorot_normal", activation="relu"))
model_deep.add(Dense(300, kernel_initializer="glorot_normal", activation="relu"))
model_deep.add(Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid"))

# Set loss, optimizer and evaluation metrics
for model in [model_shallow, model_deep]:
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy"])

# Split dataset in training and testing
inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.90, random_state=1234, shuffle=True)

# Set up preprocessing
preprocessing_input = StandardScaler()
preprocessing_input.fit(inputs_train)

# Train
for model in [model_shallow, model_deep]:
    model.fit(
            preprocessing_input.transform(inputs_train),
            targets_train,
            batch_size=100,
            epochs=10,
            validation_split=0.25)

# Save preprocessing and models
pickle.dump(preprocessing_input, open("HIGGS_preprocessing.pickle", "wb"))
for model, name in zip([model_shallow, model_deep], ["HIGGS_shallow.h5", "HIGGS_deep.h5"]):
    model.save(name)
