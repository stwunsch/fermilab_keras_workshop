#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pickle
import h5py

# Select TensorFlow as backend for Keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
np.random.seed(1234)

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.models import load_model

# Load HIGGS dataset
file_ = h5py.File("HIGGS.h5")
inputs = np.array(file_["features"])
targets = np.array(file_["targets"])

# Load models
model_shallow = load_model("HIGGS_shallow.h5")
model_deep = load_model("HIGGS_deep.h5")

# Load preprocessing
preprocessing_input = pickle.load(open("HIGGS_preprocessing.pickle", "rb"))

# Split dataset in training and testing (with same seed as done in the training script!)
inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.90, random_state=1234, shuffle=True)

# Get prediction on subset of test dataset
num_events = 10000
predictions_shallow = model_shallow.predict(
        preprocessing_input.transform(inputs_test[:num_events]))
predictions_deep = model_deep.predict(
        preprocessing_input.transform(inputs_test[:num_events]))

# Compare to ground truth and create ROC plot
fpr_shallow, tpr_shallow, _ = roc_curve(targets_test[:num_events], predictions_shallow)
fpr_deep, tpr_deep, _ = roc_curve(targets_test[:num_events], predictions_deep)

auc_shallow = auc(fpr_shallow, tpr_shallow)
auc_deep = auc(fpr_deep, tpr_deep)

# Plot ROC
plt.figure(figsize=(4, 4))
plt.plot(tpr_deep, 1.0-fpr_deep, lw=3, alpha=0.8,
        label="Deep (AUC={:.2f})".format(auc_deep))
plt.plot(tpr_shallow, 1.0-fpr_shallow, lw=3, alpha=0.8,
        label="Shallow (AUC={:.2f})".format(auc_shallow))
plt.xlabel("Signal efficiency")
plt.ylabel("Background rejection")
plt.legend(loc=3)
plt.xlim((0.0, 1.0))
plt.ylim((0.0, 1.0))
plt.savefig("HIGGS_roc.png", bbox_inches="tight")
