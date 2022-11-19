# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:47:26 2022

@author: Yue
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import Dataset
from neural_networks import MLP

DATA_PATH = "../data/"
MODEL_PATH = "../model/"

broad2features_file = "broad2features.json"
broad2fingerprints_file = "broad2fingerprints.json"
features_file = "features.npy"


with open(DATA_PATH + broad2features_file, "r") as infile1:
    broad2features = json.load(infile1)

with open(DATA_PATH + broad2fingerprints_file, "r") as infile1:
    broad2fingerprint = json.load(infile1)

features = np.load(DATA_PATH + features_file)

#%%
X = []
y = []

for broad_id, fingerprint in broad2fingerprint.items():
    # turn fingerprint (bit string) into a numpy array
    if fingerprint is None:
        continue
    row_indices = broad2features[broad_id]
    for row_ind in row_indices:
        # cell_area = features[row_ind, 0]
        # cytoplasm_area = features[row_ind, 596]
        nuclei_area = features[row_ind, 1178]
        y.append(nuclei_area)
        X.append(np.array(list(fingerprint)).astype(float))

#%%
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

#%%

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
EARLY_STOPPING = 10
EPOCHS = 20

train_set = Dataset(X_train, y_train)
test_set = Dataset(X_test, y_test)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

# Initialize the MLP
mlp = MLP(input_size=2048, hidden_layer=512).to(device)

# Define the loss function and optimizer
loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
prev_loss = 0.0
stale = 0.0
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    print(f"Starting epoch {epoch+1}")
    train_loss = 0.0
    for data in train_loader:

        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.reshape((targets.shape[0], 1))
        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        train_loss += loss.item() * len(data[0])

    train_loss /= len(train_set)
    print(f"training loss: {train_loss}")
    train_losses.append(train_loss)

    if train_loss >= 0.99 * prev_loss:
        stale += 1
        prev_loss = np.mean(train_losses[-EARLY_STOPPING:])
    else:
        stale = 0

    if stale == EARLY_STOPPING:
        print(f"train loss not imroving for {EARLY_STOPPING} epochs.")
        print("training stopped")
        break

    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            mol_vec, labels = data
            mol_vec, labels = mol_vec.to(device), labels.to(device)
            labels = labels.reshape((labels.shape[0], 1))
            y_pred = mlp(mol_vec)
            batch_loss = loss_function(y_pred, labels)
            test_loss += batch_loss.item() * len(data[0])

        test_loss /= len(test_set)
        test_losses.append(test_loss)
        print(f"test loss: {test_loss}")
        print("")


plt.figure(figsize=(10, 5))
plt.title("Training and Test Loss")
plt.plot(test_losses, label="test")
plt.plot(train_losses, label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
