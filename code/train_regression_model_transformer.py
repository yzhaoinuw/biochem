# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 23:58:22 2022

@author: Yue
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Dataset
from neural_networks import MLP, DMLP

DATA_PATH = "../data/"
WRITE_LOC = "../data/"
MODEL_PATH = "../model/"
WRITE_LOC = "../results/"

broad2smiles_file = "broad2smiles.json"
embeddings_file = "biochem_smiles.npz"
broad2features_file = "broad2features.json"
features_file = "features.npy"

with open(DATA_PATH + broad2features_file, "r") as infile1:
    broad2features = json.load(infile1)

with open(DATA_PATH + broad2smiles_file, "r") as infile1:
    broad2smiles = json.load(infile1)

embeddings = np.load(MODEL_PATH + embeddings_file)
features = np.load(DATA_PATH + features_file)

#%%

LEAST_SAMPLE_COUNT = 8
TEST_SIZE = 0.5
EARLY_STOPPING = 10
EPOCHS = 50
SAVE_MODEL = False
MODEL_NAME = f"DMLPRegressor_mol_emb_epoch{EPOCHS}_ts{TEST_SIZE}"

X_train = []
X_test = []
y_train = []
y_test = []

for broad_id, smiles in broad2smiles.items():
    # turn embeddings (saved in json as list) into a numpy array
    embedding = embeddings.get(smiles, None)
    if embedding is None:
        continue
    embedding = np.mean(embedding, axis=0)
    row_indices = broad2features[broad_id]

    l = min(len(row_indices), LEAST_SAMPLE_COUNT)
    np.random.shuffle(row_indices)
    row_indices = row_indices[:l]
    test_inds, train_inds = (
        row_indices[: int(l * TEST_SIZE)],
        row_indices[int(l * TEST_SIZE) :],
    )

    if np.random.uniform() > TEST_SIZE:
        for ind in train_inds:
            # cell_area = features[row_ind, 0]
            # cytoplasm_area = features[row_ind, 596]
            nuclei_area = features[ind, 1178]
            y_train.append(nuclei_area)
            X_train.append(embedding)
    for ind in test_inds:
        # cell_area = features[row_ind, 0]
        # cytoplasm_area = features[row_ind, 596]
        nuclei_area = features[ind, 1178]
        y_test.append(nuclei_area)
        X_test.append(embedding)


#%%
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

#%%

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

train_set = Dataset(X_train, y_train)
test_set = Dataset(X_test, y_test)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

# Initialize the MLP
mlp = DMLP(input_size=512, hidden_layer1=1024, hidden_layer2=256).to(device)

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
prev_loss = 0.0
stale = 0.0
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    print(f"Starting epoch {epoch+1}")

    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            mol_emb, labels = data
            mol_emb, labels = mol_emb.to(device), labels.to(device)
            labels = labels.reshape((labels.shape[0], 1))
            y_pred = mlp(mol_emb)
            batch_loss = torch.sqrt(loss_function(y_pred, labels))
            test_loss += batch_loss.item() * len(data[0])

        test_loss /= len(test_set)
        test_losses.append(test_loss)

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
        loss = torch.sqrt(loss_function(outputs, targets))

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        train_loss += loss.item() * len(data[0])

    train_loss /= len(train_set)
    train_losses.append(train_loss)
    print(f"training loss: {train_loss}")
    print(f"test loss: {test_loss}")
    print("")

    with open(WRITE_LOC + MODEL_NAME + ".txt", "w") as infile1:
        epoch_message = [
            f"Epoch {epoch}",
            f"Training Loss: {train_loss:.3f}",
            f"Test Loss: {test_loss:.3f}",
        ]
        infile1.write("\n".join(epoch_message) + "\n" * 2)

    if test_loss >= 0.99 * prev_loss:
        stale += 1
        prev_loss = np.mean(test_losses[-EARLY_STOPPING:])
    else:
        stale = 0

    if stale == EARLY_STOPPING:
        print(f"test loss not imroving for {EARLY_STOPPING} epochs.")
        print("training stopped")
        break

if SAVE_MODEL:
    torch.save(mlp, MODEL_PATH + MODEL_NAME)

#%%
plt.figure(figsize=(10, 5))
plt.title("Training and Test Loss")
x_axis = np.arange(1, EPOCHS, step=1)
plt.plot(x_axis, test_losses[1:], label="test")
plt.plot(x_axis, train_losses[1:], label="train")
plt.xticks(x_axis)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(axis="y")
plt.savefig(WRITE_LOC + MODEL_NAME + ".png")
plt.show()
