# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 23:44:28 2022

@author: Yue
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Dataset
from evaluate import calculate_rank
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
VALID_SIZE = 0.3
TEST_SIZE = 0.2
EARLY_STOPPING = 5
EPOCHS = 100
SAVE_MODEL = True
MODEL_NAME = f"multivariate_DMLPRegressor_mol_emb_epoch{EPOCHS}_ts{TEST_SIZE}"

X_train = []
X_test = []
X_valid = []
y_valid = []
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
    test_inds, valid_inds, train_inds = (
        row_indices[: round(l * TEST_SIZE)],
        row_indices[round(l * TEST_SIZE): round(l * (TEST_SIZE+VALID_SIZE))],
        row_indices[round(l * (TEST_SIZE+VALID_SIZE)) :],
    )

    if np.random.uniform() > TEST_SIZE+VALID_SIZE:
        # cell_area = features[train_inds, 0]
        # cytoplasm_area = features[train_inds, 596]
        # nuclei_area = features[train_inds, 1178]
        y_train.extend(features[train_inds])
        X_train.extend([embedding for i in range(len(train_inds))])
    
    #nuclei_area = features[valid_inds, 1178]
    y_valid.extend(features[valid_inds])
    X_valid.extend(embedding for i in range(len(valid_inds)))
    
    #nuclei_area = features[test_inds, 1178]
    y_test.extend(features[test_inds])
    X_test.extend(embedding for i in range(len(test_inds)))

#%%
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_valid, y_test = scaler.transform(y_valid), scaler.transform(y_test)

X_train, X_valid, X_test = np.array(X_train), np.array(X_valid), np.array(X_test)
y_train, y_valid, y_test = np.array(y_train), np.array(y_valid), np.array(y_test)

#%%

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

X_train = torch.from_numpy(X_train).float()
X_valid = torch.from_numpy(X_valid).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_valid = torch.from_numpy(y_valid).float()
y_test = torch.from_numpy(y_test).float()

train_set = Dataset(X_train, y_train)
valid_set = Dataset(X_valid, y_valid)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=True)

# Initialize the MLP
mlp = DMLP(input_size=512, hidden_layer1=1024, hidden_layer2=256, output_size=1783).to(device)

# Define the loss function and optimizer
loss_function = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
prev_loss = 0.0
stale = 0.0
train_losses = []
valid_losses = []

for epoch in range(EPOCHS):
    print(f"Starting epoch {epoch+1}")

    valid_loss = 0.0
    with torch.no_grad():
        for data in valid_loader:
            mol_vec, labels = data
            mol_vec, labels = mol_vec.to(device), labels.to(device)
            y_pred = mlp(mol_vec)
            batch_loss = loss_function(y_pred, labels)
            batch_loss = torch.sum(batch_loss)
            valid_loss += batch_loss.item()

        valid_loss /= len(valid_set)
        valid_losses.append(valid_loss)

    train_loss = 0.0
    for data in train_loader:

        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)
        loss = torch.sum(loss)
        
        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()
        
        train_loss += loss.item()

    train_loss /= len(train_set)
    train_losses.append(train_loss)
    print(f"training loss: {train_loss}")
    print(f"validation loss: {valid_loss}")
    print("")

    if SAVE_MODEL:
        with open(WRITE_LOC + MODEL_NAME + ".txt", "a") as infile1:
            epoch_message = [
                f"Epoch {epoch}",
                f"Training Loss: {train_loss:.3f}",
                f"Validation Loss: {valid_loss:.3f}",
            ]
            infile1.write("\n".join(epoch_message) + "\n" * 2)

    if valid_loss >= 0.99 * prev_loss:
        stale += 1
        prev_loss = np.mean(valid_losses[-EARLY_STOPPING:])
    else:
        stale = 0

    if stale == EARLY_STOPPING:
        print(f"validation loss not imroving for {EARLY_STOPPING} epochs.")
        print("training stopped")
        break

if SAVE_MODEL:
    torch.save(mlp, MODEL_PATH + MODEL_NAME)

#%%
plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
x_axis = np.arange(1, EPOCHS, step=1)
plt.plot(x_axis, valid_losses[1:], label="validation")
plt.plot(x_axis, train_losses[1:], label="train")
plt.xticks(x_axis)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(axis="y")
if SAVE_MODEL:
    plt.savefig(WRITE_LOC + MODEL_NAME + ".png")
plt.show()
