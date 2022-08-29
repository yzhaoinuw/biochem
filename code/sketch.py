# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:15:01 2022

@author: Yue
"""

import math
import json
import numpy as np
import matplotlib.pyplot as plt

import joblib
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Dataset
from neural_networks import MLP, DMLP

DATA_PATH = "../data/"
MODEL_PATH = "../model/"
WRITE_LOC = "../results/"
MIN_SAMPLE_COUNT = 4

data_split_file = (
    f"biochem_dataset_train_size0.6_min_sample_count{MIN_SAMPLE_COUNT}.json"
)
broad2smiles_file = "broad2smiles.json"
embeddings_file = "biochem_smiles.npz"
features_file = "features.npy"
broad2vec_file = "broad2vec.json"
randvec_file = "broad2randvec.json"
broad2fingerprints_file = "broad2fingerprints.json"

with open(DATA_PATH + broad2smiles_file, "r") as infile1:
    broad2smiles = json.load(infile1)

features = np.load(DATA_PATH + features_file)

with open(DATA_PATH + data_split_file, "r") as infile1:
    data_split = json.load(infile1)

# randomvec
with open(DATA_PATH + randvec_file, "r") as infile1:
    embeddings = json.load(infile1)

# mol2vec
# with open(DATA_PATH + broad2vec_file, "r") as infile1:
#    embeddings = json.load(infile1)

# fingerprints
# with open(DATA_PATH + broad2fingerprints_file, "r") as infile1:
#    embeddings = json.load(infile1)

# transformers
# embeddings = np.load(MODEL_PATH + embeddings_file)

#%%

X_train = []
X_valid_is = []
X_valid_os = []
y_train = []
y_valid_is = []
y_valid_os = []

for broad_id, row_ind in data_split["train"]:
    smiles = broad2smiles[broad_id]
    # embedding = embeddings.get(smiles, None) # transformer embeddings
    embedding = embeddings.get(broad_id, None)
    if embedding is None:
        continue
    # embedding = np.mean(embedding, axis=0) # transformer embeddings
    # embedding = np.array(list(embedding)).astype(float) # fingerprints
    X_train.append(embedding)
    y_train.append(features[row_ind])

for broad_id, row_ind in data_split["validation_is"]:
    smiles = broad2smiles[broad_id]
    # embedding = embeddings.get(smiles, None)
    embedding = embeddings.get(broad_id, None)
    if embedding is None:
        continue
    # embedding = np.mean(embedding, axis=0)
    # embedding = np.array(list(embedding)).astype(float)
    X_valid_is.append(embedding)
    y_valid_is.append(features[row_ind])

for broad_id, row_ind in data_split["validation_os"]:
    smiles = broad2smiles[broad_id]
    # embedding = embeddings.get(smiles, None)
    embedding = embeddings.get(broad_id, None)
    if embedding is None:
        continue
    # embedding = np.mean(embedding, axis=0)
    # embedding = np.array(list(embedding)).astype(float)
    X_valid_os.append(embedding)
    y_valid_os.append(features[row_ind])

#%%
X_train = np.array(X_train)
X_valid_is, X_valid_os = np.array(X_valid_is), np.array(X_valid_os)

y_train = np.array(y_train)
y_valid_is, y_valid_os = np.array(y_valid_is), np.array(y_valid_os)

scaler_filename = f"standard_scaler_min_sample_count{MIN_SAMPLE_COUNT}.save"
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_valid_is, y_valid_os = scaler.transform(y_valid_is), scaler.transform(y_valid_os)
joblib.dump(scaler, MODEL_PATH + scaler_filename)

#%%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

X_train = torch.from_numpy(X_train).float()
X_valid_is, X_valid_os = (
    torch.from_numpy(X_valid_is).float(),
    torch.from_numpy(X_valid_os).float(),
)
y_train = torch.from_numpy(y_train).float()
y_valid_is, y_valid_os = (
    torch.from_numpy(y_valid_is).float(),
    torch.from_numpy(y_valid_os).float(),
)

train_set = Dataset(X_train, y_train)
valid_set_is, valid_set_os = Dataset(X_valid_is, y_valid_is), Dataset(
    X_valid_os, y_valid_os
)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader_is, valid_loader_os = DataLoader(
    valid_set_is, batch_size=32, shuffle=True
), DataLoader(valid_set_os, batch_size=32, shuffle=True)

# Initialize the MLP
mlp = DMLP(input_size=2048, hidden_layer1=1024, hidden_layer2=512, output_size=1783).to(
    device
)

EPOCHS = 100
SAVE_MODEL = True
WEIGHT_DECAY = 0.1
MODEL_NAME = f"DMLPRegressor_randvec_weight_decay{WEIGHT_DECAY}_min_sample_count{MIN_SAMPLE_COUNT}"

# Define the loss function and optimizer
loss_function = nn.MSELoss(reduction="none")
L1_loss_function = nn.L1Loss(reduction="none")

optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
prev_loss = 0.0
train_losses = []
valid_losses_is = []
valid_losses_os = []
lowest_loss = math.inf
for epoch in range(EPOCHS):
    print(f"Starting epoch {epoch+1}")

    valid_loss_is = 0.0
    valid_loss_os = 0.0
    with torch.no_grad():
        for data in valid_loader_is:
            mol_vec, labels = data
            mol_vec, labels = mol_vec.to(device), labels.to(device)
            y_pred = mlp(mol_vec)
            batch_loss = L1_loss_function(y_pred, labels)
            batch_loss = torch.sum(batch_loss)
            valid_loss_is += batch_loss.item()

        valid_loss_is /= len(valid_set_is)
        valid_losses_is.append(valid_loss_is)

        for data in valid_loader_os:
            mol_vec, labels = data
            mol_vec, labels = mol_vec.to(device), labels.to(device)
            y_pred = mlp(mol_vec)
            batch_loss = L1_loss_function(y_pred, labels)
            batch_loss = torch.sum(batch_loss)
            valid_loss_os += batch_loss.item()

        valid_loss_os /= len(valid_set_os)
        valid_losses_os.append(valid_loss_os)
        if valid_loss_os < lowest_loss:
            lowest_loss = valid_loss_os
            best_model = {
                "epoch": epoch,
                "model_state_dict": mlp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": lowest_loss,
            }

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

        L1_loss = L1_loss_function(outputs, targets)
        L1_loss = torch.sum(L1_loss)
        train_loss += L1_loss.item()

    train_loss /= len(train_set)
    train_losses.append(train_loss)
    print(f"training loss: {train_loss}")
    print(f"in sample validation loss: {valid_loss_is}")
    print(f"out of sample validation loss: {valid_loss_os}")
    print("")

    if SAVE_MODEL:
        with open(WRITE_LOC + MODEL_NAME + ".txt", "a") as infile1:
            epoch_message = [
                f"Epoch {epoch}",
                f"Training Loss: {train_loss:.3f}",
                f"In sample validation Loss: {valid_loss_is:.3f}",
                f"Out of sample validation Loss: {valid_loss_os:.3f}",
            ]
            infile1.write("\n".join(epoch_message) + "\n" * 2)

if SAVE_MODEL:
    with open(WRITE_LOC + MODEL_NAME + ".txt", "a") as infile1:
        summary = [
            f"Best epoch: {best_model['epoch']}",
            f"Best loss: {best_model['loss']}",
        ]
        infile1.write("\n".join(summary) + "\n" * 2)

if SAVE_MODEL:
    torch.save(best_model, MODEL_PATH + MODEL_NAME)

#%%
plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
x_axis = np.arange(1, EPOCHS, step=1)
plt.plot(x_axis, valid_losses_is[1:], label="validation_is")
plt.plot(x_axis, valid_losses_os[1:], label="validation_os")
plt.plot(x_axis, train_losses[1:], label="train")
plt.xticks(x_axis)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(axis="y")
if SAVE_MODEL:
    plt.savefig(WRITE_LOC + MODEL_NAME + ".png")
plt.show()
