# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:15:01 2022

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

DATA_PATH = '../data/'
data_split_file = 'biochem_dataset_train_size0.6_min_sample_count4.json'
MODEL_PATH = "../model/"
WRITE_LOC = "../results/"

broad2smiles_file = "broad2smiles.json"
embeddings_file = "biochem_smiles.npz"
features_file = "features.npy"
broad2vec_file = "broad2vec.json"
broad2fingerprints_file = "broad2fingerprints.json"

with open(DATA_PATH + broad2smiles_file, "r") as infile1:
    broad2smiles = json.load(infile1)
    
features = np.load(DATA_PATH + features_file)

with open(DATA_PATH+data_split_file, 'r') as infile1:
    data_split = json.load(infile1)

# mol2vec
#with open(DATA_PATH + broad2vec_file, "r") as infile1:
#    embeddings = json.load(infile1)
  
# fingerprints  
with open(DATA_PATH + broad2fingerprints_file, "r") as infile1:
    embeddings = json.load(infile1)
    
# transformers
#embeddings = np.load(MODEL_PATH + embeddings_file)

#%%

X_train = []
X_valid_is = []
X_valid_os = []
X_test_is = []
X_test_os = []
y_train = []
y_valid_is = []
y_valid_os = []
y_test_is = []
y_test_os = []

for broad_id, row_ind in data_split['train']:
    smiles = broad2smiles[broad_id]
    #embedding = embeddings.get(smiles, None)
    embedding = embeddings.get(broad_id, None)
    if embedding is None:
        continue
    #embedding = np.mean(embedding, axis=0)
    embedding = np.array(list(embedding)).astype(float)
    X_train.append(embedding)
    y_train.append(features[row_ind])
    
for broad_id, row_ind in data_split['validation_is']:
    smiles = broad2smiles[broad_id]
    #embedding = embeddings.get(smiles, None)
    embedding = embeddings.get(broad_id, None)
    if embedding is None:
        continue
    #embedding = np.mean(embedding, axis=0)
    embedding = np.array(list(embedding)).astype(float)
    X_valid_is.append(embedding)
    y_valid_is.append(features[row_ind])
    
for broad_id, row_ind in data_split['validation_os']:
    smiles = broad2smiles[broad_id]
    #embedding = embeddings.get(smiles, None)
    embedding = embeddings.get(broad_id, None)
    if embedding is None:
        continue
    #embedding = np.mean(embedding, axis=0)
    embedding = np.array(list(embedding)).astype(float)
    X_valid_os.append(embedding)
    y_valid_os.append(features[row_ind])
    
for broad_id, row_ind in data_split['test_is']:
    smiles = broad2smiles[broad_id]
    #embedding = embeddings.get(smiles, None)
    embedding = embeddings.get(broad_id, None)
    
    if embedding is None:
        continue
    #embedding = np.mean(embedding, axis=0)
    embedding = np.array(list(embedding)).astype(float)
    X_test_is.append(embedding)
    y_test_is.append(features[row_ind])
    
for broad_id, row_ind in data_split['test_os']:
    smiles = broad2smiles[broad_id]
    #embedding = embeddings.get(smiles, None)
    embedding = embeddings.get(broad_id, None)
    if embedding is None:
        continue
    #embedding = np.mean(embedding, axis=0)
    embedding = np.array(list(embedding)).astype(float)
    X_test_os.append(embedding)
    y_test_os.append(features[row_ind])
    
#%%
X_train = np.array(X_train)
X_valid_is, X_valid_os = np.array(X_valid_is), np.array(X_valid_os)
X_test_is, X_test_os = np.array(X_test_is), np.array(X_test_os)

y_train = np.array(y_train)
y_valid_is, y_valid_os = np.array(y_valid_is), np.array(y_valid_os) 
y_test_is, y_test_os = np.array(y_test_is), np.array(y_test_os)

scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_valid_is, y_valid_os = scaler.transform(y_valid_is), scaler.transform(y_valid_os)
y_test_is, y_test_os = scaler.transform(y_test_is), scaler.transform(y_test_os)
#%%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

X_train = torch.from_numpy(X_train).float()
X_valid_is, X_valid_os = torch.from_numpy(X_valid_is).float(), torch.from_numpy(X_valid_os).float()
X_test_is, X_test_os = torch.from_numpy(X_test_is).float(), torch.from_numpy(X_test_os).float()
y_train = torch.from_numpy(y_train).float()
y_valid_is, y_valid_os = torch.from_numpy(y_valid_is).float(), torch.from_numpy(y_valid_os).float()
y_test_is, y_test_os = torch.from_numpy(y_test_is).float(), torch.from_numpy(y_test_os).float()

train_set = Dataset(X_train, y_train)
valid_set_is, valid_set_os = Dataset(X_valid_is, y_valid_is), Dataset(X_valid_os, y_valid_os)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader_is, valid_loader_os = DataLoader(valid_set_is, batch_size=32, shuffle=True), DataLoader(valid_set_os, batch_size=32, shuffle=True)

# Initialize the MLP
mlp = DMLP(input_size=2048, hidden_layer1=1024, hidden_layer2=512, output_size=1783).to(device)

EPOCHS = 100
SAVE_MODEL = True
MODEL_NAME = f"multivariate_DMLPRegressor_fingerprints_epoch{EPOCHS}"

# Define the loss function and optimizer
loss_function = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
prev_loss = 0.0
stale = 0.0
train_losses = []
valid_losses_is = []
valid_losses_os = []

for epoch in range(EPOCHS):
    print(f"Starting epoch {epoch+1}")

    valid_loss_is = 0.0
    valid_loss_os = 0.0
    with torch.no_grad():
        for data in valid_loader_is:
            mol_vec, labels = data
            mol_vec, labels = mol_vec.to(device), labels.to(device)
            y_pred = mlp(mol_vec)
            batch_loss = loss_function(y_pred, labels)
            batch_loss = torch.sum(batch_loss)
            valid_loss_is += batch_loss.item()

        valid_loss_is /= len(valid_set_is)
        valid_losses_is.append(valid_loss_is)
        
        for data in valid_loader_os:
            mol_vec, labels = data
            mol_vec, labels = mol_vec.to(device), labels.to(device)
            y_pred = mlp(mol_vec)
            batch_loss = loss_function(y_pred, labels)
            batch_loss = torch.sum(batch_loss)
            valid_loss_os += batch_loss.item()

        valid_loss_os /= len(valid_set_os)
        valid_losses_os.append(valid_loss_os)

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
    print(f"in sample validation loss: {valid_loss_is}")
    print(f"out of sample validation loss: {valid_loss_os}")
    print("")

    if SAVE_MODEL:
        with open(WRITE_LOC + MODEL_NAME + ".txt", "a") as infile1:
            epoch_message = [
                f"Epoch {epoch}",
                f"Training Loss: {train_loss:.3f}",
                f"In sample validation Loss: {valid_loss_is:.3f}",
                f"Out of sample validation Loss: {valid_loss_os:.3f}"
            ]
            infile1.write("\n".join(epoch_message) + "\n" * 2)

if SAVE_MODEL:
    torch.save(mlp, MODEL_PATH + MODEL_NAME)

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