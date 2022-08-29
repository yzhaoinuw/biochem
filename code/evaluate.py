# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 11:47:27 2022

@author: Yue
"""

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn

from neural_networks import MLP, DMLP


def calculate_rank(y_sorted, y, y_pred):

    diff = abs(y_pred - y)
    lower = y - diff
    higher = y + diff

    cover = y_sorted[y_sorted <= higher]
    cover = cover[cover >= lower]
    return len(cover) / len(y_sorted)


MODEL_PATH = "../model/"
DATA_PATH = "../data/"
MIN_SAMPLE_COUNT = 4

data_split_file = "biochem_dataset_train_size0.6_min_sample_count4.json"
scaler_filename = f"standard_scaler_min_sample_count{MIN_SAMPLE_COUNT}.save"
broad2smiles_file = "broad2smiles.json"
embeddings_file = "biochem_smiles.npz"
features_file = "features.npy"
randvec_file = "broad2randvec.json"
broad2vec_file = "broad2vec.json"
broad2fingerprints_file = "broad2fingerprints.json"
feature_name_file = "feature_names.json"

with open(DATA_PATH + broad2smiles_file, "r") as infile1:
    broad2smiles = json.load(infile1)

features = np.load(DATA_PATH + features_file)

with open(DATA_PATH + data_split_file, "r") as infile1:
    data_split = json.load(infile1)

with open(DATA_PATH + feature_name_file, "r") as infile1:
    feature_names = json.load(infile1)

feature_names = np.array(feature_names)
#%%

# randvec
# with open(DATA_PATH + randvec_file, "r") as infile1:
#    embeddings = json.load(infile1)

# mol2vec
# with open(DATA_PATH + broad2vec_file, "r") as infile1:
#    embeddings = json.load(infile1)

# fingerprints
# with open(DATA_PATH + broad2fingerprints_file, "r") as infile1:
#    embeddings = json.load(infile1)

# transformers
embeddings = np.load(MODEL_PATH + embeddings_file)

with open(DATA_PATH + data_split_file, "r") as infile1:
    data_split = json.load(infile1)

X_valid_is = []
y_valid_is = []
X_valid_os = []
y_valid_os = []
X_test_is = []
y_test_is = []
X_test_os = []
y_test_os = []

for broad_id, row_ind in data_split["validation_is"]:
    smiles = broad2smiles[broad_id]
    embedding = embeddings.get(smiles, None)
    # embedding = embeddings.get(broad_id, None)

    if embedding is None:
        continue
    embedding = np.mean(embedding, axis=0)
    # embedding = np.array(list(embedding)).astype(float)
    X_valid_is.append(embedding)
    y_valid_is.append(features[row_ind])

for broad_id, row_ind in data_split["validation_os"]:
    smiles = broad2smiles[broad_id]
    embedding = embeddings.get(smiles, None)
    # embedding = embeddings.get(broad_id, None)

    if embedding is None:
        continue
    embedding = np.mean(embedding, axis=0)
    # embedding = np.array(list(embedding)).astype(float)
    X_valid_os.append(embedding)
    y_valid_os.append(features[row_ind])

for broad_id, row_ind in data_split["test_is"]:
    smiles = broad2smiles[broad_id]
    embedding = embeddings.get(smiles, None)
    # embedding = embeddings.get(broad_id, None)

    if embedding is None:
        continue
    embedding = np.mean(embedding, axis=0)
    # embedding = np.array(list(embedding)).astype(float)
    X_test_is.append(embedding)
    y_test_is.append(features[row_ind])

for broad_id, row_ind in data_split["test_os"]:
    smiles = broad2smiles[broad_id]
    embedding = embeddings.get(smiles, None)
    # embedding = embeddings.get(broad_id, None)
    if embedding is None:
        continue
    embedding = np.mean(embedding, axis=0)
    # embedding = np.array(list(embedding)).astype(float)
    X_test_os.append(embedding)
    y_test_os.append(features[row_ind])

scaler = joblib.load(MODEL_PATH + scaler_filename)
X_valid_is, X_valid_os = np.array(X_valid_is), np.array(X_valid_os)
y_valid_is, y_valid_os = np.array(y_valid_is), np.array(y_valid_os)
X_test_is, X_test_os = np.array(X_test_is), np.array(X_test_os)
y_test_is, y_test_os = np.array(y_test_is), np.array(y_test_os)

y_valid_is, y_valid_os = scaler.transform(y_valid_is), scaler.transform(y_valid_os)
y_test_is, y_test_os = scaler.transform(y_test_is), scaler.transform(y_test_os)

X_valid_is, X_valid_os = (
    torch.from_numpy(X_valid_is).float(),
    torch.from_numpy(X_valid_os).float(),
)
y_valid_is, y_valid_os = (
    torch.from_numpy(y_valid_is).float(),
    torch.from_numpy(y_valid_os).float(),
)
X_test_is, X_test_os = (
    torch.from_numpy(X_test_is).float(),
    torch.from_numpy(X_test_os).float(),
)
y_test_is, y_test_os = (
    torch.from_numpy(y_test_is).float(),
    torch.from_numpy(y_test_os).float(),
)

#%%
model_name = "DMLPRegressor_embedding_weight_decay0.1_min_sample_count4"
model = DMLP(input_size=512, hidden_layer1=1024, hidden_layer2=512, output_size=1783)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

checkpoint = torch.load(MODEL_PATH + model_name)
model.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]

model.eval()

loss_function = nn.L1Loss(reduction="none")
y_valid_is_pred = model(X_valid_is)
y_valid_os_pred = model(X_valid_os)
y_is_pred = model(X_test_is)
y_os_pred = model(X_test_os)

y_valid_is_loss = loss_function(y_valid_is_pred, y_valid_is)
y_valid_os_loss = loss_function(y_valid_os_pred, y_valid_os)
y_is_loss = loss_function(y_is_pred, y_test_is)
y_os_loss = loss_function(y_os_pred, y_test_os)

print(f"valid in sample loss is {torch.sum(y_valid_is_loss)/len(y_valid_is_loss)}")
print(f"valid out of sample loss is {torch.sum(y_valid_os_loss)/len(y_valid_os_loss)}")
print(f"test in sample loss is {torch.sum(y_is_loss)/len(y_is_loss)}")
print(f"test out of sample loss is {torch.sum(y_os_loss)/len(y_os_loss)}")
#%%
mean_feature_loss = torch.mean(y_valid_is_loss, dim=0).detach()

k = 10
top_k_loss = torch.topk(mean_feature_loss, k=k)
bottom_k_loss = torch.topk(mean_feature_loss, largest=False, k=k)
#%%
plt.scatter(
    np.arange(len(mean_feature_loss.numpy())).astype("str"),
    mean_feature_loss.numpy(),
    s=1,
)
plt.xlabel("Morphological Features")
plt.ylabel("Mean Squared Error")
plt.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False,
)
# plt.ylim(0, 2)
# plt.yticks(np.arange(0, 20, step=1))
