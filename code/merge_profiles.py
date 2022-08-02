#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 23:55:02 2022

@author: yuezhao
"""

import os

import pandas as pd

DATA_PATH = "../data/"
profile_path = DATA_PATH + "profiles_mean_well/"

# load first 17 cols
cols = list(range(17))
dfs = []

for i, plate_file in enumerate(os.listdir(profile_path)):
    if not plate_file.endswith(".csv"):
        continue
    if i % 10 == 0:
        print(f"plate {i}")
    df = pd.read_csv(profile_path + plate_file, usecols=cols)
    df["Metadata_ASSAY_WELL_ROLE"] = df["Metadata_ASSAY_WELL_ROLE"].replace(
        {"treated": "trt", "mock": "control"}
    )
    well_no_eq = (df["Metadata_Well"] == df["Metadata_well_position"]).all()
    plate_no_eq = (df["Metadata_Assay_Plate_Barcode"] == df["Metadata_Plate"]).all()
    role_type_eq = (
        df["Metadata_ASSAY_WELL_ROLE"] == df["Metadata_broad_sample_type"]
    ).all()
    role_type_eq2 = (df["Metadata_pert_type"] == df["Metadata_broad_sample_type"]).all()
    cell_id_eq = (df["Metadata_cell_id"] == "U2OS").all()

    if not well_no_eq:
        print("not well_no_eq")
    if not plate_no_eq:
        print("not plate_no_eq")
    if not role_type_eq:
        print("not role_type_eq")
    if not role_type_eq2:
        print("not role_type_eq2")
    if not cell_id_eq:
        print("not cell_id_eq")

    dfs.append(df)

df_merge = pd.concat(dfs, ignore_index=True)
#%%
df_merge = df_merge.drop("Metadata_well_position", axis=1)
df_merge = df_merge.drop("Metadata_pert_well", axis=1)
df_merge = df_merge.drop("Metadata_Assay_Plate_Barcode", axis=1)
df_merge = df_merge.drop("Metadata_ASSAY_WELL_ROLE", axis=1)
df_merge = df_merge.drop("Metadata_broad_sample_type", axis=1)
df_merge = df_merge.drop("Metadata_cell_id", axis=1)
df_merge = df_merge.drop("Metadata_solvent", axis=1)
df_merge = df_merge.drop("Metadata_pert_vehicle", axis=1)
df_merge = df_merge.drop("Metadata_pert_id_vendor", axis=1)

loc = "../data/"
# df_merge.to_csv(loc+'profile_merged.csv', index=False)
