# -----------------------------
# REPRODUCIBILITY
# -----------------------------
import colorcet as cc
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"

import random
random.seed(42)

import numpy as np
np.random.seed(42)


# -----------------------------
# IMPORTS
# -----------------------------
import pandas as pd
import json
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# -----------------------------
# LOAD CONFIG
# -----------------------------
with open("config/config_harvard_60_cv5.json") as f:
    CONFIG = json.load(f)

HARVARD_FOLDER = CONFIG['mDamage_folder']
FULLTEST_FOLDER = "data/full_test"
SAVE_PATH = "data/data_comparison"
UMAP_PATH = "UMAP_BATCH_ANALYSIS"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(UMAP_PATH, exist_ok=True)


# -----------------------------
# HARVARD METADATA
# -----------------------------
metadata_groups = pd.read_csv(CONFIG['batch_metadata_file'], sep="\t")


# -----------------------------
# FULL TEST METADATA (TAB!)
# -----------------------------
fulltest_batch_df = pd.read_csv(
    "data/full_test/batch_names.csv",
    sep=","
)

fulltest_dict = dict(zip(
    fulltest_batch_df['RunAccession'],
    fulltest_batch_df['Batch']
))


# -----------------------------
# GET HARVARD BATCH
# -----------------------------
def get_harvard_batch(bam_name):

    matching = metadata_groups[
        metadata_groups['sample'] == bam_name
        ]

    if matching.empty:
        return None

    return matching['article_group'].iloc[0]


# -----------------------------
# DAMAGE FEATURES (SAFE)
# -----------------------------
def get_damage_features(folder_path):

    file_name = "misincorporation.txt"
    full_file_path = os.path.join(folder_path, file_name)

    try:
        df = pd.read_csv(full_file_path, sep="\t", skiprows=3)

        rows = df[df["Pos"].astype(int) < 60]
        features = rows[CONFIG['feature_names']]
        features = features.replace(r'\.(\d+)\.', r'.\1', regex=True)
        features = features.astype(float)

        total = features['Total'].replace(0, np.nan)

        normalized_features = features.copy()

        for col in features.columns:
            if col != "Total":
                normalized_features[col] = features[col] / total

        normalized_features = normalized_features.fillna(0)
        normalized_features = normalized_features.drop(columns=['Total'])

        return normalized_features

    except:
        return None


# -----------------------------
# WALK DATASET
# -----------------------------
def process_dataset(root_folder, prefix):

    df_all = pd.DataFrame()

    for root, dirs, _ in os.walk(root_folder):
        for directory in dirs:

            path = os.path.join(root, directory)
            bam_name = directory.split(".")[0]

            if prefix == "HARVARD":
                batch_name = get_harvard_batch(bam_name)
            else:
                batch_name = fulltest_dict.get(bam_name)

            if batch_name is None:
                continue

            features = get_damage_features(path)
            if features is None:
                continue

            flattened = features.values.ravel()

            new_cols = [
                f"{col}_{i}"
                for i in range(features.shape[0])
                for col in features.columns
            ]

            df_row = pd.DataFrame([flattened], columns=new_cols)

            df_row.insert(0, "label", 0)
            df_row.insert(1, "bam_name", bam_name)
            df_row.insert(2, "batch_name", f"{prefix}_{batch_name}")

            df_all = pd.concat([df_all, df_row], ignore_index=True)

    return df_all


# -----------------------------
# BUILD DATASET
# -----------------------------
print("Processing Harvard...")
df_harvard = process_dataset(HARVARD_FOLDER, "HARVARD")

print("Processing FULL TEST...")
df_fulltest = process_dataset(FULLTEST_FOLDER, "FULLTEST")

df = pd.concat([df_harvard, df_fulltest], ignore_index=True)

df['batch_name'] = df['batch_name'].astype(str)
df['bam_name'] = df['bam_name'].astype(str)

numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

save_file = os.path.join(SAVE_PATH, "batch_effect_main.csv")
df.to_csv(save_file, index=False)

print("Saved:", save_file)
print("Samples:", len(df))
print("Unique batches:", df['batch_name'].nunique())


# -----------------------------
# UMAP FUNCTION
# -----------------------------
def draw_umap(df, save_path, title="UMAP"):

    print("Running:", title)

    # ---------------------------------------
    # Remove non-numeric columns
    # ---------------------------------------
    X = df.drop(columns=['label', 'batch_name', 'bam_name'])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0).values

    # ---------------------------------------
    # Dataset label (shape)
    # ---------------------------------------
    dataset = df['batch_name'].apply(
        lambda x: 'Harvard' if x.startswith('HARVARD_') else 'FullTest'
    )

    # ---------------------------------------
    # Batch label (color)
    # ---------------------------------------
    batch = df['batch_name'].str.replace(
        r'^(HARVARD_|FULLTEST_)',
        '',
        regex=True
    )

    # ---------------------------------------
    # Scale + UMAP
    # ---------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=6,
        min_dist=0.005,
        spread=2.5,
        repulsion_strength=2.0,
        learning_rate=1.5,
        metric='cosine',
        random_state=42,
        init='spectral'
    )

    embedding = reducer.fit_transform(X_scaled)

    # ---------------------------------------
    # Glasbey palette (~25 distinct colors)
    # ---------------------------------------
    unique_batches = batch.unique()
    glasbey = cc.glasbey[:len(unique_batches)]
    batch_to_color = dict(zip(unique_batches, glasbey))

    # ---------------------------------------
    # Plot Harvard vs FULLTEST
    # ---------------------------------------
    plt.figure(figsize=(10,8))

    for b in unique_batches:

        harv_mask = (batch == b) & (dataset == "Harvard")
        full_mask = (batch == b) & (dataset == "FullTest")

        color = batch_to_color[b]

        plt.scatter(
            embedding[harv_mask,0],
            embedding[harv_mask,1],
            c=[color],
            s=50,
            marker='o',
            alpha=0.6
        )

        plt.scatter(
            embedding[full_mask,0],
            embedding[full_mask,1],
            c=[color],
            s=50,
            marker='D',
            alpha=0.6
        )

    # ---------------------------------------
    # Legend
    # ---------------------------------------
    from matplotlib.lines import Line2D

    batch_legend = [
        Line2D([0],[0],
               marker='o',
               color='w',
               label=b,
               markerfacecolor=batch_to_color[b],
               markeredgecolor='black',
               markersize=8)
        for b in unique_batches
    ]

    dataset_legend = [
        Line2D([0],[0], marker='o', color='black',
               label='Harvard', linestyle='None', markersize=8),
        Line2D([0],[0], marker='D', color='black',
               label='FullTest', linestyle='None', markersize=8)
    ]

    plt.legend(handles=batch_legend + dataset_legend,
               bbox_to_anchor=(1.05,1),
               loc='upper left',
               frameon=False)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# -----------------------------
# RUN UMAP
# -----------------------------
draw_umap(
    df,
    os.path.join(UMAP_PATH, "UMAP_pre_batch.png"),
    "UMAP for Harvard and external test"
)
