import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

AGE_FILE = "data/labels/age_raw.csv"
DAMAGE_DIR = Path("data/raw_full/damageprofiler")

# -----------------------------
# load age labels
# -----------------------------

age_df = pd.read_csv(AGE_FILE, sep=None, engine="python")
age_df.columns = age_df.columns.str.strip()

print("Detected columns:", age_df.columns.tolist())

if "bam_name" not in age_df.columns or "age" not in age_df.columns:
    raise ValueError("age_raw.csv must contain columns bam_name and age")

records = []

# -----------------------------
# extract damage features
# -----------------------------

for _, row in age_df.iterrows():

    sample = row["bam_name"]      # example: I0709
    age = row["age"]

    folder = DAMAGE_DIR / f"{sample}.bam"
    file = folder / "5pCtoT_freq.txt"

    if not file.exists():
        print(f"Skipping {sample} (missing {file})")
        continue

    df = pd.read_csv(file, sep="\t", comment="#")

    damage_pos1 = df.iloc[0]["5pC>T"]
    damage_mean10 = df["5pC>T"].iloc[:10].mean()

    records.append({
        "bam_name": sample,
        "age": age,
        "damage_pos1": damage_pos1,
        "damage_mean10": damage_mean10
    })

if len(records) == 0:
    raise RuntimeError("No samples processed")

data = pd.DataFrame(records)

print("\nSamples processed:", len(data))

# -----------------------------
# correlations
# -----------------------------

pearson1 = pearsonr(data["age"], data["damage_pos1"])
spearman1 = spearmanr(data["age"], data["damage_pos1"])

pearson10 = pearsonr(data["age"], data["damage_mean10"])
spearman10 = spearmanr(data["age"], data["damage_mean10"])

print("\nCorrelation with position-1 damage")
print("Pearson:", pearson1)
print("Spearman:", spearman1)

print("\nCorrelation with mean first 10 bp damage")
print("Pearson:", pearson10)
print("Spearman:", spearman10)