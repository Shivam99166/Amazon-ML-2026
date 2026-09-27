import pandas as pd
from pathlib import Path


FILES = {
    "source1": "data/train/train_source1.tsv",
    "source2": "data/train/train_source2.tsv",
    "source3": "data/train/train_source3.tsv",
    "ground_truth": "data/train/train_ground_truth.tsv",
}


for name, path in FILES.items():

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)

    df = pd.read_csv(path, sep="\t")

    print("Shape:", df.shape)
    print("Columns:", list(df.columns)) 

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:", df.duplicated().sum())