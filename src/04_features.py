"""
src/04_features.py

Pairwise feature engineering for candidate entity pairs (Source 1 + candidate Source 2/Source 3).

Calculates string, address, numeric overlap, and metadata similarity features
for candidate pairs produced by candidate generation / blocking.
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

FEATURE_COLUMNS = [
    "name_exact",
    "name_ratio",
    "name_token_ratio",
    "name_length_difference",
    "address_exact",
    "address_ratio",
    "address_token_ratio",
    "address_length_difference",
    "numeric_overlap",
    "country_match",
    "source_type",
]


def compute_numeric_overlap(addr1: str, addr2: str) -> float:
    """Calculate Jaccard similarity of digit sequences extracted from two addresses."""
    nums1 = set(re.findall(r"\d+", str(addr1)))
    nums2 = set(re.findall(r"\d+", str(addr2)))
    if not nums1 and not nums2:
        return 1.0
    if not nums1 or not nums2:
        return 0.0
    return len(nums1 & nums2) / len(nums1 | nums2)


def load_candidate_pairs(candidates_input: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Load candidate pairs into a normalized DataFrame with columns:
    ['source1_entity_id', 'candidate_entity_id'].

    Supports:
      - CSV/TSV files with candidate_entity_ids (comma-separated list format)
      - CSV/TSV files with pair-level columns ('source1_entity_id', 'candidate_entity_id')
      - Pre-loaded pandas DataFrames
    """
    if isinstance(candidates_input, (str, Path)):
        path = Path(candidates_input)
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    elif isinstance(candidates_input, pd.DataFrame):
        df = candidates_input.copy()
    else:
        raise ValueError(f"Unsupported input type for candidate pairs: {type(candidates_input)}")

    if "candidate_entity_ids" in df.columns:
        # Explode comma-separated candidate entity IDs
        df["candidate_entity_ids"] = df["candidate_entity_ids"].fillna("")
        df["candidate_entity_id"] = df["candidate_entity_ids"].astype(str).str.split(",")
        exploded = df.explode("candidate_entity_id")
        exploded["candidate_entity_id"] = exploded["candidate_entity_id"].astype(str).str.strip()
        exploded = exploded[exploded["candidate_entity_id"] != ""].reset_index(drop=True)
        return exploded[["source1_entity_id", "candidate_entity_id"]]

    cols = list(df.columns)
    if "source1_entity_id" in cols and "candidate_entity_id" in cols:
        return df[["source1_entity_id", "candidate_entity_id"]]
    elif len(cols) >= 2:
        df = df.rename(columns={cols[0]: "source1_entity_id", cols[1]: "candidate_entity_id"})
        return df[["source1_entity_id", "candidate_entity_id"]]
    else:
        raise ValueError(f"Could not parse candidate pair columns from DataFrame with columns {cols}")


def load_ground_truth(gt_input: Union[str, Path, pd.DataFrame]) -> Set[Tuple[str, str]]:
    """
    Load ground truth true matches into a set of (source1_entity_id, matched_entity_id) tuples.
    """
    if isinstance(gt_input, (str, Path)):
        path = Path(gt_input)
        sep = "\t" if path.suffix == ".tsv" else ","
        gt_df = pd.read_csv(path, sep=sep)
    elif isinstance(gt_input, pd.DataFrame):
        gt_df = gt_input.copy()
    else:
        raise ValueError(f"Unsupported ground truth input type: {type(gt_input)}")

    if "matched_entity_ids" in gt_df.columns:
        gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
        gt_df["cand_id"] = gt_df["matched_entity_ids"].astype(str).str.split(",")
        exploded = gt_df.explode("cand_id")
        exploded["cand_id"] = exploded["cand_id"].astype(str).str.strip()
        exploded = exploded[exploded["cand_id"] != ""]
        return set(zip(exploded["source1_entity_id"], exploded["cand_id"]))
    
    return set()


def extract_pair_features(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    gt_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Extract pairwise feature representations for candidate entity pairs.

    Parameters:
    -----------
    pairs_df : pd.DataFrame
        DataFrame with columns ['source1_entity_id', 'candidate_entity_id']
    s1_df : pd.DataFrame
        Source 1 entities with ['entity_id', 'business_name', 'business_address', 'country']
    cand_df : pd.DataFrame
        Combined Source 2 & Source 3 entities with ['entity_id', 'business_name', 'business_address', 'country']
    gt_df : pd.DataFrame, optional
        Ground truth dataframe with ['source1_entity_id', 'matched_entity_ids'] for labeling.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing pair identifiers, engineered feature columns, and optional ground truth 'label'.
    """
    if pairs_df.empty:
        cols = ["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS
        if gt_df is not None:
            cols.append("label")
        return pd.DataFrame(columns=cols)

    # Merge entity attributes
    merged = pairs_df.merge(
        s1_df.rename(
            columns={
                "business_name": "s1_name",
                "business_address": "s1_address",
                "country": "s1_country",
            }
        ),
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
    ).drop(columns=["entity_id"], errors="ignore")

    merged = merged.merge(
        cand_df.rename(
            columns={
                "business_name": "cand_name",
                "business_address": "cand_address",
                "country": "cand_country",
            }
        ),
        left_on="candidate_entity_id",
        right_on="entity_id",
        how="left",
    ).drop(columns=["entity_id"], errors="ignore")

    s1_names = merged["s1_name"].fillna("").astype(str).values
    cand_names = merged["cand_name"].fillna("").astype(str).values

    s1_addresses = merged["s1_address"].fillna("").astype(str).values
    cand_addresses = merged["cand_address"].fillna("").astype(str).values

    s1_countries = merged["s1_country"].fillna("").astype(str).values
    cand_countries = merged["cand_country"].fillna("").astype(str).values
    cand_ids = merged["candidate_entity_id"].fillna("").astype(str).values

    # 1. Name features
    name_exact = np.array(
        [1 if n1.lower().strip() == n2.lower().strip() and n1 != "" else 0 for n1, n2 in zip(s1_names, cand_names)],
        dtype=np.int32,
    )
    name_ratio = np.array(
        [fuzz.ratio(n1, n2) / 100.0 for n1, n2 in zip(s1_names, cand_names)],
        dtype=np.float32,
    )
    name_token_ratio = np.array(
        [fuzz.token_sort_ratio(n1, n2) / 100.0 for n1, n2 in zip(s1_names, cand_names)],
        dtype=np.float32,
    )
    name_len_diff = np.array(
        [abs(len(n1) - len(n2)) for n1, n2 in zip(s1_names, cand_names)],
        dtype=np.int32,
    )

    # 2. Address features
    addr_exact = np.array(
        [1 if a1.lower().strip() == a2.lower().strip() and a1 != "" else 0 for a1, a2 in zip(s1_addresses, cand_addresses)],
        dtype=np.int32,
    )
    addr_ratio = np.array(
        [fuzz.ratio(a1, a2) / 100.0 for a1, a2 in zip(s1_addresses, cand_addresses)],
        dtype=np.float32,
    )
    addr_token_ratio = np.array(
        [fuzz.token_sort_ratio(a1, a2) / 100.0 for a1, a2 in zip(s1_addresses, cand_addresses)],
        dtype=np.float32,
    )
    addr_len_diff = np.array(
        [abs(len(a1) - len(a2)) for a1, a2 in zip(s1_addresses, cand_addresses)],
        dtype=np.int32,
    )
    num_overlap = np.array(
        [compute_numeric_overlap(a1, a2) for a1, a2 in zip(s1_addresses, cand_addresses)],
        dtype=np.float32,
    )

    # 3. Other features
    country_match = np.array(
        [
            1 if (c1.lower().strip() == c2.lower().strip() or not c1 or not c2) else 0
            for c1, c2 in zip(s1_countries, cand_countries)
        ],
        dtype=np.int32,
    )
    source_type = np.array(
        [2 if cid.startswith("S2") else (3 if cid.startswith("S3") else 0) for cid in cand_ids],
        dtype=np.int32,
    )

    result_df = pd.DataFrame({
        "source1_entity_id": merged["source1_entity_id"],
        "candidate_entity_id": merged["candidate_entity_id"],
        "name_exact": name_exact,
        "name_ratio": name_ratio,
        "name_token_ratio": name_token_ratio,
        "name_length_difference": name_len_diff,
        "address_exact": addr_exact,
        "address_ratio": addr_ratio,
        "address_token_ratio": addr_token_ratio,
        "address_length_difference": addr_len_diff,
        "numeric_overlap": num_overlap,
        "country_match": country_match,
        "source_type": source_type,
    })

    # Ground truth labeling if provided
    if gt_df is not None:
        true_matches = load_ground_truth(gt_df)
        labels = [
            1 if (s1, cand) in true_matches else 0
            for s1, cand in zip(result_df["source1_entity_id"], result_df["candidate_entity_id"])
        ]
        result_df["label"] = np.array(labels, dtype=np.int32)

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Extract features for candidate entity resolution pairs.")
    parser.add_argument("--candidates", type=str, default="output/candidate_pairs.tsv", help="Path to candidate pairs file.")
    parser.add_argument("--s1", type=str, default="data/train/train_source1.tsv", help="Path to Source 1 entities TSV.")
    parser.add_argument("--s2", type=str, default="data/train/train_source2.tsv", help="Path to Source 2 entities TSV.")
    parser.add_argument("--s3", type=str, default="data/train/train_source3.tsv", help="Path to Source 3 entities TSV.")
    parser.add_argument("--gt", type=str, default="data/train/train_ground_truth.tsv", help="Path to ground truth TSV (optional).")
    parser.add_argument("--output", type=str, default="output/features_train.csv", help="Path to save output features.")

    args = parser.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        print(f"Error: Candidates file {cand_path} does not exist.")
        return

    print(f"Loading candidate pairs from {cand_path}...")
    pairs_df = load_candidate_pairs(cand_path)
    print(f"Loaded {len(pairs_df)} candidate pairs.")

    print("Loading entity source files...")
    s1_df = pd.read_csv(args.s1, sep="\t")
    s2_df = pd.read_csv(args.s2, sep="\t")
    s3_df = pd.read_csv(args.s3, sep="\t")
    cand_entities_df = pd.concat([s2_df, s3_df], ignore_index=True)

    gt_df = None
    gt_path = Path(args.gt)
    if gt_path.exists():
        print(f"Loading ground truth from {gt_path}...")
        gt_df = pd.read_csv(gt_path, sep="\t")

    print("Extracting pair features...")
    features_df = extract_pair_features(pairs_df, s1_df, cand_entities_df, gt_df=gt_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Saved extracted features ({len(features_df)} rows, {features_df.shape[1]} columns) to {output_path}")


if __name__ == "__main__":
    main()
