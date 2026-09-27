"""
src/05_train.py

Matching Model Training Pipeline.

Trains a tree-based binary classifier to predict P(match) for candidate entity pairs
(S1 entity + candidate S2/S3 entity). Saves the trained model and feature metadata under models/.
"""

import argparse
import json
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# Import feature columns from 04_features module
try:
    features_mod = importlib.import_module("src.04_features")
    FEATURE_COLUMNS = features_mod.FEATURE_COLUMNS
except Exception:
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


def predict_match_probabilities(
    model: object,
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Predict P(match) probability for candidate entity pairs.

    Parameters:
    -----------
    model : trained classifier
        Scikit-learn compatible binary classifier with predict_proba method.
    df : pd.DataFrame
        DataFrame containing feature columns.
    feature_cols : list of str, optional
        List of feature column names in trained model order. Defaults to FEATURE_COLUMNS.

    Returns:
    --------
    np.ndarray
        1D array of floats containing P(match) in [0.0, 1.0].
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"DataFrame is missing required feature columns: {missing_cols}")

    X = df[feature_cols].copy()
    probs = model.predict_proba(X)[:, 1]
    return probs


def evaluate_model(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate binary matching model predictions.
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    ap = float(average_precision_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    f05 = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))

    return {
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "f0.5": f05,
    }


def train_matching_model(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "label",
    classifier_type: str = "hist_gb",
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[object, Dict[str, float]]:
    """
    Train a binary matching classifier on pair features.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing feature columns and binary target label.
    feature_cols : list of str, optional
        List of feature column names to train on.
    target_col : str
        Column name for ground truth label (1 = true match, 0 = non-match).
    classifier_type : str
        'hist_gb' (HistGradientBoostingClassifier) or 'random_forest'.
    val_size : float
        Validation split fraction.
    random_state : int
        Random seed.

    Returns:
    --------
    Tuple[object, Dict[str, float]]
        (trained_model, validation_metrics_dict)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in training DataFrame.")

    X = df[feature_cols].copy()
    y = df[target_col].values

    if len(np.unique(y)) < 2:
        raise ValueError("Training dataset must contain both positive (1) and negative (0) candidate labels.")

    if val_size > 0.0 and len(df) >= 10:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = X, X, y, y

    if classifier_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        # Default: HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            max_depth=10,
            random_state=random_state,
            class_weight="balanced",
        )

    print(f"Training {model.__class__.__name__} on {len(X_train)} candidate pairs...")
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_model(y_val, val_probs)

    print("Validation Performance Metrics:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    return model, metrics


def save_model_artifacts(
    model: object,
    output_dir: Union[str, Path] = "models",
    model_name: str = "matching_model.joblib",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[Path, Path]:
    """
    Save trained model and feature schema metadata under models/.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / model_name
    joblib.dump(model, model_path)
    print(f"Saved trained model artifact to {model_path}")

    schema_path = out_dir / "feature_columns.json"
    with open(schema_path, "w") as f:
        json.dump({"feature_columns": feature_cols}, f, indent=2)
    print(f"Saved feature column schema to {schema_path}")

    return model_path, schema_path


def main():
    parser = argparse.ArgumentParser(description="Train binary matching classifier on entity candidate pair features.")
    parser.add_argument("--features", type=str, default="output/features_train.csv", help="Path to input features CSV/parquet.")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save trained model artifacts.")
    parser.add_argument("--classifier", type=str, choices=["hist_gb", "random_forest"], default="hist_gb", help="Classifier type.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation set split fraction.")

    args = parser.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"Error: Features file {feat_path} does not exist. Run src/04_features.py first.")
        return

    print(f"Loading feature dataset from {feat_path}...")
    if feat_path.suffix == ".parquet":
        df = pd.read_parquet(feat_path)
    else:
        df = pd.read_csv(feat_path)

    print(f"Loaded {len(df)} candidate pair rows.")
    print("Label distribution:")
    print(df["label"].value_counts(normalize=True).to_dict())

    model, metrics = train_matching_model(
        df=df,
        feature_cols=FEATURE_COLUMNS,
        target_col="label",
        classifier_type=args.classifier,
        val_size=args.val_size,
    )

    save_model_artifacts(model, output_dir=args.model_dir, feature_cols=FEATURE_COLUMNS)


if __name__ == "__main__":
    main()
