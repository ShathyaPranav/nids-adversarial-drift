"""Clean raw CICFlowMeter CSVs and produce a train/val/test split ready for training.

Handles the data-quality issues found during dataset verification:
  - Embedded junk header rows (a literal "Label" row mixed into the data)
  - Exact duplicate flows
  - Inf/NaN in Flow Byts/s and Flow Pkts/s (division-by-zero on near-zero duration)
  - Extremely rare classes (SQL Injection, XSS, etc. can have single-digit counts),
    which break a normal stratified split -- those classes are kept whole in train
    rather than dropped, and a warning is logged.

Usage:
    python src/data/preprocess.py --input data/raw/*.csv --out data/processed
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

NON_FEATURE_COLS = {"Timestamp", "Label"}
MIN_SAMPLES_FOR_STRATIFIED_SPLIT = 6  # need >=1 per split leg, comfortably


def load_raw(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        print(f"Loading {p} ...")
        frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows total")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["Label"] != "Label"].copy()
    print(f"Dropped {before - len(df)} embedded-header junk rows")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} exact duplicate rows")

    before = len(df)
    finite_mask = np.isfinite(df[feature_cols]).all(axis=1)
    df = df[finite_mask].copy()
    print(f"Dropped {before - len(df)} rows with NaN/Inf feature values")

    return df.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = 42):
    counts = df["Label"].value_counts()
    rare = counts[counts < MIN_SAMPLES_FOR_STRATIFIED_SPLIT].index.tolist()
    if rare:
        print(f"WARNING: classes too rare to stratify-split, keeping entirely in train: {rare}")
    rare_mask = df["Label"].isin(rare)
    df_rare, df_main = df[rare_mask], df[~rare_mask]

    train, temp = train_test_split(
        df_main, test_size=0.30, stratify=df_main["Label"], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["Label"], random_state=seed
    )
    train = pd.concat([train, df_rare], ignore_index=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", default=["data/raw/*.csv"])
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    paths = []
    for pattern in args.input:
        paths.extend(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched {args.input}. Run src/data/download.py first.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(paths)
    df = clean(df)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["Label"])

    print("\nClass distribution:")
    print(df["Label"].value_counts())

    train, val, test = split(df)

    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        X = scaler.transform(split_df[feature_cols]).astype(np.float32)
        y = split_df["label_id"].to_numpy(dtype=np.int64)
        np.savez(out_dir / f"{name}.npz", X=X, y=y)
        print(f"{name}: {X.shape[0]:,} rows -> {out_dir / f'{name}.npz'}")

    meta = {
        "feature_cols": feature_cols,
        "classes": label_encoder.classes_.tolist(),
        "n_features": len(feature_cols),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata ({len(feature_cols)} features, {len(meta['classes'])} classes) to {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
