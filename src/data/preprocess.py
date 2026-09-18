"""Build capped, cleaned train/val/test splits from the corrected CSE-CIC-IDS2018 zip.

Input is Liu et al.'s corrected release (data/raw/CSECICIDS2018_improved.zip): 10 daily CSVs,
~63M flows, ~94% benign, with fine-grained labels including "- Attempted" variants. The
full set is far too large to train on directly, so each class is capped at --cap rows using
uniform random sampling (all rows kept for classes smaller than the cap), which keeps every
rare attack class (Web Attack SQL/XSS/Brute Force, Infiltration variants) fully represented.
NOTE: capping changes the benign/attack ratio relative to real traffic.

Cleaning:
  - Drop identifier/leakage columns (id, Flow ID, IPs, Src Port, Timestamp).
  - Drop flows with NaN/Inf features (Flow Bytes/s, Flow Packets/s on ~zero durations).
  - Drop exact duplicate flows (prevents identical rows landing in both train and test).
  - "- Attempted" labels: --attempted merge (default, folded into the parent attack class),
    drop, or keep. FTP-BruteForce exists ONLY as "Attempted" in this release, so `drop`
    removes that class entirely.
  - signed log1p on features (heavy-tailed byte/packet/time counts), then standardize.

Usage:
    python src/data/preprocess.py --cap 200000
"""
import argparse
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DROP_COLS = ["id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", "Attempted Category"]
ATTEMPTED_SUFFIX = " - Attempted"
MIN_SAMPLES_FOR_STRATIFIED_SPLIT = 6
CHUNK_ROWS = 500_000


def normalize_labels(labels: pd.Series, attempted: str) -> pd.Series:
    if attempted == "merge":
        return labels.str.replace(ATTEMPTED_SUFFIX, "", regex=False)
    return labels


def sample_day(zip_path: str, member: str, cap: int, seed: int, attempted: str) -> dict:
    """Stream one day's CSV, keeping a uniform random sample of at most `cap` rows per class."""
    rng = np.random.default_rng(seed)
    kept: dict[str, pd.DataFrame] = {}
    seen = 0
    with zipfile.ZipFile(zip_path) as z, z.open(member) as f:
        for chunk in pd.read_csv(f, chunksize=CHUNK_ROWS, low_memory=False):
            seen += len(chunk)
            chunk = chunk.drop(columns=DROP_COLS)
            if attempted == "drop":
                chunk = chunk[~chunk["Label"].str.endswith(ATTEMPTED_SUFFIX)]
            chunk["Label"] = normalize_labels(chunk["Label"], attempted)

            feats = chunk.columns.drop("Label")
            chunk[feats] = chunk[feats].astype(np.float32)
            chunk = chunk[np.isfinite(chunk[feats].to_numpy()).all(axis=1)]

            chunk = chunk.assign(_r=rng.random(len(chunk)))
            for label, g in chunk.groupby("Label"):
                prev = kept.get(label)
                g = g if prev is None else pd.concat([prev, g], ignore_index=True)
                kept[label] = g.nsmallest(cap, "_r") if len(g) > cap else g
    print(f"  {member}: {seen:,} rows read", flush=True)
    return kept


def build_sample(zip_path: str, cap: int, seed: int, attempted: str, workers: int) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        members = [i.filename for i in z.infolist() if i.filename.endswith(".csv")]
    print(f"Sampling {len(members)} day files with {workers} workers (cap {cap:,}/class)...")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(sample_day, zip_path, m, cap, seed + i, attempted) for i, m in enumerate(members)]
        results = [f.result() for f in futures]

    merged: dict[str, list[pd.DataFrame]] = {}
    for kept in results:
        for label, g in kept.items():
            merged.setdefault(label, []).append(g)

    parts = []
    for label, gs in merged.items():
        g = pd.concat(gs, ignore_index=True)
        parts.append(g.nsmallest(cap, "_r") if len(g) > cap else g)
    return pd.concat(parts, ignore_index=True).drop(columns="_r")


def split(df: pd.DataFrame, seed: int):
    counts = df["Label"].value_counts()
    rare = counts[counts < MIN_SAMPLES_FOR_STRATIFIED_SPLIT].index.tolist()
    if rare:
        print(f"WARNING: classes too rare to stratify-split, keeping entirely in train: {rare}")
    rare_mask = df["Label"].isin(rare)
    df_rare, df_main = df[rare_mask], df[~rare_mask]

    train, temp = train_test_split(df_main, test_size=0.30, stratify=df_main["Label"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["Label"], random_state=seed)
    train = pd.concat([train, df_rare], ignore_index=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default="data/raw/CSECICIDS2018_improved.zip")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--cap", type=int, default=200_000, help="max rows kept per class")
    parser.add_argument("--attempted", choices=["merge", "drop", "keep"], default="merge")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.zip).exists():
        raise SystemExit(f"{args.zip} not found. Run: python src/data/download.py --corrected")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_sample(args.zip, args.cap, args.seed, args.attempted, args.workers)
    print(f"\nSampled {len(df):,} rows")

    feature_cols = [c for c in df.columns if c != "Label"]
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df):,} exact duplicate rows")

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["Label"])
    print("\nClass distribution (after capping + dedupe):")
    print(df["Label"].value_counts().to_string())

    train, val, test = split(df, args.seed)

    scaler = StandardScaler()
    scaler.fit(signed_log1p(train[feature_cols].to_numpy()))

    class_names = label_encoder.classes_.tolist()
    class_counts = {}
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        X = scaler.transform(signed_log1p(split_df[feature_cols].to_numpy())).astype(np.float32)
        y = split_df["label_id"].to_numpy(dtype=np.int64)
        np.savez(out_dir / f"{name}.npz", X=X, y=y)
        class_counts[name] = {c: int((y == i).sum()) for i, c in enumerate(class_names)}
        print(f"{name}: {X.shape[0]:,} rows -> {out_dir / f'{name}.npz'}")

    meta = {
        "source": "CSECICIDS2018_improved (Liu et al., 2022 corrected release)",
        "cap_per_class": args.cap,
        "attempted_mode": args.attempted,
        "transform": "signed_log1p + standardize (fit on train)",
        "feature_cols": feature_cols,
        "classes": class_names,
        "n_features": len(feature_cols),
        "class_counts": class_counts,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata ({len(feature_cols)} features, {len(class_names)} classes) to {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
