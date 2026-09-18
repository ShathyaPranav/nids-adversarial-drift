"""Fetch CSE-CIC-IDS2018 data.

Two sources:
  --sample     Smallest single-day CICFlowMeter CSV from the official AWS Open Data
               bucket (no credentials needed). Fast, good for pipeline development,
               but this is the *raw* release with known label-corruption issues
               (Liu et al., 2022).
  --corrected  Liu et al.'s relabeled/corrected release (~10.4 GB zip, parallel + resumable). This is what
               the synopsis commits to training on. No account needed, but large.

Usage:
    python src/data/download.py --sample
    python src/data/download.py --corrected
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SAMPLE_S3_KEY = (
    "Processed Traffic Data for ML Algorithms/"
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
)
SAMPLE_BUCKET = "cse-cic-ids2018"

CORRECTED_SIZE_BYTES = 10426851729
CORRECTED_URL = (
    "https://intrusion-detection.distrinet-research.be/"
    "CNS2022/Datasets/CSECICIDS2018_improved.zip"
)


def download_sample() -> None:
    dest = RAW_DIR / "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
    if dest.exists():
        print(f"Already present: {dest}")
        return
    print("Downloading smallest raw day (Thursday-01-03-2018) from AWS Open Data...")
    subprocess.run(
        [
            "aws", "s3", "cp", "--no-sign-request",
            f"s3://{SAMPLE_BUCKET}/{SAMPLE_S3_KEY}",
            str(dest),
        ],
        check=True,
    )
    print(f"Saved to {dest}")


def download_corrected() -> None:
    from parallel_download import main as parallel_main

    dest = RAW_DIR / "CSECICIDS2018_improved.zip"
    if dest.exists() and dest.stat().st_size == CORRECTED_SIZE_BYTES:
        print(f"Already present: {dest}")
        return
    print(f"Downloading corrected release (~10.4 GB) from {CORRECTED_URL}")
    parallel_main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="Download the small raw sample day")
    parser.add_argument("--corrected", action="store_true", help="Download the full corrected release")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.sample and not args.corrected:
        parser.error("pass --sample or --corrected")

    if args.sample:
        download_sample()
    if args.corrected:
        download_corrected()


if __name__ == "__main__":
    main()
