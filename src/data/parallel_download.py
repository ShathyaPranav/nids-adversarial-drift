"""Resumable multi-connection downloader for the corrected CSE-CIC-IDS2018 zip.

The host drops long single connections and is slow per-connection, so the file is split
into byte ranges fetched in parallel; each range resumes from its own partial file.

Usage:
    python src/data/parallel_download.py
"""
import sys
import threading
import time
import urllib.request
from pathlib import Path

URL = "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip"
DEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "CSECICIDS2018_improved.zip"
N_PARTS = 8
BLOCK = 1 << 20

lock = threading.Lock()
progress = {}


def total_size() -> int:
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_part(idx: int, start: int, end: int) -> None:
    part = DEST.with_name(f"{DEST.name}.part{idx}")
    want = end - start + 1
    while True:
        have = part.stat().st_size if part.exists() else 0
        progress[idx] = have
        if have >= want:
            return
        req = urllib.request.Request(URL, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                while True:
                    chunk = r.read(BLOCK)
                    if not chunk:
                        break
                    f.write(chunk)
                    with lock:
                        progress[idx] = progress.get(idx, 0) + len(chunk)
        except Exception as e:  # connection drops are expected; retry from where we left off
            time.sleep(2)


def main() -> None:
    size = total_size()
    print(f"Total size: {size:,} bytes")

    # Seed part 0 from an existing single-stream partial download, if any.
    if DEST.exists() and not DEST.with_name(f"{DEST.name}.part0").exists():
        existing = DEST.stat().st_size
        print(f"Existing partial file: {existing:,} bytes")

    step = -(-size // N_PARTS)
    ranges = [(i, i * step, min(size - 1, (i + 1) * step - 1)) for i in range(N_PARTS)]

    # Reuse the single-stream partial as the head of part 0.
    p0 = DEST.with_name(f"{DEST.name}.part0")
    if DEST.exists() and not p0.exists():
        keep = min(DEST.stat().st_size, ranges[0][2] - ranges[0][1] + 1)
        with open(DEST, "rb") as src, open(p0, "wb") as dst:
            remaining = keep
            while remaining:
                buf = src.read(min(BLOCK, remaining))
                if not buf:
                    break
                dst.write(buf)
                remaining -= len(buf)

    threads = [threading.Thread(target=fetch_part, args=r, daemon=True) for r in ranges]
    for t in threads:
        t.start()

    while any(t.is_alive() for t in threads):
        done = sum(progress.values())
        sys.stdout.write(f"\r{done / 1e9:.2f} / {size / 1e9:.2f} GB ({done * 100 // size}%)   ")
        sys.stdout.flush()
        time.sleep(5)

    print("\nAssembling final file...")
    tmp = DEST.with_name(DEST.name + ".assembled")
    with open(tmp, "wb") as out:
        for i, _, _ in ranges:
            with open(DEST.with_name(f"{DEST.name}.part{i}"), "rb") as p:
                while True:
                    buf = p.read(BLOCK)
                    if not buf:
                        break
                    out.write(buf)
    if tmp.stat().st_size != size:
        raise SystemExit(f"Size mismatch: {tmp.stat().st_size} != {size}")
    tmp.replace(DEST)
    for i, _, _ in ranges:
        DEST.with_name(f"{DEST.name}.part{i}").unlink()
    print(f"Done: {DEST}")


if __name__ == "__main__":
    main()
