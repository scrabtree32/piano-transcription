"""
Downloads the pretrained piano_transcription_inference checkpoint using
Python's urllib instead of the package's internal `os.system('wget ...')`
call, which silently fails on Windows (no wget by default).

Run this once, before any script that imports PianoTranscription. If the
checkpoint already exists, this does nothing -- safe to call every run.

Usage:
    python download_weights.py
"""

import os
import urllib.request
from pathlib import Path

CHECKPOINT_DIR = Path.home() / "piano_transcription_inference_data"
CHECKPOINT_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME
ZENODO_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)


def main():
    if CHECKPOINT_PATH.exists():
        print(f"Checkpoint already present at {CHECKPOINT_PATH}, skipping download.")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pretrained weights (~165MB) to {CHECKPOINT_PATH} ...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            print(f"\r  {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(ZENODO_URL, CHECKPOINT_PATH, reporthook=_progress)
    print("\nDone.")


if __name__ == "__main__":
    main()
