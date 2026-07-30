"""
Download a small subset of the MAESTRO v3.0.0 test split.

Grabs the official metadata CSV, filters by split and optional composer, 
sorts by duration, and downloads audio/MIDI pairs.

Usage:
    python download_maestro_subset.py --composer "Mozart" --num 1 --max-minutes 5 --out data/maestro_subset
"""

import argparse
import csv
import io
import json
import os
import urllib.request

CSV_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"
AUDIO_MIDI_BASE_URL = "https://huggingface.co/datasets/RichardErkhov/maestro/resolve/main/"


def fetch_csv_rows():
    print(f"Fetching metadata: {CSV_URL}")
    with urllib.request.urlopen(CSV_URL) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def fetch_available_hf_files():
    api_url = "https://huggingface.co/api/datasets/RichardErkhov/maestro"
    print(f"Checking what's available on the mirror: {api_url}")
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {sibling["rfilename"] for sibling in data.get("siblings", [])}


def download_file(remote_path, local_path):
    if os.path.exists(local_path):
        print(f"  already have {local_path}")
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = AUDIO_MIDI_BASE_URL + remote_path
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(local_path, "wb") as out_f:
        out_f.write(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composer", type=str, default="", help="optional composer filter (e.g. 'Mozart')")
    ap.add_argument("--num", type=int, default=1, help="how many recordings to grab")
    ap.add_argument("--max-minutes", type=float, default=5.0,
                     help="only consider recordings shorter than this")
    ap.add_argument("--out", default="data/maestro_subset", help="output directory")
    args = ap.parse_args()

    rows = fetch_csv_rows()
    test_rows = [r for r in rows if r["split"] == "test"]
    
    # Filter by composer if specified
    if args.composer:
        test_rows = [r for r in test_rows if args.composer.lower() in r["canonical_composer"].lower()]
        print(f"Filtered to composer matching '{args.composer}': found {len(test_rows)} tracks.")

    test_rows = [r for r in test_rows if float(r["duration"]) <= args.max_minutes * 60]
    test_rows.sort(key=lambda r: float(r["duration"]))

    if not test_rows:
        raise SystemExit("No test recordings found matching your filters. Try raising --max-minutes or checking composer spelling.")

    available = fetch_available_hf_files()
    test_rows = [
        r for r in test_rows
        if r["audio_filename"] in available and r["midi_filename"] in available
    ]

    if not test_rows:
        raise SystemExit("None of the candidate recordings matched your filter and were found on the mirror.")

    chosen = test_rows[: args.num]

    manifest = []
    for row in chosen:
        audio_rel = row["audio_filename"]
        midi_rel = row["midi_filename"]
        local_audio = os.path.join(args.out, audio_rel)
        local_midi = os.path.join(args.out, midi_rel)

        print(f"\nDownloading: {row['canonical_composer']} - {row['canonical_title']} "
              f"({float(row['duration']):.1f}s)")
        download_file(audio_rel, local_audio)
        download_file(midi_rel, local_midi)

        manifest.append({
            "composer": row["canonical_composer"],
            "title": row["canonical_title"],
            "duration": float(row["duration"]),
            "audio_path": local_audio,
            "midi_path": local_midi,
        })

    manifest_path = os.path.join(args.out, "manifest.json")
    
    # Merge with existing manifest if it exists so we don't overwrite Scarlatti
    existing_manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            try:
                existing_manifest = json.load(f)
            except Exception:
                pass
    
    # Avoid duplicate entries
    existing_titles = {m["title"] for m in existing_manifest}
    for m in manifest:
        if m["title"] not in existing_titles:
            existing_manifest.append(m)

    with open(manifest_path, "w") as f:
        json.dump(existing_manifest, f, indent=2)

    print(f"\nDone. Updated manifest written to {manifest_path}")


if __name__ == "__main__":
    main()