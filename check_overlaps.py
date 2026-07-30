"""
Diagnostic for the blank-notation bug: checks whether the quantized note
events for a given recording contain genuine time-overlaps within the same
staff that AREN'T simple identical-onset chords.

Why this matters: build_score() in quantize_to_musicxml.py only merges notes
into a chord when they share the exact same (onset, duration). Real piano
music very often has partial overlaps instead -- e.g. a held note while
something else starts and stops underneath it. Two such notes end up as
separate Note objects inserted into the same single, unsplit "voice" stream.
Standard notation requires a single voice's notes to be non-overlapping
(true simultaneities must be an actual chord, or split into voice 1/voice 2)
-- if that's violated, some renderers (possibly MuseScore here) may fail to
render the offending notes rather than raising a clear error.

Usage: same args as quantize_to_musicxml.py (reuses its functions directly).

    python check_overlaps.py --manifest data/maestro_subset/manifest.json \
        --scores results/scores.csv --results-dir results \
        --composer-filter scarlatti --title "Sonata in D Major, K96"
"""

import argparse
import json

from quantize_to_musicxml import (
    load_scores, pick_recording, safe_name, estimate_beat_grid, quantize_notes,
)
import os


def find_overlaps(events, staff_split_pitch):
    treble = [e for e in events if e["pitch"] >= staff_split_pitch]
    bass = [e for e in events if e["pitch"] < staff_split_pitch]

    for label, evs in (("treble", treble), ("bass", bass)):
        evs_sorted = sorted(evs, key=lambda e: e["onset"])
        overlap_count = 0
        identical_count = 0
        examples = []
        for i in range(len(evs_sorted)):
            a = evs_sorted[i]
            a_end = a["onset"] + a["duration"]
            for j in range(i + 1, len(evs_sorted)):
                b = evs_sorted[j]
                if b["onset"] >= a_end:
                    break  # sorted by onset, no further j can overlap a
                # genuine overlap in time
                if a["onset"] == b["onset"] and a["duration"] == b["duration"]:
                    identical_count += 1  # this one WOULD get chorded correctly
                else:
                    overlap_count += 1
                    if len(examples) < 5:
                        examples.append((a, b))

        print(f"\n{label} staff: {len(evs)} notes")
        print(f"  identical-onset+duration overlaps (correctly chorded): {identical_count}")
        print(f"  OTHER time overlaps (NOT chorded -- these are the suspect ones): {overlap_count}")
        for a, b in examples:
            print(f"    e.g. note A onset={a['onset']} dur={a['duration']} pitch={a['pitch']}  "
                  f"vs  note B onset={b['onset']} dur={b['duration']} pitch={b['pitch']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/maestro_subset/manifest.json")
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--composer-filter", default="scarlatti")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subdivision", type=float, default=0.25)
    ap.add_argument("--staff-split-pitch", type=int, default=60)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    scores = load_scores(args.scores)

    entry = pick_recording(manifest, scores, args.composer_filter, args.title)
    name = safe_name(entry)
    midi_path = os.path.join(args.results_dir, f"{name}.mid")

    _, beat_times = estimate_beat_grid(entry["audio_path"])
    events = quantize_notes(midi_path, beat_times, args.subdivision)

    find_overlaps(events, args.staff_split_pitch)


if __name__ == "__main__":
    main()
