"""
Day 2: transcribe the MAESTRO test subset and compute note-level /
onset-level F1 scores with mir_eval.

Requires:
    pip install pretty_midi   (in addition to your existing venv311 packages)

Usage:
    python run_evaluation.py --manifest data/maestro_subset/manifest.json --out results

Produces:
    results/<recording>.mid   -- model's transcription for each recording
    results/scores.csv        -- per-recording + averaged metrics
"""

import argparse
import csv
import json
import os
import re

import librosa
import numpy as np
import pretty_midi
from mir_eval import transcription as mir_transcription

from transcribe import transcribe


# Builds a filesystem-safe, guaranteed-unique name for a recording's output
# files. Using composer+title is NOT safe -- MAESTRO has multiple different
# recordings/performances that share the exact same title (e.g. two
# recordings both titled "Etude Op. 10 No. 12"), which collide and cause one
# recording's transcription to silently get reused for a different one. The
# original audio filename is unique per recording, so derive the output
# name from that instead.
def safe_name(entry):
    base = os.path.splitext(os.path.basename(entry["audio_path"]))[0]
    return re.sub(r"[^A-Za-z0-9_-]", "_", base)


# Reads a MIDI file and pulls out every note as an (onset, offset) time
# interval plus a pitch. mir_eval wants pitches in Hz rather than MIDI note
# numbers (60, 61, ...), so we convert with librosa.midi_to_hz before
# returning. Used on both the ground-truth MIDI and the model's output MIDI,
# so the two can be compared apples-to-apples.
def load_midi_notes(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    intervals = []
    pitches_midi = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            intervals.append([note.start, note.end])
            pitches_midi.append(note.pitch)

    if not intervals:
        return np.zeros((0, 2)), np.zeros(0)

    intervals = np.array(intervals, dtype=float)
    pitches_hz = librosa.midi_to_hz(np.array(pitches_midi, dtype=float))
    return intervals, pitches_hz


# Loads a ground-truth MIDI file and the model's transcribed MIDI file,
# then hands both note lists to mir_eval.transcription.evaluate(), which
# does the actual matching (aligning estimated notes to reference notes
# within onset/pitch/offset tolerances) and returns a dict of precision/
# recall/F1 scores at a few different strictness levels.
def score_pair(ref_midi_path, est_midi_path):
    ref_intervals, ref_pitches = load_midi_notes(ref_midi_path)
    est_intervals, est_pitches = load_midi_notes(est_midi_path)

    # mir_eval.transcription.evaluate returns a dict with, among others:
    #   Precision, Recall, F-measure                 (onset + offset + pitch)
    #   Precision_no_offset, Recall_no_offset, F-measure_no_offset  (onset + pitch)
    #   Onset_Precision, Onset_Recall, Onset_F-measure              (onset only)
    scores = mir_transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches)
    return scores


# Ties it together: for each recording in the manifest, run the existing
# model (transcribe()) to produce an estimated MIDI file, score it against
# that recording's ground-truth MIDI, print running results, and at the end
# write everything to scores.csv plus print averages across the subset.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/maestro_subset/manifest.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    os.makedirs(args.out, exist_ok=True)

    all_rows = []
    for i, entry in enumerate(manifest):
        name = safe_name(entry)
        est_midi_path = os.path.join(args.out, f"{name}.mid")

        print(f"\n[{i+1}/{len(manifest)}] Transcribing: {entry['composer']} - {entry['title']}")
        if os.path.exists(est_midi_path):
            print(f"  already transcribed -> {est_midi_path} (skipping)")
        else:
            transcribe(entry["audio_path"], est_midi_path)

        print("  Scoring against ground truth...")
        scores = score_pair(entry["midi_path"], est_midi_path)

        row = {
            "composer": entry["composer"],
            "title": entry["title"],
            "duration": entry["duration"],
            "note_f1": scores["F-measure_no_offset"],
            "note_precision": scores["Precision_no_offset"],
            "note_recall": scores["Recall_no_offset"],
            "onset_f1": scores["Onset_F-measure"],
            "onset_precision": scores["Onset_Precision"],
            "onset_recall": scores["Onset_Recall"],
            "note_with_offset_f1": scores["F-measure"],
        }
        all_rows.append(row)
        print(f"  note F1 (onset+pitch): {row['note_f1']:.3f}   onset F1: {row['onset_f1']:.3f}")

    # write CSV
    csv_path = os.path.join(args.out, "scores.csv")
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # print averages
    print("\n=== Averages across subset ===")
    for key in ["note_f1", "note_precision", "note_recall",
                "onset_f1", "onset_precision", "onset_recall",
                "note_with_offset_f1"]:
        avg = sum(r[key] for r in all_rows) / len(all_rows)
        print(f"  {key}: {avg:.3f}")

    print(f"\nPer-recording scores written to {csv_path}")


if __name__ == "__main__":
    main()