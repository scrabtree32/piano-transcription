"""
Day 2 stretch goal: error analysis on top of the F1 scores.

Rather than just aggregate precision/recall/F1, this looks at *what kind*
of mistakes the model makes:
  - Missed notes (in the ground truth, but the model didn't find them)
  - Spurious notes (the model invented a note that isn't in the ground truth)
  - Octave errors: when a missed note has a near-simultaneous estimated
    note exactly 12 semitones away (classic transcription failure mode --
    the model got the "chroma" right but the register wrong)
  - Per-register performance: is the model worse in the bass, mid, or
    treble range?
  - Per-composer performance: raw miss/spurious rates don't separate
    cleanly by composer/era, but near-miss error SEVERITY does -- Baroque
    (Scarlatti) near-misses stay within about an octave, while Romantic/
    Impressionist pieces (Chopin, Rachmaninoff, Debussy, Scriabin) produce
    near-misses several octaves off. This grouping makes that visible
    instead of eyeballing it out of console output.

Requires: same manifest.json produced by download_maestro_subset_v2.py,
and the transcribed MIDI files already produced by run_evaluation_v2.py
in the --results-dir folder (defaults to "results", matching that script).

Usage:
    python error_analysis.py --manifest data/maestro_subset/manifest.json --results-dir results
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import librosa
import numpy as np
import pretty_midi
from mir_eval import transcription as mir_transcription

# Register buckets by MIDI pitch number (60 = middle C).
REGISTERS = [
    ("bass", 0, 55),
    ("mid", 56, 76),
    ("treble", 77, 127),
]


# Must match the naming scheme in run_evaluation.py exactly, so we look up
# the transcription file that actually corresponds to each recording.
# Derived from the audio filename (unique per recording) rather than
# composer+title, which can collide when MAESTRO has multiple recordings
# of the same piece.
def safe_name(entry):
    base = os.path.splitext(os.path.basename(entry["audio_path"]))[0]
    return re.sub(r"[^A-Za-z0-9_-]", "_", base)


# Loads a MIDI file and returns intervals (seconds), pitches in Hz (for
# mir_eval matching), and pitches as raw MIDI note numbers (for reporting
# semitone differences and register buckets without Hz rounding issues).
def load_midi_notes(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    intervals, midi_pitches = [], []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            intervals.append([note.start, note.end])
            midi_pitches.append(note.pitch)

    if not intervals:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=int)

    intervals = np.array(intervals, dtype=float)
    midi_pitches = np.array(midi_pitches, dtype=int)
    pitches_hz = librosa.midi_to_hz(midi_pitches.astype(float))
    return intervals, pitches_hz, midi_pitches


# Returns which register bucket ("bass"/"mid"/"treble") a MIDI pitch falls into.
def register_of(pitch):
    for name, lo, hi in REGISTERS:
        if lo <= pitch <= hi:
            return name
    return "mid"


# Matches ref notes to est notes on onset+pitch only (ignoring offset, since
# offset is a separate, already-documented problem -- see project_notes.md).
# Then classifies every ref note as matched/missed, every est note as
# matched/spurious, and for missed notes, checks whether an unmatched est
# note lands at nearly the same time but a different pitch (a "near miss"),
# recording the semitone gap -- this is how we detect octave errors etc.
def analyze_pair(ref_intervals, ref_pitches_hz, ref_midi_pitches,
                  est_intervals, est_pitches_hz, est_midi_pitches):
    matches = mir_transcription.match_notes(
        ref_intervals, ref_pitches_hz, est_intervals, est_pitches_hz,
        offset_ratio=None,  # ignore offsets -- onset + pitch only
    )
    matched_ref = {m[0] for m in matches}
    matched_est = {m[1] for m in matches}

    missed_idx = [i for i in range(len(ref_midi_pitches)) if i not in matched_ref]
    spurious_idx = [i for i in range(len(est_midi_pitches)) if i not in matched_est]

    near_miss_semitone_gaps = []
    for mi in missed_idx:
        ref_onset = ref_intervals[mi, 0]
        ref_pitch = ref_midi_pitches[mi]
        # look for any unmatched estimated note starting within 100ms
        best = None
        for ei in spurious_idx:
            est_onset = est_intervals[ei, 0]
            if abs(est_onset - ref_onset) <= 0.1:
                gap = est_midi_pitches[ei] - ref_pitch
                if best is None or abs(gap) < abs(best):
                    best = gap
        if best is not None:
            near_miss_semitone_gaps.append(best)

    per_register = {name: {"ref_total": 0, "ref_matched": 0} for name, _, _ in REGISTERS}
    for i, pitch in enumerate(ref_midi_pitches):
        reg = register_of(pitch)
        per_register[reg]["ref_total"] += 1
        if i in matched_ref:
            per_register[reg]["ref_matched"] += 1

    return {
        "n_ref_notes": len(ref_midi_pitches),
        "n_est_notes": len(est_midi_pitches),
        "n_missed": len(missed_idx),
        "n_spurious": len(spurious_idx),
        "near_miss_gaps": near_miss_semitone_gaps,
        "per_register": per_register,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/maestro_subset/manifest.json")
    ap.add_argument("--results-dir", default="results",
                     help="folder where run_evaluation.py wrote the transcribed .mid files")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    all_gaps = Counter()
    total_missed = total_spurious = total_ref = total_est = 0
    register_totals = {name: {"ref_total": 0, "ref_matched": 0} for name, _, _ in REGISTERS}
    per_recording_rows = []

    # Per-composer running totals: note counts plus every near-miss gap seen,
    # so we can report both rate (missed/spurious %) and severity (how far
    # off the near-misses were) broken out by composer.
    per_composer = defaultdict(lambda: {
        "ref_total": 0, "est_total": 0, "missed": 0, "spurious": 0, "gaps": []
    })

    for entry in manifest:
        name = safe_name(entry)
        est_midi_path = os.path.join(args.results_dir, f"{name}.mid")
        if not os.path.exists(est_midi_path):
            print(f"Skipping {entry['title']} -- no transcribed MIDI found at {est_midi_path} "
                  f"(run run_evaluation.py first)")
            continue

        ref_intervals, ref_hz, ref_midi = load_midi_notes(entry["midi_path"])
        est_intervals, est_hz, est_midi = load_midi_notes(est_midi_path)

        result = analyze_pair(ref_intervals, ref_hz, ref_midi, est_intervals, est_hz, est_midi)

        print(f"\n{entry['composer']} - {entry['title']}")
        print(f"  ref notes: {result['n_ref_notes']}  est notes: {result['n_est_notes']}")
        print(f"  missed: {result['n_missed']}  spurious: {result['n_spurious']}")
        if result["near_miss_gaps"]:
            print(f"  near-miss semitone gaps: {result['near_miss_gaps']}")

        all_gaps.update(result["near_miss_gaps"])
        total_missed += result["n_missed"]
        total_spurious += result["n_spurious"]
        total_ref += result["n_ref_notes"]
        total_est += result["n_est_notes"]
        for reg in register_totals:
            register_totals[reg]["ref_total"] += result["per_register"][reg]["ref_total"]
            register_totals[reg]["ref_matched"] += result["per_register"][reg]["ref_matched"]

        composer = entry["composer"]
        c = per_composer[composer]
        c["ref_total"] += result["n_ref_notes"]
        c["est_total"] += result["n_est_notes"]
        c["missed"] += result["n_missed"]
        c["spurious"] += result["n_spurious"]
        c["gaps"].extend(result["near_miss_gaps"])

        per_recording_rows.append({
            "composer": entry["composer"],
            "title": entry["title"],
            "ref_notes": result["n_ref_notes"],
            "est_notes": result["n_est_notes"],
            "missed": result["n_missed"],
            "spurious": result["n_spurious"],
        })

    print("\n=== Aggregate across all recordings ===")
    print(f"Total ref notes: {total_ref}   Total est notes: {total_est}")
    print(f"Total missed: {total_missed} ({100*total_missed/total_ref:.1f}% of ref notes)")
    print(f"Total spurious: {total_spurious} ({100*total_spurious/total_est:.1f}% of est notes)")

    print("\nPer-register recall (of ground-truth notes, how many did the model find):")
    for reg in register_totals:
        t = register_totals[reg]["ref_total"]
        m = register_totals[reg]["ref_matched"]
        if t > 0:
            print(f"  {reg}: {m}/{t} = {100*m/t:.1f}%")
        else:
            print(f"  {reg}: no notes in this range")

    print("\nSemitone-gap histogram for missed notes with a same-time near-miss "
          "(±12 = octave error, ±7 = fifth, ±1 = semitone slip):")
    for gap, count in sorted(all_gaps.items(), key=lambda x: -x[1]):
        print(f"  {gap:+d} semitones: {count}")

    # Per-composer summary: rate (miss/spurious %) tends to be noisy
    # piece-to-piece and doesn't cleanly separate by era on a small sample.
    # Near-miss SEVERITY (max/mean absolute semitone gap) is the metric that
    # actually shows a composer/era pattern, so report both rather than
    # just rate alone.
    print("\n=== Per-composer summary ===")
    composer_rows = []
    for composer, c in sorted(per_composer.items()):
        miss_pct = 100 * c["missed"] / c["ref_total"] if c["ref_total"] else None
        spur_pct = 100 * c["spurious"] / c["est_total"] if c["est_total"] else None
        abs_gaps = [abs(g) for g in c["gaps"]]
        max_gap = max(abs_gaps) if abs_gaps else None
        mean_gap = sum(abs_gaps) / len(abs_gaps) if abs_gaps else None

        print(f"{composer}:")
        print(f"  ref notes: {c['ref_total']}   missed: {miss_pct:.1f}%   spurious: {spur_pct:.1f}%")
        if abs_gaps:
            print(f"  near-miss gaps: n={len(abs_gaps)}  max={max_gap}  mean={mean_gap:.1f} semitones")
        else:
            print("  near-miss gaps: none")

        composer_rows.append({
            "composer": composer,
            "ref_notes": c["ref_total"],
            "est_notes": c["est_total"],
            "missed_pct": round(miss_pct, 2) if miss_pct is not None else None,
            "spurious_pct": round(spur_pct, 2) if spur_pct is not None else None,
            "n_near_misses": len(abs_gaps),
            "max_abs_semitone_gap": max_gap,
            "mean_abs_semitone_gap": round(mean_gap, 2) if mean_gap is not None else None,
        })

    # Save the aggregate stats (register recall, semitone histogram,
    # per-composer breakdown) to files too -- these were previously only
    # printed to the terminal and easy to lose track of once the console
    # scrolls away.
    summary_path = os.path.join(args.results_dir, "error_analysis_summary.json")
    summary = {
        "total_ref_notes": total_ref,
        "total_est_notes": total_est,
        "total_missed": total_missed,
        "total_spurious": total_spurious,
        "missed_pct": 100 * total_missed / total_ref,
        "spurious_pct": 100 * total_spurious / total_est,
        "per_register_recall": {
            reg: {
                "ref_total": register_totals[reg]["ref_total"],
                "ref_matched": register_totals[reg]["ref_matched"],
                "recall_pct": (100 * register_totals[reg]["ref_matched"] / register_totals[reg]["ref_total"]
                               if register_totals[reg]["ref_total"] > 0 else None),
            }
            for reg in register_totals
        },
        "semitone_gap_histogram": {int(gap): count for gap, count in sorted(all_gaps.items(), key=lambda x: -x[1])},
        "per_composer": composer_rows,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregate summary written to {summary_path}")

    out_csv = os.path.join(args.results_dir, "error_analysis.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["composer", "title", "ref_notes", "est_notes", "missed", "spurious"])
        writer.writeheader()
        writer.writerows(per_recording_rows)
    print(f"Per-recording breakdown written to {out_csv}")

    composer_csv = os.path.join(args.results_dir, "error_analysis_by_composer.csv")
    with open(composer_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["composer", "ref_notes", "est_notes", "missed_pct", "spurious_pct",
                      "n_near_misses", "max_abs_semitone_gap", "mean_abs_semitone_gap"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(composer_rows)
    print(f"Per-composer breakdown written to {composer_csv}")


if __name__ == "__main__":
    main()
