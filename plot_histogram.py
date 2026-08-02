"""
Renders the semitone_gap_histogram from error_analysis_summary.json as a bar chart.

Usage:
    python plot_histogram.py --results-dir results
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    summary_path = os.path.join(args.results_dir, "error_analysis_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    histogram = summary["semitone_gap_histogram"]
    items = sorted(((int(k), v) for k, v in histogram.items()), key=lambda x: x[0])
    gaps = [g for g, _ in items]
    counts = [c for _, c in items]

    colors = []
    for g in gaps:
        if g == 0:
            colors.append('#d62728')
        elif g in (12, -12):
            colors.append('#1f77b4')
        else:
            colors.append('#aaaaaa')

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(gaps, counts, color=colors, width=0.8)
    ax.set_xlabel('Semitone gap (missed vs. spurious note pitch difference)')
    ax.set_ylabel('Count')
    ax.set_title('Semitone Gap Histogram — Missed/Spurious Note Pitch Errors')
    ax.set_xticks(range(min(gaps), max(gaps) + 1, 2))
    ax.tick_params(axis='x', rotation=45)

    ax.legend(handles=[
        Patch(facecolor='#d62728', label='0 semitones (timing mismatch, e.g. ornaments/trills)'),
        Patch(facecolor='#1f77b4', label='±12 semitones (octave confusion)'),
        Patch(facecolor='#aaaaaa', label='Other'),
    ], loc='upper right', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(args.results_dir, "semitone_gap_histogram.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()