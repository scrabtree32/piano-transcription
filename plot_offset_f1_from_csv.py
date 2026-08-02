"""
Renders a sorted, era-colored bar chart of note_with_offset_f1 across all
recordings in scores.csv, annotating known outliers.

Usage:
    python plot_offset_f1.py --results-dir results
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Manual era lookup by composer -- mir_eval/MAESTRO metadata doesn't include
# era/style, so this is asserted here based on standard music history
# periodization, not derived from the data itself.
ERA_BY_COMPOSER = {
    "Domenico Scarlatti": "Baroque",
    "Bach": "Baroque",
    "Wolfgang Amadeus Mozart": "Classical",
    "Sergei Rachmaninoff": "Romantic",
    "Franz Schubert": "Romantic",
    "Frédéric Chopin": "Romantic",
    "Alexander Scriabin": "Romantic",
    "Claude Debussy": "Romantic/Impressionist",
}

COLORS = {
    "Baroque": "#4C72B0",
    "Classical": "#55A868",
    "Romantic": "#C44E52",
    "Romantic/Impressionist": "#C44E52",
}

# Recordings to annotate directly on the chart, keyed by a substring match
# against "composer - title", with the annotation text to display.
ANNOTATIONS = {
    "Scarlatti - Sonata in A Major, K. 208":
        "K. 208: Baroque, but heavy\npedal use in this performance",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--scores-csv", default=None,
                     help="defaults to <results-dir>/scores.csv")
    args = ap.parse_args()

    scores_path = args.scores_csv or os.path.join(args.results_dir, "scores.csv")
    with open(scores_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    data = []
    for r in rows:
        composer = r["composer"]
        era = ERA_BY_COMPOSER.get(composer, "Unknown")
        label = f'{composer.split()[-1]} - {r["title"]}'
        data.append((label, float(r["note_with_offset_f1"]), era))

    data.sort(key=lambda x: x[1])
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [COLORS.get(d[2], "#999999") for d in data]

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(data) + 1))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("note_with_offset_f1")
    ax.set_title("Offset F1 by Recording, Sorted (color = era)", fontweight='bold')
    ax.set_xlim(0, 1.05)

    for label, note_key in ANNOTATIONS.items():
        matches = [i for i, l in enumerate(labels) if label.split(" - ")[1] in l]
        if matches:
            i = matches[0]
            ax.annotate(
                note_key,
                xy=(values[i], i),
                xytext=(min(values[i] + 0.28, 0.75), i - 0.3),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
            )

    present_eras = sorted(set(d[2] for d in data), key=lambda e: list(COLORS).index(e) if e in COLORS else 99)
    legend_elements = [Patch(facecolor=COLORS.get(e, "#999999"), label=e) for e in present_eras]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(args.results_dir, "offset_f1_by_recording.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
