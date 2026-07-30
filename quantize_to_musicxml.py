"""
Day 3: convert a transcribed MIDI file into MusicXML with AI downbeat tracking.

Features:
  - Deep-learning downbeat & beat tracking via all-in-one-infer
  - Auto-calculates meter (3/4 vs 4/4) from detected downbeats
  - Key signature parsing from title & note analysis
  - MusicXML & PDF export via MuseScore
"""

import argparse
import csv
import json
import os
import re
import subprocess

import numpy as np
import pretty_midi
from music21 import environment, key, metadata, meter, note, stream
from music21 import tempo as m21tempo
from music21.musicxml.m21ToXml import GeneralObjectExporter
from beat_this.inference import File2Beats


def safe_name(entry):
    base = os.path.splitext(os.path.basename(entry["audio_path"]))[0]
    return re.sub(r"[^A-Za-z0-9_-]", "_", base)


def load_scores(scores_csv):
    if not os.path.exists(scores_csv):
        return []
    with open(scores_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_recording(manifest, scores, composer_filter, title=None):
    candidates = [e for e in manifest if composer_filter.lower() in e["composer"].lower()]
    if title:
        candidates = [e for e in candidates if e["title"] == title]

    if not candidates:
        raise SystemExit(f"No manifest entries match composer filter '{composer_filter}'"
                         + (f" and title '{title}'" if title else ""))

    if len(candidates) == 1:
        return candidates[0]

    score_lookup = {(r["composer"], r["title"]): r for r in scores}
    ranked = []
    for e in candidates:
        row = score_lookup.get((e["composer"], e["title"]))
        f1 = float(row["note_with_offset_f1"]) if row else -1.0
        ranked.append((f1, e))
    ranked.sort(key=lambda x: -x[0])

    print(f"{len(candidates)} matches for '{composer_filter}':")
    for f1, e in ranked:
        f1_str = f"{f1:.3f}" if f1 >= 0 else "no score found"
        print(f"  - {e['title']}  (note_with_offset_f1: {f1_str})")
    print(f"Using best-scoring match: {ranked[0][1]['title']}\n")

    return ranked[0][1]


def parse_meter_from_title(title):
    """Checks if meter is explicitly written in the title (e.g. 3/4, 3/8, 6/8)."""
    match = re.search(r"\b([23469]|12)/([248])\b", title)
    if match:
        return match.group(0)
    return None


def estimate_beat_grid_and_meter(audio_path, title="", user_time_sig="auto", midi_path=None):
    print(f"Running beat-this classical tracking on:\n  {audio_path}")
    
    file2beats = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = file2beats(audio_path)

    beat_times = np.array(beats)
    downbeat_times = np.array(downbeats)

    if len(beat_times) < 2:
        raise SystemExit("Beat tracking failed to detect enough beats in audio.")

    avg_beat_dur = np.median(np.diff(beat_times))

    # --- AUTOMATIC PHASE CORRECTION (Fixes the "starts on beat 2" bug) ---
    if midi_path and os.path.exists(midi_path):
        pm = pretty_midi.PrettyMIDI(midi_path)
        all_notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
        if all_notes:
            first_note_time = min(n.start for n in all_notes)
            diff = beat_times[0] - first_note_time
            # If the model lagged by roughly one beat interval, shift grid back
            if 0.5 * avg_beat_dur <= diff <= 1.5 * avg_beat_dur:
                print(f"Detected ~1-beat phase lag ({diff:.3f}s). Shifting beat grid backward.")
                beat_times = beat_times - avg_beat_dur
                downbeat_times = downbeat_times - avg_beat_dur
                avg_beat_dur = np.median(np.diff(beat_times))

    tempo_val = 60.0 / avg_beat_dur

    # Safe padding that avoids shape/dimension mismatch errors
    start_padding = beat_times[0] - avg_beat_dur
    end_padding = beat_times[-1] + avg_beat_dur
    extended_beat_times = np.insert(beat_times, 0, start_padding)
    extended_beat_times = np.append(extended_beat_times, end_padding)

    # 2. Determine Time Signature... (keep the rest of your time sig logic unchanged)
    if user_time_sig != "auto":
        final_time_sig = user_time_sig
    else:
        title_sig = parse_meter_from_title(title)
        if title_sig:
            final_time_sig = title_sig
            print(f"Time Signature (parsed from title): {final_time_sig}")
        elif len(downbeat_times) > 2:
            beats_per_bar = len(beat_times) / len(downbeat_times)
            print(f"Classical AI detected ratio: {beats_per_bar:.2f} beats/bar")
            if 2.4 <= beats_per_bar < 3.4:
                final_time_sig = "3/8" if "presto" in title.lower() or "allegro" in title.lower() else "3/4"
            elif 5.5 <= beats_per_bar < 6.5:
                final_time_sig = "6/8"
            else:
                final_time_sig = "4/4"
        else:
            final_time_sig = "4/4"

    print(f"Tempo: {tempo_val:.1f} BPM | Final Time Signature: {final_time_sig}")
    return tempo_val, extended_beat_times, final_time_sig

def time_to_beat(t, beat_times):
    idx = np.searchsorted(beat_times, t) - 1
    idx = int(np.clip(idx, 0, len(beat_times) - 2))
    b0, b1 = beat_times[idx], beat_times[idx + 1]
    frac = (t - b0) / (b1 - b0) if b1 > b0 else 0.0
    return idx + frac


def analyze_ioi(midi_path, beat_times, candidate_subdivisions=(1.0, 0.5, 1/3, 0.25, 1/6, 0.125, 1/12, 1/16),
                 tolerance=0.15, coverage_threshold=0.85):
    """
    Auto-detect the piece's minimum rhythmic subdivision and a sane note-
    duration cap from the *onsets* alone -- deliberately ignoring note-off
    times, since sustain pedal smears those but leaves onset timing intact.

    Approach:
      1. Collapse near-simultaneous onsets (chords) into single onset times,
         so IOIs reflect melodic rhythm rather than chord-voicing jitter.
      2. Convert consecutive onset gaps ("IOIs") into beat units.
      3. For each candidate subdivision, from coarsest to finest, check what
         fraction of IOIs land close to an integer multiple of it. Take the
         coarsest subdivision that explains most of the IOIs -- this avoids
         over-fitting to noise/expressive timing by defaulting to 64th notes.
      4. Suggest a max_note_duration from the IOI distribution itself
         (rather than a fixed 2.0), so the duration cap scales with the
         piece's actual tempo/note density instead of a one-size constant.

    Returns (chosen_subdivision, suggested_max_duration, diagnostics).
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    onset_times = sorted({n.start for inst in pm.instruments if not inst.is_drum for n in inst.notes})

    if len(onset_times) < 3:
        print("IOI analysis: too few onsets detected; falling back to defaults.")
        return 0.25, 2.0, {"reason": "too few onsets"}

    onset_beats = np.array([time_to_beat(t, beat_times) for t in onset_times])

    # Collapse onsets that land within ~1/64 of a beat of each other --
    # these are effectively simultaneous (chord notes), not separate rhythmic events.
    collapsed = [onset_beats[0]]
    for b in onset_beats[1:]:
        if b - collapsed[-1] > (1.0 / 64):
            collapsed.append(b)
    collapsed = np.array(collapsed)

    iois = np.diff(collapsed)
    iois = iois[iois > 1e-3]

    if len(iois) == 0:
        print("IOI analysis: no usable positive IOIs found; falling back to defaults.")
        return 0.25, 2.0, {"reason": "no positive IOIs"}

    diagnostics = {}
    chosen_subdivision = candidate_subdivisions[-1]  # finest, as a fallback
    for s in sorted(candidate_subdivisions, reverse=True):  # coarse -> fine
        ratios = iois / s
        nearest = np.round(ratios)
        nearest[nearest == 0] = 1  # guard against tiny IOIs / div-by-zero
        rel_error = np.abs(ratios - nearest) / nearest
        coverage = float(np.mean(rel_error <= tolerance))
        diagnostics[f"coverage_at_{round(s, 4)}"] = round(coverage, 3)
        if coverage >= coverage_threshold:
            chosen_subdivision = s
            break

    # Cap sustained notes at roughly the longest plausible single note implied
    # by the piece's own rhythm: a small multiple of the detected grid, or
    # 2x the 90th-percentile IOI, whichever is larger -- so a slow Adagio
    # doesn't get its half-notes clipped, but pedal smear in a fast Presto
    # still gets reined in.
    ioi_p90 = float(np.percentile(iois, 90))
    suggested_max_duration = max(chosen_subdivision * 4, ioi_p90 * 2)

    diagnostics.update({
        "n_onsets": len(onset_times),
        "n_iois_after_chord_collapse": len(iois),
        "ioi_median_beats": float(np.median(iois)),
        "ioi_p90_beats": ioi_p90,
        "chosen_subdivision": chosen_subdivision,
        "suggested_max_duration": suggested_max_duration,
    })

    print(f"IOI analysis: {len(onset_times)} onsets -> {len(iois)} inter-onset gaps "
          f"(median {diagnostics['ioi_median_beats']:.3f} beats)")
    print(f"  Detected minimum rhythmic unit: {chosen_subdivision:.4f} quarterLength "
          f"(coverage {diagnostics.get(f'coverage_at_{round(chosen_subdivision, 4)}', 0):.2f})")
    print(f"  Suggested max_note_duration: {suggested_max_duration:.3f} quarterLength")

    return chosen_subdivision, suggested_max_duration, diagnostics


def quantize_notes(midi_path, beat_times, subdivision, max_note_duration=2.0):
    pm = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for n in instrument.notes:
            onset_beat = time_to_beat(n.start, beat_times)
            offset_beat = time_to_beat(n.end, beat_times)

            q_onset = round(onset_beat / subdivision) * subdivision
            q_offset = round(offset_beat / subdivision) * subdivision
            
            duration = q_offset - q_onset
            # Cap maximum duration to avoid pedal smearing/endless ties
            if duration > max_note_duration:
                duration = max_note_duration
            if duration <= 0:
                duration = subdivision

            events.append({"pitch": n.pitch, "onset": q_onset, "duration": duration})
    return events

def determine_key_signature(score, key_arg="auto", title=""):
    if key_arg and key_arg.lower() != "auto":
        try:
            k = key.Key(key_arg)
            print(f"Key Signature (specified via CLI): {k.name}")
            return k
        except Exception as ex:
            print(f"Could not parse CLI key '{key_arg}' ({ex}); switching to auto-detect.")

    if title:
        match = re.search(r"\bin\s+([A-G][b#♯♭]?)\s+(major|minor)\b", title, re.IGNORECASE)
        if match:
            pitch_str, mode = match.group(1), match.group(2).lower()
            key_str = pitch_str.lower() if mode == "minor" else pitch_str.upper()
            try:
                k = key.Key(key_str)
                print(f"Key Signature (parsed from title): {k.name}")
                return k
            except Exception:
                pass

    try:
        detected_key = score.analyze('key')
        print(f"Key Signature (auto-detected from notes): {detected_key.name}")
        return detected_key
    except Exception as ex:
        print(f"Key analysis failed ({ex}); defaulting to C Major.")
        return key.Key('C')


def build_score(events, staff_split_pitch, time_sig, key_arg="auto", title="", composer=""):
    treble_events = [e for e in events if e["pitch"] >= staff_split_pitch]
    bass_events = [e for e in events if e["pitch"] < staff_split_pitch]

    score = stream.Score()

    md = metadata.Metadata()
    if title:
        md.title = title
    if composer:
        md.composer = composer
    score.metadata = md

    for label, evs in (("treble", treble_events), ("bass", bass_events)):
        part = stream.Part(id=label)
        part.insert(0, meter.TimeSignature(time_sig))

        for e in evs:
            el = note.Note(e["pitch"])
            el.duration.quarterLength = e["duration"]
            part.insert(e["onset"], el)

        part.makeVoices(inPlace=True, fillGaps=True)
        score.insert(0, part)

    target_key = determine_key_signature(score, key_arg=key_arg, title=title)
    for part in score.parts:
        part.insert(0, target_key)
        part.makeNotation(inPlace=True)

    return score


def find_musescore_path():
    us = environment.UserSettings()
    for k in ("musescoreDirectPNGPath", "musicxmlPath"):
        try:
            val = us[k]
        except Exception:
            val = None
        if val and os.path.exists(str(val)):
            return str(val)
    return None


def export_with_fixed_voices(score, out_path):
    exporter = GeneralObjectExporter()
    xml_bytes = exporter.parse(score)
    xml_str = xml_bytes.decode("utf-8")

    fixed_xml, count = re.subn(
        r"<voice>(\d+)</voice>",
        lambda m: f"<voice>{int(m.group(1)) + 1}</voice>",
        xml_str,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(fixed_xml)

    print(f"\nMusicXML written to {out_path} (bumped {count} <voice> tags)")


def try_render_pdf(musicxml_path):
    musescore_path = find_musescore_path()
    if not musescore_path:
        print("MuseScore not configured -- skipping PDF render.")
        return

    pdf_path = os.path.splitext(musicxml_path)[0] + ".pdf"
    try:
        subprocess.run([musescore_path, "-o", pdf_path, musicxml_path], check=True)
        print(f"Rendered notation -> {pdf_path}")
    except Exception as ex:
        print(f"PDF rendering failed: {ex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/maestro_subset/manifest.json")
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--composer-filter", default="scarlatti")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subdivision", default="auto",
                     help="quarterLength grid size (e.g. '0.25'), or 'auto' to detect via IOI analysis")
    ap.add_argument("--max-note-duration", default="auto",
                     help="quarterLength cap on note length (e.g. '2.0'), or 'auto' to derive from IOI analysis")
    ap.add_argument("--time-signature", default="auto", help="e.g. '3/4', '3/8', '4/4', or 'auto'")
    ap.add_argument("--key", default="auto", help="e.g. 'd', 'D', or 'auto'")
    ap.add_argument("--staff-split-pitch", type=int, default=60)
    ap.add_argument("--out", default="results/quantized_output.musicxml")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    scores = load_scores(args.scores)

    entry = pick_recording(manifest, scores, args.composer_filter, args.title)
    name = safe_name(entry)
    midi_path = os.path.join(args.results_dir, f"{name}.mid")
    if not os.path.exists(midi_path):
        raise SystemExit(f"No transcribed MIDI found at {midi_path}")

    entry_title = entry.get("title", "")
    entry_composer = entry.get("composer", "")

    print(f"\nUsing: {entry_composer} - {entry_title}")
    print(f"  audio: {entry['audio_path']}")
    print(f"  transcribed MIDI: {midi_path}")

    tempo_val, beat_times, time_sig = estimate_beat_grid_and_meter(
        entry["audio_path"], title=entry_title, user_time_sig=args.time_signature, midi_path=midi_path
    )

    if args.subdivision == "auto" or args.max_note_duration == "auto":
        detected_subdivision, detected_max_duration, _ = analyze_ioi(midi_path, beat_times)
    else:
        detected_subdivision = detected_max_duration = None

    subdivision = float(args.subdivision) if args.subdivision != "auto" else detected_subdivision
    max_note_duration = (float(args.max_note_duration) if args.max_note_duration != "auto"
                          else detected_max_duration)

    events = quantize_notes(midi_path, beat_times, subdivision, max_note_duration=max_note_duration)
    print(f"Quantized {len(events)} notes to {subdivision}-quarterLength grid "
          f"(max_note_duration={max_note_duration:.3f})")

    score = build_score(
        events,
        args.staff_split_pitch,
        time_sig,
        key_arg=args.key,
        title=entry_title,
        composer=entry_composer,
    )
    score.insert(0, m21tempo.MetronomeMark(number=round(tempo_val)))

    export_with_fixed_voices(score, args.out)
    try_render_pdf(args.out)


if __name__ == "__main__":
    main()