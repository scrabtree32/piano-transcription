# piano-transcription

End-to-end piano transcription system — audio to MIDI and sheet music using deep learning.

Built on a pretrained CRNN (Kong et al., 2021) for note/pedal transcription and a beat-tracking transformer (`beat_this`, Foscarin et al., 2024) for rhythmic alignment. Full pipeline: audio → MIDI (transcription) → quantized MusicXML → rendered PDF sheet music.

## Setup

1. **Python version:** Requires Python 3.11. Python 3.14 (a common current default) is incompatible with this ML stack due to the removal of `pkg_resources`. Create the environment explicitly with 3.11 rather than relying on `python -m venv`'s default:
   ```bash
   python3.11 -m venv venv311
   ```
2. **Activate the environment** (repeat this every new terminal session):
   ```bash
   # Windows
   venv311\Scripts\activate
   # Mac/Linux
   source venv311/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **ffmpeg:** required for audio format conversion. Install via your platform's package manager (e.g. `winget install ffmpeg` on Windows) and restart your terminal so it's recognized on PATH.
5. **MuseScore 4:** required for PDF rendering (music21 shells out to it via subprocess). Install from musescore.org, then configure its path once via:

python -c "from music21 import environment; us = environment.UserSettings(); us['musicxmlPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'; us['musescoreDirectPNGPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'"

(adjust the path to wherever MuseScore is actually installed). Without this, MusicXML output is still generated correctly, but PDF export will fail silently.
6. **Pretrained model weights (~165MB):** downloaded automatically via download_weights.py (run this once before any transcription script) — this avoids the underlying package's Windows-incompatible wget call.

## Usage

### 1. Transcribe audio to MIDI
```bash
python transcribe.py --input data/test_audio/bach_prelude_short.wav
```

### 2. Quantize to MusicXML / rendered PDF
```bash
python quantize_to_musicxml.py --audio-path data/test_audio/bach_prelude_short.wav --out results/score.musicxml
```
Optional flags: `--title`, `--composer` (embedded in the output score), `--target-start-beat` (for pieces beginning on an upbeat), `--key-signature` (manual override if auto-detection misreads a chromatic passage — see report Section 5 for why this exists).

### 3. (Optional) Try it via the Gradio UI
```bash
python app.py
```
This launches a local web UI (Gradio will print a URL, typically `http://127.0.0.1:7860`, to open in your browser). Upload an audio file, optionally fill in Title/Composer and a Target Start Beat if the piece begins on an upbeat, and the app will run the full transcribe → quantize → render pipeline and return the MIDI, MusicXML, and PDF outputs.

## Reproducing the evaluation results

The formal evaluation (MAESTRO test subset, error analysis, and figures used in the report) is reproducible via:

1. **Build the MAESTRO test subset + manifest:**
   ```bash
   python download_maestro_subset_v2.py
   ```
   This pulls audio/MIDI pairs from the Hugging Face mirror (`RichardErkhov/maestro`) rather than Google's official ~101GB single-zip bucket, verifying via the Hugging Face API that both audio and MIDI exist for each candidate recording before selecting it. Produces `data/maestro_subset/manifest.json`.

2. **Run transcription + scoring on the subset:**
   ```bash
   python run_evaluation_v2.py --manifest data/maestro_subset/manifest.json --out results
   ```
   Produces `results/scores.csv` (per-recording onset F1 / note F1 / note_with_offset_f1 via `mir_eval`).

3. **Run error analysis:**
   ```bash
   python error_analysis.py --manifest data/maestro_subset/manifest.json --results-dir results
   ```
   Produces `results/error_analysis.csv`, `results/error_analysis_by_composer.csv`, and `results/error_analysis_summary.json` (aggregate missed/spurious rates, per-register recall, semitone-gap histogram).

4. **Generate report figures:**
   ```bash
   python plot_histogram.py --results-dir results
   python plot_offset_f1_from_csv.py --results-dir results
   ```
   Produces `results/semitone_gap_histogram.png` and `results/offset_f1_by_recording.png`.

## Known limitations

See the full report (Sections 5–6) for a detailed discussion. Briefly: offset/duration accuracy is sensitive to sustain pedal use and doesn't cleanly separate by composer/era; rubato can cause bar-line misplacement in the rendered score; and `music21`'s automatic key-signature detection can misread highly chromatic passages (manual override available via `--key-signature`). The MAPS dataset was evaluated for use in generalization testing but was not accessible — its registration system rejected account confirmation with no working login path, a known issue also reported by other users of the dataset.

PDF export via MuseScore's command-line interface can occasionally fail with an opaque exit code, because MuseScore's headless mode has no way to respond to its own 'file contains errors, open anyway?' confirmation dialog. If this happens, the .musicxml file can still be opened manually in MuseScore (clicking 'Open Anyway' if prompted) and exported to PDF from the GUI.