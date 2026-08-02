# Project Notes — Piano Transcription

Running log of observations, decisions, and gotchas as I go. Newest at the bottom of each day's section.

## Day 1 - Environment Set Up

Environment Setup

Discovered that Python 3.14 (default on machine) is incompatible with the ML ecosystem due to removal of pkg_resources — resolved by creating a new virtual environment using Python 3.11 which is the current ML standard
Learned that python -m venv defaults to the highest installed Python version, not necessarily the most compatible one for ML work
Virtual environment must be reactivated with source venv311/Scripts/activate each new terminal session

Dependencies

Installed core stack: piano_transcription_inference, torch, librosa, mir_eval, music21, gradio
Encountered librosa version compatibility issues with the ByteDance package — resolved by using librosa.load() directly instead of the package's built in load_audio() function
ffmpeg required for audio format conversion, installed via winget but requires PATH restart to be recognized

Model

Using ByteDance CRNN model (piano_transcription_inference) — jointly detects note onsets, offsets, frames, and velocity
Model weights (~165MB) must be manually downloaded from Zenodo due to Windows lacking wget by default, stored at ~/piano_transcription_inference_data/
First successful transcription of Bach Prelude No. 1 (BWV 846) to MIDI — output recognizably correct though MIDI playback sounds different from original recording due to synthesized piano sound

General Observations

Python package ecosystem moves slowly relative to new Python releases — using cutting edge Python versions causes significant compatibility friction in ML work
Modular code design (isolating transcription in a single function) will make swapping models easier later if needed
CPU inference is functional but slow — 27 segments processed sequentially for a ~2 minute piece

## Day 2 — MAESTRO eval pipeline

- **MAESTRO's Google Cloud bucket only hosts the full dataset as one zip** (~101GB compressed / 120GB uncompressed), not individual audio/MIDI files at per-recording paths. Every code example I could find that uses this dataset downloads the whole zip and extracts locally — nobody pulls individual files from that bucket directly.
- Worked around this using a community mirror on Hugging Face (`RichardErkhov/maestro`) that hosts the same files individually at matching year/filename paths. Still pull the official CSV metadata from Google (that's a real hosted file, works fine).
- That mirror is user-uploaded, so it may not have every recording. Fixed by querying Hugging Face's API for the full file list first (`GET /api/datasets/{repo_id}`) and only choosing candidate recordings where both audio and MIDI are confirmed present, instead of guessing filenames and hitting 404s.
- Environment gotcha: installed `pretty_midi` once, but into the wrong Python environment (not `venv311`), so the eval script couldn't find it even though I'd "already installed it." Fixed with `python -m pip install pretty_midi` while `venv311` was active, to make sure it goes into the right place.
- CSV writer needs `encoding="utf-8"` explicitly, or accented characters (e.g. "Frédéric") get mangled when written to `scores.csv`.
- **Key finding:** onset detection + pitch (`onset_f1`, `note_f1`) are consistently strong across all 5 test pieces (0.966–0.989) — in line with the ByteDance paper's own reported numbers. But `note_with_offset_f1` (the strict metric requiring onset+offset+pitch match) is wildly inconsistent: ~0.76–0.78 for Scarlatti (Baroque, mostly non-legato) vs. ~0.06–0.08 for Schubert/Chopin (Romantic, heavy pedal/legato). This suggests the model is very good at *when a note starts and what it is*, but *when a note ends* is a much harder, more ambiguous problem — and that difficulty scales with how legato/pedaled the repertoire is. Worth a real paragraph in the write-up, not just a stat.
Filename collision bug: caching by composer_title broke when two MAESTRO recordings shared a title — one's transcription got silently reused for the other. Tell-tale sign: note_f1 and onset_f1 decoupled (one cratered, one stayed fine), which normally doesn't happen. Fixed by keying filenames off the (unique) audio path instead. Lesson: decoupled metrics → suspect a pipeline bug before a model behavior.
Error analysis (15 pieces): 2.8% missed / 1.1% spurious overall, recall balanced across registers (96–98%). Dominant error types: (1) near-duplicate-timing misses on same pitch (ornaments/trills), (2) octave confusion (±12 semitones) — both known transcription failure modes, not random noise.
MuseScore auto-render never worked despite several fixes (chord grouping → voice split → voice renumbering); root cause not found. .musicxml output itself may also be worth sanity-checking later, since even MuseScore's own GUI couldn't display it — that's a bit more concerning than just the automated PDF step failing
Pipeline Testing with Bach: Switched from erratic opening rolls to a full-length performance of J.S. Bach's Prelude in C Major (BWV 846) (test.mp3) to test grid alignment on steady arpeggios.

Diagnosing the Phase Shift: Explored why neural beat-trackers like beat-this often suffer from a slight temporal lag, causing music to start on beat 2 instead of beat 1.

Tackling Sustain Smearing: Discovered how piano pedal resonance causes raw MIDI transcriptions to stretch note durations unnaturally, creating a wall of overlapping ties.

Dynamic Quantization Strategies: Discussed moving away from hardcoded global values toward statistical analysis of the score to automatically determine minimum rhythmic units and handle staff-splitting across the grand staff.

Refining Rhythmic Detection: Realized that looking at raw note durations fails due to the pedal, and established that analyzing the Inter-Onset Intervals (IOIs)—the spacing between consecutive note start times—is the correct way to isolate the piece's true underlying rhythmic pulse.

Custom Metadata Inputs: Expanded the Gradio UI to include optional text boxes for Piece Title and Composer, passing them seamlessly through CLI arguments to embed them directly into the generated MusicXML and PDF scores.

User-Friendly Target Start Beat: Replaced the confusing raw-second offset slider with an intuitive Target Start Beat input, allowing you to explicitly declare which beat number the audio should begin on.

Dynamic Tempo-Based Grid Shifting: Implemented automated backend calculations that use the piece's tempo (beat_interval) to scale and slide the entire rhythmic grid precisely when a target beat is specified.

Event Timestamp Normalization: Added an automated pre-check in build_score to shift raw event onsets cleanly to 0.0, eliminating music21 stream boundary crashes caused by shifted grid offsets.

Neural Transcription Noise & Ghost Notes: * Issue: Frame-level output from piano_transcription_inference combined with sustain pedal resonance and hammer echoes introduced micro-timing jitter.

Impact: When fed into rigid quantization grids, this jitter manifested as phantom short notes (sixteenths/thirty-seconds) that cluttered the sheet music.

Mitigation: Adjusted inter-onset interval (IOI) analysis thresholds (targeting an 80% coverage baseline) to prefer coarser rhythmic subdivisions and prevent over-quantization.

Rubato vs. Rigid Grid Alignment: * Issue: Expressive human timing variations (rubato), particularly in Romantic repertoire like Chopin's Prelude in E minor, caused severe tempo fluctuations.

Impact: The downbeat transformer (beat_this) and mathematical bar-line grids suffered from phase drift, causing measures to split mid-phrase and breaking standard metric assumptions.

Target Start Beat (Anacrusis) Offset Drift: * Issue: Pieces starting on upbeats or non-standard downbeats required explicit grid offsetting.

Impact: While the underlying mathematical offset shift executed, warping an elastic rubato performance onto a rigid grid often just shifted the bar line break to a different incorrect location rather than resolving the phrasing.

Key Signature Estimation Failures on Chromatic Works: * Issue: music21's automatic key signature detection algorithm relies heavily on diatonic note distributions. Highly chromatic works with extensive passing tones and diminished harmonies (e.g., Chopin) caused incorrect auto-detections.

Mitigation: Implemented a manual key signature dropdown override in the Gradio UI and linked it to the backend quantize_to_musicxml.py script to force correct key signatures when auto-detection fails.