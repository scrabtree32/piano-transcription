import argparse
import os
import re
import librosa
import soundfile as sf
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

def safe_name(audio_path):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    return re.sub(r"[^A-Za-z0-9_-]", "_", base)

def transcribe(audio_path, output_midi_path):
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Load model (downloads weights automatically on first run)
    transcriptor = PianoTranscription(device='cpu')
    
    # Transcribe
    transcribed_dict = transcriptor.transcribe(audio, output_midi_path)
    
    print(f"Transcription complete! MIDI saved to {output_midi_path}")
    return transcribed_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to input audio file")
    parser.add_argument("--output", default=None,
                         help="path to write output MIDI file; defaults to "
                              "results/<audio_filename>.mid to match quantize_to_musicxml.py's "
                              "expected naming")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        os.makedirs(args.results_dir, exist_ok=True)
        output_path = os.path.join(args.results_dir, f"{safe_name(args.input)}.mid")

    transcribe(args.input, output_path)