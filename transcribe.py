import librosa
import soundfile as sf
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

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
    transcribe("test.mp3", "output.mid")