"""

Features:
  - Deep-learning downbeat & beat tracking via all-in-one-infer
  - Auto-calculates meter (3/4 vs 4/4) from detected downbeats
  - Key signature parsing from title & note analysis
  - **Dynamic fallback metadata support for arbitrary custom audio uploads**
  - MusicXML & PDF export via MuseScore
"""
import os
import sys
import re
import subprocess
import gradio as gr
from transcribe import transcribe

# Force UTF-8 encoding for terminal logging
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def safe_name(audio_path):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    return re.sub(r"[^A-Za-z0-9_-]", "_", base)

def process_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file first.", None, None
    
    os.makedirs("results", exist_ok=True)
    
    # Use safe_name so app.py and quantize_to_musicxml.py match filenames
    base_name = safe_name(audio_file)
    
    midi_output = os.path.join("results", f"{base_name}.mid")
    musicxml_output = os.path.join("results", f"{base_name}.musicxml")
    pdf_output = os.path.join("results", f"{base_name}.pdf")
    
    try:
        # Step 1: Run audio-to-MIDI transcription
        print(f"Transcribing {audio_file}...")
        transcribe(audio_file, midi_output)
        
        # Step 2: Run quantization script using the matched sanitized paths
        print(f"Quantizing and rendering score...")
        cmd = [
            sys.executable, "quantize_to_musicxml.py",
            "--manifest", "data/maestro_subset/manifest.json",
            "--audio-path", audio_file,
            "--out", musicxml_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Quantization Error: {result.stderr}", None, None
            
        if not os.path.exists(musicxml_output):
            return f"Error: Expected output file {musicxml_output} was not found after quantization.", None, None
            
        status_msg = f"Successfully processed {base_name}!"
        return status_msg, musicxml_output, pdf_output

    except Exception as e:
        return f"An error occurred: {str(e)}", None, None

with gr.Blocks() as demo:
    gr.Markdown("# 🎹 AI Piano Transcription & Sheet Music Renderer")
    gr.Markdown("Upload an audio recording (WAV/MP3) to generate interactive sheet music and a PDF.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Piano Audio")
            submit_btn = gr.Button("Transcribe & Quantize", variant="primary")
            
        with gr.Column():
            status_output = gr.Textbox(label="Status")
            musicxml_file = gr.File(label="Download MusicXML")
            pdf_file = gr.File(label="Download PDF Score")
            
    submit_btn.click(
        fn=process_audio, 
        inputs=audio_input, 
        outputs=[status_output, musicxml_file, pdf_file]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())