import os
import sys  # 1. Import sys
import subprocess
import gradio as gr
from transcribe import transcribe

def process_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file first.", None, None
    
    os.makedirs("results", exist_ok=True)
    
    original_filename = os.path.basename(audio_file)
    base_name = os.path.splitext(original_filename)[0]
    
    midi_output = os.path.join("results", f"{base_name}.mid")
    musicxml_output = os.path.join("results", f"{base_name}.musicxml")
    pdf_output = os.path.join("results", f"{base_name}.pdf")
    
    try:
        # Step 1: Run audio-to-MIDI transcription
        print(f"Transcribing {audio_file}...")
        transcribe(audio_file, midi_output)
        
        # Step 2: Run quantization script using the active venv interpreter
        print(f"Quantizing and rendering score...")
        cmd = [
            sys.executable, "quantize_to_musicxml.py",  # 2. Uses venv Python instead of global Python
            "--manifest", "data/maestro_subset/manifest.json",
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
    gr.Markdown("Upload an audio recording (WAV/MP3) from your MAESTRO subset to generate interactive sheet music and a PDF.")
    
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