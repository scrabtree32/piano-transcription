import os
import sys
import re
import subprocess
import gradio as gr
from transcribe import transcribe

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def safe_name(audio_path):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    return re.sub(r"[^A-Za-z0-9_-]", "_", base)

def process_audio(audio_file, custom_title, custom_composer, beat_input):
    if audio_file is None:
        return "Please upload an audio file first.", None, None
    
    os.makedirs("results", exist_ok=True)
    base_name = safe_name(audio_file)
    
    midi_output = os.path.join("results", f"{base_name}.mid")
    musicxml_output = os.path.join("results", f"{base_name}.musicxml")
    pdf_output = os.path.join("results", f"{base_name}.pdf")
    
    try:
        print(f"Transcribing {audio_file}...")
        transcribe(audio_file, midi_output)
        
        print(f"Quantizing and rendering score...")
        cmd = [
            sys.executable, "quantize_to_musicxml.py",
            "--manifest", "data/maestro_subset/manifest.json",
            "--audio-path", audio_file,
            "--out", musicxml_output
        ]
        
        # Append optional user overrides if provided
        if custom_title and custom_title.strip():
            cmd.extend(["--title", custom_title.strip()])
        if custom_composer and custom_composer.strip():
            cmd.extend(["--composer", custom_composer.strip()])
        # Ensure it passes the target beat parameter instead
        if beat_input:
            cmd.extend(["--target-beat", str(int(beat_input))])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Quantization Error: {result.stderr}", None, None
            
        if not os.path.exists(musicxml_output):
            return f"Error: Expected output file {musicxml_output} was not found after quantization.", None, None
            
        status_msg = f"Successfully processed {base_name} with custom parameters!"
        return status_msg, musicxml_output, pdf_output

    except Exception as e:
        return f"An error occurred: {str(e)}", None, None

with gr.Blocks() as demo:
    gr.Markdown("# 🎹 AI Piano Transcription & Sheet Music Renderer")
    gr.Markdown("Upload an audio recording, specify optional metadata, and fine-tune downbeat tracking.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Piano Audio")
            
            with gr.Row():
                title_input = gr.Textbox(label="Piece Title (Optional)", placeholder="e.g., Prelude in C Major")
                composer_input = gr.Textbox(label="Composer (Optional)", placeholder="e.g., J.S. Bach")
                
            beat_input = gr.Number(
                label="Target Start Beat", 
                value=1, 
                precision=0,
                info="Which beat number should the piece actually start on? (e.g., 1 for downbeat, 2 for an upbeat/second beat)"
            )
            
            submit_btn = gr.Button("Transcribe & Quantize", variant="primary")
            
        with gr.Column():
            status_output = gr.Textbox(label="Status")
            musicxml_file = gr.File(label="Download MusicXML")
            pdf_file = gr.File(label="Download PDF Score")
            
    submit_btn.click(
        fn=process_audio, 
        inputs=[audio_input, title_input, composer_input, beat_input], 
        outputs=[status_output, musicxml_file, pdf_file]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())