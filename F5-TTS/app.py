from flask import Flask, request, jsonify, send_file
import subprocess
import os
import uuid

# For Whisper transcription
from faster_whisper import WhisperModel

app = Flask(__name__)

INFER_SCRIPT = "src/f5_tts/infer/infer_cli.py"
OUTPUT_DIR = "tests"
VOICES_DIR = "voices"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

# Initialize whisper model once, use for all requests, use GPU (A6000) for whisper
# WHISPER_MODEL = WhisperModel("large-v3", device="cuda", compute_type="float16")
# Uncomment the below line if you get Cuda errors and force use CPU (much slower)
WHISPER_MODEL = WhisperModel("base", device="cpu")

def transcribe_with_whisper(audio_path):
    """Transcribe audio using Whisper (faster-whisper)."""
    segments, info = WHISPER_MODEL.transcribe(audio_path)
    transcript = "".join([segment.text for segment in segments])
    return transcript

def get_unique_filename(base_dir, base_name):
    name, ext = os.path.splitext(base_name)
    candidate = os.path.join(base_dir, base_name)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{name}_{i}{ext}")
        i += 1
    return candidate

@app.route('/clone', methods=['POST'])
def clone_voice():
    try:
        ref_audio = request.files['ref_audio']
        gen_text = request.form['gen_text']
        ref_text = request.form.get('ref_text', '').strip()

        # New slider parameters from frontend
        speed = request.form.get('speed', 1.0)
        nfe_step = request.form.get('nfe_step', 32)
        cross_fade_duration = request.form.get('cross_fade_duration', 0.15)

        # Validate or convert slider values
        try:
            speed = float(speed)
            nfe_step = int(nfe_step)
            cross_fade_duration = float(cross_fade_duration)
        except Exception as e:
            return jsonify({"error": "Invalid slider parameters."}), 400

        unique_id = str(uuid.uuid4())
        audio_path = f"{VOICES_DIR}/{unique_id}_ref.wav"
        ref_audio.save(audio_path)

        if not ref_text:
            ref_text = transcribe_with_whisper(audio_path)

        ref_text_path = f"{VOICES_DIR}/{unique_id}_ref.txt"
        with open(ref_text_path, "w", encoding="utf-8") as f:
            f.write(ref_text)

        orig_filename = os.path.splitext(ref_audio.filename)[0]
        out_base = f"{orig_filename}_clone.wav"
        output_path = get_unique_filename(OUTPUT_DIR, out_base)

        command = [
            "python3", INFER_SCRIPT,
            "--gen_text", gen_text,
            "--ref_audio", audio_path,
            "--ref_text", ref_text_path,
            "--output_dir", OUTPUT_DIR,
            "--output_file", os.path.basename(output_path),
            "--speed", str(speed),
            "--nfe_step", str(nfe_step),
            "--cross_fade_duration", str(cross_fade_duration)
        ]

        subprocess.run(command, check=True)

        try:
            os.remove(audio_path)
        except Exception as e:
            print(f"Warning: Could not delete temp audio file {audio_path}: {e}")
        try:
            os.remove(ref_text_path)
        except Exception as e:
            print(f"Warning: Could not delete temp text file {ref_text_path}: {e}")

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6969, debug=True)
