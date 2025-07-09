from flask import Flask, request, jsonify, send_file
import subprocess
import os
import uuid
import json
import requests
import re
import random
import sys
from pydub import AudioSegment
from faster_whisper import WhisperModel

app = Flask(__name__)

INFER_SCRIPT = "src/f5_tts/infer/infer_cli.py"
OUTPUT_DIR = "tests"
VOICES_DIR = "voices"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

# ---------------------- TTS/CLONE CONFIG -------------------------
WHISPER_MODEL = WhisperModel("large-v3-turbo", device="cuda")

def transcribe_with_whisper(audio_path):
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

def shift_pitch(input_path, output_path, octave_shift):
    audio = AudioSegment.from_file(input_path)
    new_sample_rate = int(audio.frame_rate * (2.0 ** octave_shift))
    shifted_audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    shifted_audio = shifted_audio.set_frame_rate(16000)
    shifted_audio.export(output_path, format="wav")

# ---------------------- GRADING CONFIG ---------------------------
RUBRIC_PATH = 'rubric.json'
LLAMA_URL = "http://localhost:11434/api/chat"
LLAMA_MODEL = "phi3"
WHISPER_MODEL_NAME = "large-v3-turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float32"

WHISPER_MODEL_GRADING = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)

with open(RUBRIC_PATH, 'r') as f:
    RUBRIC = json.load(f)

def transcribe_audio(audio_path):
    segments, info = WHISPER_MODEL_GRADING.transcribe(audio_path)
    return ''.join(segment.text for segment in segments)

def extract_gold_script(llama_output):
    # Try delimiters first
    pattern = r"---GOLD SCRIPT START---([\s\S]*?)---GOLD SCRIPT END---"
    match = re.search(pattern, llama_output)
    if match:
        return match.group(1).strip()
    # If code block present, extract inside
    match = re.search(r"```(?:plaintext)?\n([\s\S]*?)```", llama_output)
    if match:
        return match.group(1).strip()
    return None


def grade_audio_with_llama(transcript):
    rubric_text = json.dumps(RUBRIC, indent=2)
    prompt = f"""
You are a speaking coach. ONLY rewrite the EXACT text in the provided transcript, making improvements to grammar, flow, clarity, and natural speech. 
Do NOT invent or change topics. Do NOT summarize. Do NOT create a speech about a different subject. 
Add punctuation for natural speech. Insert [pause] where you would pause. 
Output only the improved script, and nothing else, between the following delimiters (do not output a code block):

---GOLD SCRIPT START---
[YOUR IMPROVED SCRIPT HERE]
---GOLD SCRIPT END---

Here is the transcript:
\"\"\"
{transcript}
\"\"\"
"""
    llama_response = requests.post(
        LLAMA_URL,
        json={
            "model": LLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )

    result = llama_response.json()
    llama_output = result['message']['content']
    print("RAW LLAMA OUTPUT:", llama_output)

    script = extract_gold_script(llama_output)
    if not script:
        raise ValueError(f"Could not extract gold-starred script from Llama output. Raw: {llama_output}")
    return script

def process_pauses(script):
    return script.replace('[pause]', '.')

# ---------------------- FLEXIBLE CLONE ROUTE ----------------------------

@app.route('/clone', methods=['POST'])
@app.route('/clone', methods=['POST'])
def clone_voice():
    try:
        ref_audio = request.files['ref_audio']
        ref_text = request.form.get('ref_text', '').strip()
        user_gen_text = request.form.get('gen_text', '').strip()

        speed = request.form.get('speed', 1.0)
        nfe_step = request.form.get('nfe_step', 32)
        cross_fade_duration = request.form.get('cross_fade_duration', 0.15)
        pitch_shift = request.form.get('pitch_shift', 0.0)
        randomize_seed = request.form.get('randomize_seed', 'true').lower() in ['true', 'on', '1']
        seed = request.form.get('seed', None)

        try:
            speed = float(speed)
            nfe_step = int(nfe_step)
            cross_fade_duration = float(cross_fade_duration)
            pitch_shift = float(pitch_shift)
        except Exception as e:
            return jsonify({"error": "Invalid slider parameters."}), 400

        unique_id = str(uuid.uuid4())
        audio_path = f"{VOICES_DIR}/{unique_id}_ref.wav"
        ref_audio.save(audio_path)

        if user_gen_text:
            gen_text = user_gen_text
        else:
            transcript = transcribe_audio(audio_path)
            print("WHISPER TRANSCRIPT:", transcript)   # <---- ADDED DEBUG LINE
            gen_text = grade_audio_with_llama(transcript)
            if not gen_text or not gen_text.strip():
                raise ValueError("No improved script found in Llama output!")
            gen_text = process_pauses(gen_text)
            print("USING GEN_TEXT FOR CLONE:", gen_text)

        if not ref_text:
            ref_text = transcribe_with_whisper(audio_path)

        ref_text_path = f"{VOICES_DIR}/{unique_id}_ref.txt"
        with open(ref_text_path, "w", encoding="utf-8") as f:
            f.write(ref_text)

        if randomize_seed:
            seed_value = random.randint(0, 2**32 - 1)
        else:
            if seed is not None and str(seed).strip() != '':
                seed_value = int(seed)
            else:
                seed_value = random.randint(0, 2**32 - 1)

        orig_filename = os.path.splitext(ref_audio.filename)[0]
        out_base = f"{orig_filename}_clone.wav"
        output_path = get_unique_filename(OUTPUT_DIR, out_base)

        command = [
            sys.executable, INFER_SCRIPT,
            "--gen_text", gen_text,
            "--ref_audio", audio_path,
            "--ref_text", ref_text_path,
            "--output_dir", OUTPUT_DIR,
            "--output_file", os.path.basename(output_path),
            "--speed", str(speed),
            "--nfe_step", str(nfe_step),
            "--cross_fade_duration", str(cross_fade_duration)
            # Uncomment --seed if your infer_cli.py supports it
            # "--seed", str(seed_value)
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath("src")

        subprocess.run(command, check=True, env=env)

        if pitch_shift != 0.0:
            shifted_output_path = output_path.replace(".wav", "_pitch_shifted.wav")
            shift_pitch(output_path, shifted_output_path, pitch_shift)
            os.remove(output_path)
            output_path = shifted_output_path

        try: os.remove(audio_path)
        except Exception as e: print(f"Warning: Could not delete temp audio file {audio_path}: {e}")
        try: os.remove(ref_text_path)
        except Exception as e: print(f"Warning: Could not delete temp text file {ref_text_path}: {e}")

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/grade', methods=['POST'])
def grade_audio():
    try:
        audio = request.files['audio']
        unique_id = str(uuid.uuid4())
        audio_path = f"{VOICES_DIR}/{unique_id}_grade.wav"
        audio.save(audio_path)

        transcript = transcribe_audio(audio_path)
        
        print("AUDIO PATH:", audio_path)
        print("WHISPER TRANSCRIPT:", transcript)
        
        gold_script = grade_audio_with_llama(transcript)

        try:
            os.remove(audio_path)
        except Exception as e:
            print(f"Warning: Could not delete temp audio file {audio_path}: {e}")

        return jsonify({
            "transcript": transcript,
            "gold_starred_script": gold_script
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6969, debug=True)
