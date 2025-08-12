# import openai
# import subprocess
# from pathlib import Path
import key
import os
import subprocess
from pathlib import Path
from openai import OpenAI

#OpenAI.api_key = key.key

# --- CONFIG ---
VIDEO_FILE = "sources/x-downloader.com_QpsbkR.mp4"
AUDIO_FILE = "temp_audio.wav"
OUTPUT_SRT = "output.srt"

client = OpenAI(api_key=key.key)

# --- 1. Extract audio from video ---
def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", 
        "-ar", "16000", "-ac", "1", audio_path
    ], check=True)

# --- 2. Translate + create subtitles ---
def translate_to_srt(audio_path, output_srt_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-translate",
            file=audio_file,
            response_format="srt"
        )

    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(transcript)

# --- MAIN ---
if __name__ == "__main__":
    if not Path(VIDEO_FILE).exists():
        raise FileNotFoundError(f"Video file '{VIDEO_FILE}' not found")

    print("[1/2] Extracting audio...")
    extract_audio(VIDEO_FILE, AUDIO_FILE)

    print("[2/2] Translating and generating subtitles...")
    translate_to_srt(AUDIO_FILE, OUTPUT_SRT)

    print(f"✅ Subtitles saved to {OUTPUT_SRT}")

    # Cleanup temporary audio
    if Path(AUDIO_FILE).exists():
        os.remove(AUDIO_FILE)

