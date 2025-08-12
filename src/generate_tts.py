import sys
import os
import json
from openai import OpenAI

def generate_tts_per_segment(input_json_path, output_dir, model="gpt-4o-mini-tts", voice="onyx"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments")
    if not segments:
        print("No segments found in JSON.")
        return

    for i, segment in enumerate(segments, start=1):
        segment_text = segment.get("text", "").strip()
        if not segment_text:
            print(f"Skipping empty segment {i}")
            continue

        print(f"Segment {i} text preview:")
        print(segment_text[:200])

        # Call TTS API for the segment text
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=segment_text
        )

        audio_bytes = response.read()

        # Filename with zero-padded segment number
        audio_filename = os.path.join(output_dir, f"segment_{i:03d}.wav")
        with open(audio_filename, "wb") as out_f:
            out_f.write(audio_bytes)

        print(f"Saved segment {i} audio to: {audio_filename}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_tts.py <input_translated_json> <output_audio_directory>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_dir = sys.argv[2]

    generate_tts_per_segment(input_json, output_dir)
