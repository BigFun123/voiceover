import sys
import json
import os
from openai import OpenAI

# translates segments from a JSON file using OpenAI's API
# input JSON should have a "segments" key with "text", "start", and "end" fields
# output JSON will have the same structure with translated text

def translate_segments(input_json_path, output_json_path, target_language="en"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("No 'segments' found in input JSON")
        return

    translated_segments = []

    print(f"Translating {len(segments)} segments to {target_language}...")

    for i, segment in enumerate(segments, 1):
        original_text = segment.get("text", "")
        if not original_text.strip():
            translated_text = ""
        else:
            messages = [
                {"role": "system", "content": f"You are a helpful assistant that translates text into {target_language}."},
                {"role": "user", "content": original_text},
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
            )
            translated_text = response.choices[0].message.content.strip()

        translated_segment = {
            "id": segment.get("id", i),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": translated_text
        }
        translated_segments.append(translated_segment)
        print(f"Segment {i}/{len(segments)} translated.")

    output_data = {
        "segments": translated_segments,
        "language": target_language
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Translated segments saved to {output_json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python translate_segments.py <input_json> <output_json> [target_language]")
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2]
    target_lang = sys.argv[3] if len(sys.argv) > 3 else "en"

    translate_segments(input_json, output_json, target_lang)
