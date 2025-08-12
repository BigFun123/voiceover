import sys
import json
import os
from openai import OpenAI

# Transcribe audio file using OpenAI's Whisper model

def transcribe_audio(audio_path, output_json_path):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json"
        )
    
    # Access attributes instead of dict keys
    print("Full transcript:")
    print(transcript.text)  # .text attribute
    
    # Convert to dict for JSON serialization
    transcript_dict = transcript.model_dump()  # converts dataclass to dict
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(transcript_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Transcription JSON saved to: {output_json_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python transcribe_audio.py <input_audio_file> <output_json_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_json_path = sys.argv[2]
    print(f"Transcribing audio file: {audio_path}")

    transcribe_audio(audio_path, output_json_path)
