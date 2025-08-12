import sys
import os
import json
from pydub import AudioSegment

def split_audio_by_segments(audio_path, segments_json_path, output_dir):
    # Load full translated audio
    audio = AudioSegment.from_wav(audio_path)

    # Load segments data
    with open(segments_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("No 'segments' found in JSON")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, seg in enumerate(segments, 0):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        segment_audio = audio[start_ms:end_ms]

        filename = os.path.join(output_dir, f"segment_{i:03}.wav")
        segment_audio.export(filename, format="wav")
        print(f"Exported segment {i}: {filename} [{seg['start']}s to {seg['end']}s]")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python split_audio_by_segments.py <full_audio.wav> <segments.json> <output_dir>")
        sys.exit(1)

    full_audio_path = sys.argv[1]
    segments_json_path = sys.argv[2]
    output_dir = sys.argv[3]

    split_audio_by_segments(full_audio_path, segments_json_path, output_dir)
