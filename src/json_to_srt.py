import sys
import json
import re

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def segments_to_srt(segments):
    srt_entries = []
    for idx, seg in enumerate(segments, start=1):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip().replace("\n", " ")
        srt_entry = f"{idx}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n"
        srt_entries.append(srt_entry)
    return "\n".join(srt_entries)

def main(input_json_path, output_srt_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments")
    if not segments:
        print(f"No 'segments' key found in {input_json_path}")
        sys.exit(1)

    srt_text = segments_to_srt(segments)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    print(f"SRT subtitles written to: {output_srt_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_srt.py <input_transcript.json> <output_subtitles.srt>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
