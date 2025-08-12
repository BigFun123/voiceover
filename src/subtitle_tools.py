import re
from typing import List, Dict

def format_timestamp(seconds: float) -> str:
    """Convert seconds (float) to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def segments_to_srt(segments: List[Dict]) -> str:
    """
    Convert list of segments to SRT formatted string.
    Each segment dict should have: 'index', 'start', 'end', 'text'
    """
    srt_entries = []
    for seg in segments:
        idx = seg.get("index")
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text").strip().replace("\n", " ")
        srt_entry = f"{idx}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n"
        srt_entries.append(srt_entry)
    return "\n".join(srt_entries)

def write_srt_file(srt_text: str, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(srt_text)

def parse_srt_file(filepath: str) -> List[Dict]:
    """
    Parse an SRT file into segments list.
    Returns list of dict with keys: index(int), start(float), end(float), text(str)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+([\s\S]*?)(?=\n\n|\Z)", re.MULTILINE)
    segments = []

    for match in pattern.finditer(content):
        idx = int(match.group(1))
        start_ts = match.group(2)
        end_ts = match.group(3)
        text = match.group(4).strip()

        # Convert timestamps to seconds
        def ts_to_seconds(ts):
            h, m, s_ms = ts.split(":")
            s, ms = s_ms.split(",")
            return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
        
        segments.append({
            "index": idx,
            "start": ts_to_seconds(start_ts),
            "end": ts_to_seconds(end_ts),
            "text": text
        })
    return segments
