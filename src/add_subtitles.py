import sys
import subprocess
from pathlib import Path


def burn_subtitles(video_path:Path, srt_path:Path, output_path:Path):
    #srt_abs = srt_path.resolve()
    #srt_for_filter = str(srt_abs).replace("\\", "/")
    #filter_arg = f"subtitles='{srt_for_filter}'"
    srt_path = srt_path.replace("\\", "/")
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}:force_style='FontName=Arial'",
        "-c:a", "copy",
        output_path
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def add_soft_subtitles(video_path, srt_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(srt_path),
        "-c", "copy",
        "-c:s", "mov_text",
        str(output_path)
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python subtitle_tool.py <burn|soft> <video_path> <srt_path> <output_path>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    video_path = sys.argv[2]
    srt_path = sys.argv[3]
    output_path = sys.argv[4]

    if mode == "burn":
        burn_subtitles(video_path, srt_path, output_path)
    elif mode == "soft":
        add_soft_subtitles(video_path, srt_path, output_path)
    else:
        print("Invalid mode. Use 'burn' to hardcode subtitles or 'soft' to add soft subtitles.")
        sys.exit(1)
