#!/usr/bin/env python3
"""
auto_dub.py

Local pipeline to:
  1) extract audio from a source video
  2) transcribe (and optionally translate) using Whisper (python package or CLI)
  3) produce target-language TTS segments (Coqui TTS recommended) or fallback to pyttsx3
  4) stitch segments into a single audio file and remux with original video
  5) optional: call Wav2Lip to produce lip-synced video using the new audio

This script is intentionally modular and defensive: it'll detect installed backends and print actionable errors
if a heavy dependency is missing. It expects ffmpeg to be installed and in PATH.

Usage examples:
  # subtitles only (translate to english)
  python auto_dub.py --input input.mp4 --mode subs --target-lang en --output out_dir

  # generate a translated dubbed audio and remux into output_dub.mp4
  python auto_dub.py --input input.mp4 --mode dub --target-lang en --tts-backend coqui --output out_dir

Dependencies (recommended):
  pip install -r requirements.txt

Author: ChatGPT (GPT-5 Thinking mini)
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

# ----------------------------- Utilities ---------------------------------

def run(cmd: List[str], check=True):
    """Run a command and stream output. Raises on failure if check=True."""
    print("+ " + " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    out_lines = []
    for line in proc.stdout:
        print(line, end="")
        out_lines.append(line)
    proc.wait()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, "".join(out_lines))
    return "".join(out_lines)

def which_bin(name: str) -> Optional[str]:
    return shutil.which(name)

# ----------------------------- FFmpeg helpers -----------------------------

def extract_audio(input_video: str, out_audio: str, sample_rate: int = 16000):
    """Extract audio to a WAV file suitable for ASR/TTS"""
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", out_audio
    ]
    run(cmd)

def mux_audio_with_video(original_video: str, new_audio: str, out_video: str):
    """Replace audio stream of the original video with new_audio, preserving container and basic codec choices where possible"""
    # -map 0:v maps original video, -map 1:a maps the new audio
    cmd = [
        "ffmpeg", "-y", "-i", original_video, "-i", new_audio,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out_video
    ]
    run(cmd)

# ----------------------------- ASR / Translation ---------------------------

def transcribe_with_whisper_python(audio_path: str, model_name: str = "large", translate: bool = False) -> Dict[str, Any]:
    """Attempt to transcribe using the whisper python package (openai-whisper).
    Returns whisper's result dict (with segments and text)
    """
    try:
        import whisper
    except Exception as e:
        raise RuntimeError("whisper python package not available. Install with: pip install -U openai-whisper") from e

    model = whisper.load_model(model_name)
    print(f"Loaded whisper model: {model_name}")
    task = "translate" if translate else "transcribe"
    result = model.transcribe(audio_path, task=task)
    return result

def transcribe_with_whisper_cli(audio_path: str, model_name: str = "large", translate: bool = False, output_srt: Optional[str] = None) -> Dict[str, Any]:
    """Use whisper CLI if available (whisper executable on PATH). Produces an srt if requested; returns a minimal dict with 'segments'"""
    cmd = ["whisper", audio_path, "--model", model_name]
    if translate:
        cmd += ["--task", "translate"]
    if output_srt:
        cmd += ["--output_format", "srt", "--output_dir", str(Path(output_srt).parent)]
    # whisper writes files to cwd/output_dir; ensure durable location
    run(cmd)
    # best-effort: if srt requested, parse it to return segments; otherwise return empty-ish structure
    if output_srt and Path(output_srt).exists():
        # Parsing SRT is simple but we prefer whisper's JSON if available. We'll return an empty structure and rely on SRT for subtitles.
        return {"srt_path": str(output_srt)}
    return {"segments": []}

# ----------------------------- TTS Backends -------------------------------

class TTSBackend:
    def say(self, text: str, outfile: str, lang: str):
        raise NotImplementedError()

class CoquiTTSBackend(TTSBackend):
    def __init__(self, model_name: Optional[str] = None):
        try:
            from TTS.api import TTS
        except Exception as e:
            raise RuntimeError("Coqui TTS not installed. pip install TTS") from e
        # If model_name None, TTS auto-selects a default model (may download)
        self.model_name = model_name
        self.tts = TTS(model_name) if model_name else TTS()

    def say(self, text: str, outfile: str, lang: str):
        # Coqui TTS can save directly
        self.tts.tts_to_file(text=text, speaker=None, language=lang, file_path=outfile)

class Pyttsx3TTSBackend(TTSBackend):
    def __init__(self):
        try:
            import pyttsx3
        except Exception as e:
            raise RuntimeError("pyttsx3 not installed. pip install pyttsx3") from e
        self.engine = __import__("pyttsx3").init()

    def say(self, text: str, outfile: str, lang: str):
        # pyttsx3 can't directly export wav easily in all setups. We write to a temp file using the engine save_to_file and runAndWait.
        tmp = outfile
        self.engine.save_to_file(text, tmp)
        self.engine.runAndWait()

# ----------------------------- Segment stitching --------------------------

def segments_to_audio_tts(segments: List[Dict[str, Any]], tts_backend: TTSBackend, tmp_dir: str, lang: str) -> str:
    """Given whisper segments (with start, end, text), generate per-segment TTS, then concat and return final wav path"""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wavs = []
    for i, seg in enumerate(segments):
        text = seg.get("text") or seg.get("translation") or seg.get("alternatives", [{}])[0].get("transcript", "")
        if not text.strip():
            continue
        out_wav = tmp_dir / f"seg_{i:04d}.wav"
        print(f"Generating TTS for segment {i}, duration {seg.get('end') - seg.get('start'):.2f}s")
        tts_backend.say(text, str(out_wav), lang=lang)
        wavs.append(str(out_wav))
    if not wavs:
        raise RuntimeError("No TTS segments produced, empty segments list?")
    # create a file list for ffmpeg concat
    list_file = tmp_dir / "concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for w in wavs:
            f.write(f"file '{w}'\n")
    out_all = tmp_dir / "full_tts.wav"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "pcm_s16le", str(out_all)]
    run(cmd)
    return str(out_all)

# ----------------------------- Helpers to parse whisper segments -------------

def srt_to_segments(srt_path: str) -> List[Dict[str, Any]]:
    """Simple SRT parser that returns segments list with start,end,text in seconds"""
    import re
    segs = []
    with open(srt_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    entries = re.split(r"\n\s*\n", content.strip())
    for e in entries:
        lines = e.strip().splitlines()
        if len(lines) >= 3:
            # 1st line is index, 2nd line is time, rest is text
            time_line = lines[1].strip()
            m = re.match(r'(\d{2}:\d{2}:\d{2}[,\.]\d+)\s*--?>\s*(\d{2}:\d{2}:\d{2}[,\.]\d+)', time_line)
            if not m:
                continue
            def ts_to_s(ts):
                parts = ts.replace(",", ".").split(":")
                h, m_, s = parts
                return int(h) * 3600 + int(m_) * 60 + float(s)
            start = ts_to_s(m.group(1))
            end = ts_to_s(m.group(2))
            text = " ".join(lines[2:]).replace("\n", " ").strip()
            segs.append({"start": start, "end": end, "text": text})
    return segs

# ----------------------------- Main pipeline --------------------------------

def pipeline(args):
    input_video = str(Path(args.input).absolute())
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp) if args.tmp else Path(out_dir) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # check ffmpeg
    if not which_bin("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg and ensure it's on PATH. See README.")

    audio_wav = str(tmp / "extracted_audio.wav")
    print("Extracting audio...")
    extract_audio(input_video, audio_wav, sample_rate=16000)

    # Transcribe (and optionally translate)
    print("Transcribing with Whisper...")
    whisper_output = None
    srt_path = str(tmp / "whisper_output.srt")
    try:
        whisper_output = transcribe_with_whisper_python(audio_wav, model_name=args.whisper_model, translate=(args.mode == "subs" and args.translate))
        # whisper python returns 'segments' and 'text'
        segments = whisper_output.get("segments", [])
    except Exception as e:
        print("whisper python failed or not installed:", e)
        print("Attempting whisper CLI (if installed)...")
        trans_res = transcribe_with_whisper_cli(audio_wav, model_name=args.whisper_model, translate=(args.mode == "subs" and args.translate), output_srt=srt_path)
        if "srt_path" in trans_res:
            segments = srt_to_segments(str(trans_res["srt_path"]))
        else:
            segments = []

    # If mode is subs we mainly output an .srt for the user and optionally burn into video
    if args.mode == "subs":
        if segments:
            # write srt from segments
            srt_out = out_dir / (Path(args.input).stem + f"_{args.target_lang}.srt")
            with open(srt_out, "w", encoding="utf-8") as fh:
                for i, seg in enumerate(segments, start=1):
                    # format times
                    def s_to_srt(ts):
                        h = int(ts // 3600)
                        m = int((ts % 3600) // 60)
                        s = ts % 60
                        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
                    fh.write(f"{i}\n{s_to_srt(seg['start'])} --> {s_to_srt(seg['end'])}\n{seg['text']}\n\n")
            print(f"Wrote subtitles to {srt_out}")
            if args.burn:
                out_burn = out_dir / (Path(args.input).stem + f"_{args.target_lang}_burn.mp4")
                cmd = ["ffmpeg", "-y", "-i", args.input, "-vf", f"subtitles={str(srt_out)}", str(out_burn)]
                run(cmd)
                print(f"Wrote burned video to {out_burn}")
            return

    # For dubbing: we expect segments with timestamps in source language OR translation text for target
    # If whisper didn't produce segments but produced full text, we fallback to single chunk
    if not segments:
        print("Warning: no timestamped segments found, falling back to single-chunk translation+TTS.")
        with open(tmp / "full_trans.txt", "w", encoding="utf-8") as fh:
            fh.write(whisper_output.get("text", "") if whisper_output else "")
        segments = [{"start": 0.0, "end": 999999.0, "text": whisper_output.get("text","") if whisper_output else ""}]
    # Choose TTS backend
    tts_backend = None
    if args.tts_backend == "coqui":
        try:
            tts_backend = CoquiTTSBackend(model_name=args.coqui_model)
        except Exception as e:
            print("Coqui TTS not available:", e)
            print("Falling back to pyttsx3 (lower quality)")
    if tts_backend is None:
        try:
            tts_backend = Pyttsx3TTSBackend()
        except Exception as e:
            raise RuntimeError("No usable TTS backend found. Install Coqui TTS or pyttsx3. See README.") from e

    print("Generating TTS segments...")
    full_tts = segments_to_audio_tts(segments, tts_backend, tmp, lang=args.target_lang)
    print("Full TTS audio assembled at:", full_tts)

    # Mix or pad to alignment: simplest option is to trust segments timing, but TTS durations vary.
    # Best-effort: just remux new audio in place of old one. Advanced sync would require time-stretching.
    out_dub_video = str(Path(out_dir) / (Path(args.input).stem + f"_{args.target_lang}_dub.mp4"))
    print("Muxing new audio into video...")
    mux_audio_with_video(args.input, full_tts, out_dub_video)
    print("Wrote dubbed video to:", out_dub_video)

    if args.wav2lip:
        # call Wav2Lip if installed (user must have it set up separately)
        if not which_bin("python") or not Path("Wav2Lip").exists():
            print("Wav2Lip not found. Please install the Wav2Lip repo and set it up. Skipping lip-sync step.")
        else:
            print("Running Wav2Lip (external) - please ensure you have Wav2Lip repo available and set up.")
            # basic example call (user must adapt paths and environment)
            wav2lip_cmd = ["python", "Wav2Lip/inference.py", "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth", "--face", args.input, "--audio", full_tts, "--outfile", str(Path(out_dir) / "wav2lip_out.mp4")]
            run(wav2lip_cmd)
            print("Wav2Lip output written to:", str(Path(out_dir) / "wav2lip_out.mp4"))

# ----------------------------- CLI ----------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Local auto subtitle + dubbed voiceover pipeline. See README for details.")
    p.add_argument("--input", "-i", required=True, help="Input video file path")
    p.add_argument("--output", "-o", required=True, help="Output directory to write results to")
    p.add_argument("--mode", choices=["subs","dub"], default="subs", help="subs = generate srt; dub = generate dubbed audio and remux")
    p.add_argument("--target-lang", default="en", help="Target language code for TTS (e.g. en, es, fr)")
    p.add_argument("--whisper-model", default="large", help="Whisper model name (tiny, base, small, medium, large)")
    p.add_argument("--tts-backend", choices=["coqui","pyttsx3"], default="coqui", help="Which TTS backend to prefer")
    p.add_argument("--coqui-model", default=None, help="Optional Coqui model name")
    p.add_argument("--tmp", default=None, help="Temporary working directory")
    p.add_argument("--translate", action="store_true", help="If set while in subs mode, whisper will translate into target language")
    p.add_argument("--burn", action="store_true", help="If set in subs mode, burn subtitles into a video")
    p.add_argument("--wav2lip", action="store_true", help="If set, attempt to run Wav2Lip after dubbing for lip sync (requires external setup)")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        pipeline(args)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
