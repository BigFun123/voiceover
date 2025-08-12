#!/usr/bin/env python3
"""
openai_dub_stepwise.py

Interactive, step-by-step pipeline for:
  - verifying ffmpeg
  - checking OpenAI API access & available models
  - extracting audio
  - test-transcribing a short clip (STT)
  - optional translation (per-segment)
  - optional TTS generation (test + full)
  - muxing voiceover into video and optional burning of subtitles

Usage:
  python openai_dub_stepwise.py -i input.mp4 -o out_dir --target-lang en

Notes:
- Requires openai>=1.0.0 (new client), ffmpeg in PATH, Python 3.10+ recommended.
- The script is defensive and prints actionable errors.
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI  # new client
except Exception:
    raise SystemExit("Please install openai (pip install openai>=1.0.0) before running this script.")

# ----------------- Helpers -----------------
def run(cmd: List[str], check=True):
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

def confirm(prompt: str, non_interactive: bool=False) -> bool:
    if non_interactive:
        print("(auto-confirm) " + prompt)
        return True
    resp = input(prompt + " (Enter to continue, 'q' to quit): ")
    if resp.strip().lower() == "q":
        print("Aborted by user.")
        sys.exit(1)
    return True

def which_bin(name: str) -> Optional[str]:
    return shutil.which(name)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ----------------- Step Implementations -----------------
def check_ffmpeg():
    print("Step 1: Checking ffmpeg...")
    ff = which_bin("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg and make sure it's on PATH.")
    out = run([ff, "-version"], check=True)
    print("ffmpeg OK.")
    return out.splitlines()[0]

def make_client(api_key: Optional[str]):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        raise RuntimeError("OPENAI_API_KEY not set. Set env var or pass --api-key.")
    client = OpenAI()
    return client

def probe_models(client: OpenAI):
    print("Step 2: Checking available models (this shows which OpenAI models your key can access)...")
    try:
        res = client.models.list()
    except Exception as e:
        raise RuntimeError(f"Could not list models: {e}")
    model_ids = []
    # new client may return .data or a list-like, handle both
    if hasattr(res, "data"):
        model_ids = [m.id for m in res.data]
    elif isinstance(res, list):
        model_ids = [m.get("id") if isinstance(m, dict) else getattr(m, "id", None) for m in res]
    else:
        # try to be tolerant
        try:
            model_ids = [m["id"] for m in res]
        except Exception:
            model_ids = []
    print(f"Found {len(model_ids)} models (showing some candidates):")
    for m in model_ids[:60]:
        print("  -", m)
    return model_ids

def choose_model(available: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in available:
            print(f"Selected model: {c}")
            return c
    print("None of the preferred models are available:", candidates)
    return None

def extract_audio_first_pass(video_path: str, audio_out: str):
    print("Step 3: Extracting full audio (mono, 16kHz WAV)...")
    run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_out
    ])
    if not Path(audio_out).exists():
        raise RuntimeError("Audio extraction failed, file not found: " + str(audio_out))
    # print duration via ffprobe
    dur = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_out], check=False)
    print("Extracted audio duration (s):", dur.strip())
    return audio_out

def make_short_clip(in_audio: str, out_clip: str, seconds: int = 10):
    print(f"Creating a short {seconds}s clip for testing...")
    run(["ffmpeg", "-y", "-i", in_audio, "-t", str(seconds), out_clip])
    if not Path(out_clip).exists():
        raise RuntimeError("Short clip creation failed")
    return out_clip

def transcribe_test(client: OpenAI, model: str, clip_path: str):
    print(f"Step 4: Testing transcription with model {model} on short clip: {clip_path}")
    with open(clip_path, "rb") as fh:
        try:
            # attempt a structured json-like response first
            resp = client.audio.transcriptions.create(model=model, file=fh, response_format="verbose_json")
            print("Transcription (verbose_json) succeeded.")
            return resp  # user code will inspect resp
        except Exception as e:
            print("verbose_json failed or unsupported by model:", e)
            fh.seek(0)
            try:
                # try text response
                resp = client.audio.transcriptions.create(model=model, file=fh, response_format="text")
                print("Transcription (text) succeeded.")
                return {"text": resp}
            except Exception as e2:
                raise RuntimeError(f"Transcription failed with model {model}: {e2}") from e2

def segments_from_response(resp: Any) -> List[Dict[str, Any]]:
    """
    Normalize various response shapes into a list of segments: [{start, end, text}, ...]
    If model returned plain text, return single segment covering entire duration (caller must set times).
    """
    if resp is None:
        return []
    # If it's already dict-like with 'segments'
    if hasattr(resp, "get") and resp.get("segments"):
        return resp.get("segments")
    if isinstance(resp, dict) and "segments" in resp:
        return resp["segments"]
    # If it's a string/single text
    if isinstance(resp, str):
        return [{"start": 0.0, "end": None, "text": resp}]
    if isinstance(resp, dict) and resp.get("text"):
        return [{"start": 0.0, "end": None, "text": resp.get("text")}]
    # fallback
    return []

def write_srt(segments: List[Dict[str,Any]], out_srt: str):
    def s_to_srt(ts):
        if ts is None:
            ts = 0.0
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    with open(out_srt, "w", encoding="utf-8") as fh:
        for i, seg in enumerate(segments, start=1):
            print(f"Segment {i}: '{seg.get('text', '')[:50]}'")
            start = seg.get("start", 0.0) or 0.0
            end = seg.get("end", start + 3.0) or (start + 3.0)
            text = seg.get("text", "").strip()
            fh.write(f"{i}\n{s_to_srt(start)} --> {s_to_srt(end)}\n{text}\n\n")
    print("Wrote preview subtitles to:", out_srt)

def translate_segments_with_model(client: OpenAI, segments: List[Dict[str,Any]], target_lang: str, translator_model_candidates: List[str], available_models: List[str]):
    # pick a text model for translation (prefer gpt-4o-mini, else gpt-3.5-turbo)
    chosen = choose_model(available_models, translator_model_candidates) or "gpt-3.5-turbo"
    print(f"Using {chosen} to translate segments to {target_lang}")
    translated = []
    for seg in segments:
        text = seg.get("text","").strip()
        if not text:
            translated.append({**seg, "text_translated": ""})
            continue
        prompt = f"Translate the following text to {target_lang}, preserving meaning and conciseness. Output only the translation.\n\nText:\n{text}"
        try:
            resp = client.responses.create(model=chosen, input=prompt)
            # new client may return .output or .choices; attempt robust extraction
            new_text = ""
            if hasattr(resp, "output") and resp.output:
                # join textual content parts
                parts = []
                for item in resp.output:
                    if isinstance(item, dict) and item.get("content"):
                        parts.append(item.get("content"))
                    elif isinstance(item, str):
                        parts.append(item)
                new_text = " ".join(parts).strip()
            elif isinstance(resp, dict) and "output" in resp:
                # older style
                new_text = resp["output"][0].get("content", "")
            else:
                # fallback: str(resp)
                new_text = str(resp)
        except Exception as e:
            print("Translation call failed:", e)
            new_text = text
        translated.append({**seg, "text_translated": new_text})
    return translated

def tts_generate_with_model(client: OpenAI, text: str, out_path: str, tts_model_candidates: List[str], available_models: List[str], voice: Optional[str]=None):
    chosen = choose_model(available_models, tts_model_candidates) or None
    if chosen is None:
        raise RuntimeError("No compatible TTS model available in your account. Candidates tried: " + ", ".join(tts_model_candidates))
    print(f"Using TTS model {chosen} to synthesize voice.")
    # chunk large text to avoid token limits
    tmp_path = Path(out_path).with_suffix(".part.wav")
    # use audio.speech.create per docs
    try:
        resp = client.audio.speech.create(model=chosen, voice=(voice or "alloy"), input=text)
        # resp likely supports stream_to_file
        if hasattr(resp, "stream_to_file"):
            resp.stream_to_file(out_path)
        else:
            # fallback: resp.read() or resp.content
            try:
                data = resp.read()
                with open(out_path, "wb") as fh:
                    fh.write(data)
            except Exception as e:
                raise RuntimeError("Couldn't write TTS output: " + str(e))
    except Exception as e:
        raise RuntimeError("TTS generation failed: " + str(e))
    print("Wrote TTS audio to:", out_path)
    return out_path

def mux_audio_into_video(video_in: str, audio_in: str, out_video: str):
    print("Muxing new audio into video (replacing original audio)...")
    run([
        "ffmpeg", "-y", "-i", video_in, "-i", audio_in,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", out_video
    ])
    print("Muxed video written to:", out_video)

# ----------------- CLI and main -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="input video path")
    p.add_argument("--output", "-o", required=True, help="output directory")
    p.add_argument("--api-key", help="OpenAI API key (optional; or set OPENAI_API_KEY)")
    p.add_argument("--target-lang", default="en", help="target language code (for translation/TTS)")
    p.add_argument("--yes", action="store_true", help="non-interactive: auto-confirm each step")
    p.add_argument("--no-tts", action="store_true", help="skip TTS stage")
    p.add_argument("--no-translate", action="store_true", help="skip translation (keep original transcript language)")
    p.add_argument("--burn", action="store_true", help="burn subtitles into final video")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output)
    safe_mkdir(out_dir)
    non_interactive = args.yes

    # Step 1: ffmpeg
    try:
        ver = check_ffmpeg()
        print(ver)
        confirm("ffmpeg is present and working.", non_interactive)
    except Exception as e:
        raise SystemExit("ffmpeg check failed: " + str(e))

    # Step 2: OpenAI client + model list
    try:
        client = make_client(args.api_key)
    except Exception as e:
        raise SystemExit("OpenAI client init failed: " + str(e))
    try:
        models = probe_models(client)
        confirm("Model list retrieved. Check that the STT/TTS candidates are present.", non_interactive)
    except Exception as e:
        raise SystemExit("Model list failed: " + str(e))

    # prefer models in order
    stt_candidates = ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"]
    tts_candidates = ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]
    translator_text_candidates = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

    chosen_stt = choose_model(models, stt_candidates) or choose_model(models, ["whisper-1"]) or "whisper-1"
    print("STT model to try:", chosen_stt)

    # Step 3: extract audio
    audio_full = str(out_dir / "extracted_audio.wav")
    try:
        extract_audio_first_pass(args.input, audio_full)
        confirm("Full audio extracted.", non_interactive)
    except Exception as e:
        raise SystemExit("Audio extraction failed: " + str(e))

    # Step 4: create short clip and test transcription
    clip = str(out_dir / "test_clip.wav")
    try:
        make_short_clip(audio_full, clip, seconds=10)
        resp = transcribe_test(client, chosen_stt, clip)
        segs = segments_from_response(resp)
        print("Sample transcription (first segments):")
        for s in segs[:5]:
            print(" ", s.get("text") if isinstance(s, dict) else s)
        srt_preview = out_dir / "preview.srt"
        write_srt(segs, str(srt_preview))
        confirm("Transcription test OK and SRT preview written: " + str(srt_preview), non_interactive)
    except Exception as e:
        raise SystemExit("Transcription test failed: " + str(e))

    # Step 5: optionally translate segments
    if not args.no_translate:
        print("Translating segments to target language:", args.target_lang)
        translated_segs = translate_segments_with_model(client, segs, args.target_lang, translator_text_candidates, models)
        # write translated srt
        srt_trans = out_dir / f"translated_{args.target_lang}.srt"
        write_srt([{**s, "text": s.get("text_translated", s.get("text",""))} for s in translated_segs], str(srt_trans))
        confirm("Translated SRT written: " + str(srt_trans), non_interactive)
    else:
        srt_trans = out_dir / "translated_skipped.srt"
        print("Skipping translation as requested.")
        translated_segs = segs

    # Step 6: optional TTS
    final_audio = out_dir / "final_tts.wav"
    if args.no_tts:
        print("Skipping TTS generation as requested.")
    else:
        try:
            # join translated text for TTS (you can generate per-segment if you want smaller chunks)
            joined = " ".join([s.get("text_translated", s.get("text","")) for s in translated_segs])
            print(f"DEBUG: Joined TTS input text length: {len(joined)}")
            print(f"DEBUG: Sample start of TTS text:\n{joined[:200]}")
            tts_out = str(final_audio)
            tts_generate_with_model(client, joined, tts_out, tts_candidates, models, voice="alloy")
            confirm("TTS audio generated: " + tts_out, non_interactive)
        except Exception as e:
            raise SystemExit("TTS generation failed: " + str(e))

    # Step 7: mux into video
    try:
        final_video = out_dir / (Path(args.input).stem + f"_dub_{args.target_lang}.mp4")
        if not args.no_tts:
            mux_audio_into_video(args.input, str(final_audio), str(final_video))
        else:
            print("No TTS audio to mux; skipping mux step.")
            final_video = None
        if args.burn:
            # burn subtitles (use translated srt)
            if final_video:
                burned = out_dir / (Path(args.input).stem + f"_dub_{args.target_lang}_burn.mp4")
                run(["ffmpeg", "-y", "-i", str(final_video), "-vf", f"subtitles={str(srt_trans)}", "-c:a", "copy", str(burned)])
                print("Burned video written:", burned)
        print("All done. Outputs in:", out_dir)
    except Exception as e:
        raise SystemExit("Muxing/Finalizing failed: " + str(e))


if __name__ == "__main__":
    main()
