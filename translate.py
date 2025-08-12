import os
import subprocess
import sys

def run_step(cmd, description, output_file):
    print(f"\n=== {description} ===")
    if os.path.exists(output_file):
        print(f"[SKIP] {description} — already exists: {output_file}")
        return

    try:
        subprocess.run(cmd, check=True)
        print(f"{description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        sys.exit(1)

def main(project_dir):
    video_path = os.path.join(project_dir, "video.mp4")
    audio_path = os.path.join(project_dir, "audio.wav")
    transcribe_json = os.path.join(project_dir, "transcript.json")
    translated_json = os.path.join(project_dir, "translated.json")
    subtitles_srt = os.path.join(project_dir, "subtitles.srt")
    tts_audio = os.path.join(project_dir, "tts.wav")
    original_parts = os.path.join(project_dir, "parts_original")
    parts_dir = os.path.join(project_dir, "parts")
    merged_video = os.path.join(project_dir, "merged_video.mp4")
    mixed_audio_video = os.path.join(project_dir, "mixed_audio_video.mp4")
    burned_video = os.path.join(project_dir, "burned_video.mp4")
    watermarked_video = os.path.join(project_dir, "final_video.mp4")
    watermark_image = os.path.join("media", "watermark.png")  # Expect this present in project dir

    # Ensure parts dir exists
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(original_parts, exist_ok=True)
    
    # Step 1: Extract audio    
    run_step([
        "ffmpeg",
        "-i", str(video_path),
        "-vn",  # no video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",  # 16 kHz
        "-ac", "1",  # mono
        str(audio_path)
    ], "Extracting audio from video", audio_path)

    # 1. Transcribe
    run_step(
        [sys.executable, "src/transcribe_audio.py", audio_path, transcribe_json],
        "Transcribing audio",
        transcribe_json
    )

    # 2. Translate
    run_step(
        [sys.executable, "src/translate_segments.py", transcribe_json, translated_json],
        "Translating transcript",
        translated_json
    )

    # 3. JSON to SRT
    run_step(
        [sys.executable, "src/json_to_srt.py", translated_json, subtitles_srt],
        "Generating subtitles (SRT)",
        subtitles_srt
    )

    # 4. Generate TTS
    run_step(
        [sys.executable, "src/generate_tts.py", translated_json, parts_dir],
        "Generating TTS audio",
        os.path.join(parts_dir, "segment_001.wav")
    )

    # 5. Split TTS audio into segments
    run_step(
        [sys.executable, "src/split_audio_by_segments.py", audio_path, transcribe_json, original_parts],
        "Splitting Original audio into segments for reference",
        os.path.join(original_parts, "segment_001.wav")
    )

    # # 6. Merge TTS audio segments into video
    # run_step(
    #     [sys.executable, "src/merge_audio.py", video_path, translated_json, parts_dir, merged_video],
    #     "Merging TTS audio segments into video"
    # )

    # 7. Mix original audio with TTS audio
    run_step(
        [sys.executable, "src/audio_mix.py", video_path, translated_json, parts_dir, mixed_audio_video],
        "Mixing original audio with TTS",
        mixed_audio_video
    )

    # 8. Burn subtitles into video
    run_step(
        [sys.executable, "src/add_subtitles.py", "burn", mixed_audio_video, subtitles_srt, burned_video],
        "Burning subtitles into video",
        burned_video
    )

    # 9. Add watermark
    if os.path.isfile(watermark_image):
        run_step(
            [sys.executable, "src/add_watermark.py", burned_video, watermark_image, watermarked_video],
            "Adding watermark",
            watermarked_video
        )
        print(f"\nFinal video saved as: {watermarked_video}")
    else:
        print(f"\nWatermark image not found at {watermark_image}, skipping watermark step.")
        print(f"Final video saved as: {burned_video}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_all.py <project_folder>")
        sys.exit(1)

    main(sys.argv[1])
