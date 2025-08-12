import os
import sys
import json
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, afx

#mixes TTS audio with original video audio
#segments_json should contain timing info for TTS audio segments

def mix_audio_with_original(
    video_path: str,
    segments_json_path: str,
    tts_audio_dir: str,
    output_video_path: str,
    original_volume=0.2,
):
    # Load video and original audio
    video = VideoFileClip(video_path)
    original_audio = video.audio.volumex(original_volume)

    # Load segments with timing info
    with open(segments_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments")
    if not segments:
        print(f"No 'segments' key found in {segments_json_path}")
        return

    tts_audio_clips = []

    # Compose TTS audio clips at correct start times
    for i, segment in enumerate(segments):
        start = segment["start"]
        # Audio segment filename, padded with zeros (assumes segment_001.wav etc)
        filename = os.path.join(tts_audio_dir, f"segment_{i+1:03d}.wav")
        if not os.path.isfile(filename):
            print(f"Warning: Audio segment not found: {filename}")
            continue

        tts_clip = AudioFileClip(filename).set_start(start)
        tts_audio_clips.append(tts_clip)

    # Combine original audio with TTS overlays
    final_audio = CompositeAudioClip([original_audio] + tts_audio_clips)

    # Set combined audio on the video
    final_video = video.set_audio(final_audio)

    # Write the output video
    print(f"Writing output video with mixed audio to: {output_video_path}")
    final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

    # Clean up
    final_video.close()
    video.close()
    for clip in tts_audio_clips:
        clip.close()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python mix_audio.py <input_video> <segments_json> <tts_audio_dir> <output_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    segments_json_path = sys.argv[2]
    tts_audio_dir = sys.argv[3]
    output_video_path = sys.argv[4]

    mix_audio_with_original(video_path, segments_json_path, tts_audio_dir, output_video_path)
