import sys
import os
import json
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

def merge_audio(video_path, segments_json_path, audio_parts_dir, output_file):
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    
    video = VideoFileClip(video_path)
    print(f"Video duration: {video.duration} seconds")

    # Load segments json
    with open(segments_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = data.get("segments", None)
    if not segments:
        print("No 'segments' key found in JSON, aborting.")
        return

    audio_clips = []
    for i, segment in enumerate(segments):
        start = segment["start"]
        segment_filename = os.path.join(audio_parts_dir, f"segment_{i+1:03d}.wav")
        
        if not os.path.exists(segment_filename):
            print(f"Warning: Audio segment not found: {segment_filename}")
            continue
        
        try:
            audio_clip = AudioFileClip(segment_filename).set_start(start)
        except Exception as e:
            print(f"Error loading audio segment {segment_filename}: {e}")
            continue

        audio_clips.append(audio_clip)
    
    if not audio_clips:
        print("No audio segments found to merge, exiting.")
        return
    
    final_audio = CompositeAudioClip(audio_clips)
    final_audio = final_audio.set_duration(video.duration)
    
    final_video = video.set_audio(final_audio)
    
   # output_path = os.path.join(output_dir, "final_video.mp4")
    final_video.write_videofile(output_file, codec="libx264", audio_codec="aac")

    print(f"Output video saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python merge_audio.py <video_path> <segments_json> <audio_parts_dir> <output_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    segments_json_path = sys.argv[2]
    audio_parts_dir = sys.argv[3]
    output_file = sys.argv[4]

    merge_audio(video_path, segments_json_path, audio_parts_dir, output_file)
