import sys
import subprocess

def add_watermark(input_video, watermark_image, output_video, margin=10):
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-i", watermark_image,
        "-filter_complex", f"overlay=W-w-{margin}:{margin}",
        "-c:a", "copy",
        output_video
    ]
    subprocess.run(cmd, check=True)
    print(f"Watermarked video saved to: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python add_watermark.py <input_video> <watermark_image> <output_video>")
        sys.exit(1)

    input_video = sys.argv[1]
    watermark_image = sys.argv[2]
    output_video = sys.argv[3]

    add_watermark(input_video, watermark_image, output_video)
