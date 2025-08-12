AUTO DUB - README

Overview
--------
This repository contains auto_dub.py, a modular local pipeline to create subtitles and dubbed audio for foreign-language videos.
It relies on ffmpeg being installed and available in your PATH. For best quality, install Whisper (openai-whisper), and Coqui TTS (TTS).
Wav2Lip is optional for lip-syncing but requires separate setup and checkpoints.

Quick setup (Windows, RTX machine) - create a python venv, install deps
---------------------------------------------------------------------
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Notes and tips
--------------
- openai-whisper will download models the first time it runs. The "large" model is accurate but heavy. Use "small" or "medium" for faster runs.
- Coqui TTS (TTS package) may download models automatically; pick a model that supports your target language.
- If TTS generation is slow or you prefer cloud quality, consider ElevenLabs API or other cloud TTS providers (not included here).
- This script performs a simple workflow and does not attempt advanced time-stretching to precisely align TTS durations to original speech. For better sync, split into small segments (whisper provides timestamps) and generate segment-by-segment, then use audio alignment/time-stretch if necessary.
- Wav2Lip integration is left basic; see the Wav2Lip repo for best results and GPU tuning.

Usage examples
--------------
# generate subtitles (transcribed in original language)
python auto_dub.py -i input.mp4 -o out_dir --mode subs

# generate translated subtitles (whisper translate)
python auto_dub.py -i input.mp4 -o out_dir --mode subs --translate --target-lang en --burn

# generate dubbed audio using Coqui local TTS
python auto_dub.py -i input.mp4 -o out_dir --mode dub --tts-backend coqui --target-lang en

