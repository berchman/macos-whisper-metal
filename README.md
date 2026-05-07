# macos-whisper-metal (Beta 0.5)

A local, subscription-free audio → transcript (and optional WordPress draft) pipeline for **macOS on Apple Silicon** using **whisper.cpp + Metal**.

## What changed in Beta 0.5
- **Timestamps deferred** (not generated yet).
- **Better format**: conservative paragraph breaks based on pauses (≥ 1.5s) when JSON segmentation is available.
- **Broader input support**: handles `.mp3`, `.m4a/.aac`, `.mov` (anything `ffmpeg` can decode).
- **Footer + word count** appended to transcript files, with a **publishing-safe footer block** that WordPress ignores by default.
- **Checksum-based skip** so reprocessing the same audio won't re-run or double-append.
- **Optional filler cleanup**: `--clean-fillers` writes `.cleaned.wav` and `.fillers.json`.

## Requirements (fresh machine friendly)
1. macOS 13+ (Ventura or newer recommended)
2. Apple Silicon (M1/M2/M3/M4)
3. Homebrew (installer below)
4. `ffmpeg`, `fswatch`, `whisper-cpp` (Homebrew formula)

### Install Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install dependencies
```bash
brew update
brew install ffmpeg fswatch whisper-cpp
```

### Model
Download a whisper.cpp model (example: medium) and put it here:
```
~/myscripts/_MODELS/ggml-medium.bin
```

## Run one file manually
```bash
python3 ./transcribe_audio.py ./_01_Processed/example.m4a ./_02_Transcripts
```

## Clean filler audio
```bash
python3 ./transcribe_audio.py ./_01_Processed/example.m4a ./_02_Transcripts --clean-fillers
```

This mutes detected filler words such as `um`, `uh`, `ah`, `er`, and `erm` with 75ms padding, preserving the original duration.

## Notes
- Transcription uses `whisper-cli` from the Homebrew `whisper-cpp` package.
- If you later enable timestamps for scripts, we’ll add them as an optional mode without polluting raw transcripts or blog drafts.
