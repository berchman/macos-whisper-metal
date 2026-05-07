#!/usr/bin/env python3
"""
transcribe_audio.py (Beta 0.5)

Transcribe an audio/video file locally on macOS (Apple Silicon) using:
- ffmpeg for decode/resample -> 16kHz mono WAV
- whisper.cpp via Homebrew `whisper-cli` with Metal acceleration

Outputs:
- <output_dir>/<stem>.txt

Formatting goals:
- No timestamps (for now)
- Conservative paragraph breaks based on pause gaps >= 1.5s (when JSON segments are available)
- Append a footer block containing:
  - word count (also useful inside WP)
  - a short "transcribed with" tagline + link to repo
  - Footer is wrapped in markers so the WordPress publisher can strip it by default.

Idempotency:
- Writes a state file in _LOGS to avoid reprocessing identical inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PAUSE_PARAGRAPH_SECONDS = 1.5

FOOTER_START = "[[TRANSCRIBE_FOOTER_START]]"
FOOTER_END   = "[[TRANSCRIBE_FOOTER_END]]"

REPO_URL = "https://github.com/berchman/macos-whisper-metal"


def _log(log_file: Path, msg: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(state_file: Path) -> Dict[str, Any]:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state_file: Path, state: Dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def resolve_bins() -> Tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    whisper = shutil.which("whisper-cli") or "/opt/homebrew/bin/whisper-cli"

    if not Path(ffmpeg).exists():
        raise SystemExit("ffmpeg not found. Install with: brew install ffmpeg")
    if not Path(whisper).exists():
        raise SystemExit("whisper-cli not found. Install with: brew install whisper-cpp")

    return ffmpeg, whisper


def convert_to_wav(ffmpeg_bin: str, in_path: Path, wav_path: Path) -> None:
    """
    Convert anything ffmpeg can decode (m4a/aac/mp3/mov/...) to:
      - mono
      - 16kHz
      - pcm_s16le wav
    """
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(in_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_whisper(
    whisper_bin: str,
    model_path: Path,
    wav_path: Path,
    out_prefix: Path,
    threads: int,
    beam_size: int,
    best_of: int,
    language: str,
    try_json: bool = True,
) -> Tuple[Optional[Path], Path]:
    """
    Runs whisper-cli. Produces a .txt always, and tries to produce .json when supported.
    Returns (json_path_or_None, txt_path).
    """
    cmd = [
        whisper_bin,
        "-m", str(model_path),
        "-f", str(wav_path),
        "--language", language,
        "--output-txt",
        "--no-timestamps",
        "--threads", str(threads),
        "--beam-size", str(beam_size),
        "--best-of", str(best_of),
        "--output-file", str(out_prefix),
    ]

    # Try JSON if the binary supports it.
    json_path = out_prefix.with_suffix(".json")
    txt_path = out_prefix.with_suffix(".txt")

    if try_json:
        cmd_json = cmd.copy()
        cmd_json.insert(cmd_json.index("--output-txt"), "--output-json")
        try:
            subprocess.run(cmd_json, check=True)
            if json_path.exists() and txt_path.exists():
                return json_path, txt_path
        except subprocess.CalledProcessError:
            # Fall back to txt-only
            pass

    subprocess.run(cmd, check=True)
    return (json_path if json_path.exists() else None), txt_path


def parse_whisper_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Expect whisper.cpp-style json containing segments with start/end and text.
    We'll tolerate a few shapes.
    """
    data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))

    # Common: {"transcription":[{"timestamps":{"from":0.0,"to":1.2},"text":"..."}]}
    # Or: {"segments":[{"t0":..., "t1":..., "text":...}, ...]}
    segments = []
    if isinstance(data, dict):
        if "segments" in data and isinstance(data["segments"], list):
            segments = data["segments"]
        elif "transcription" in data and isinstance(data["transcription"], list):
            segments = data["transcription"]
    if not isinstance(segments, list):
        return []

    out = []
    for s in segments:
        if not isinstance(s, dict):
            continue

        text = str(s.get("text", "")).strip()
        if not text:
            continue

        # timing variants
        start = None
        end = None
        if "t0" in s and "t1" in s:
            # t0/t1 often in 10ms units. If huge, convert.
            start = float(s["t0"])
            end = float(s["t1"])
            # heuristic: if values are > 1000, assume centiseconds
            if start > 1000 or end > 1000:
                start /= 100.0
                end /= 100.0
        elif "start" in s and "end" in s:
            start = float(s["start"])
            end = float(s["end"])
        elif "timestamps" in s and isinstance(s["timestamps"], dict):
            start = float(s["timestamps"].get("from", 0.0))
            end = float(s["timestamps"].get("to", 0.0))

        out.append({"start": start, "end": end, "text": text})
    return out


def build_paragraph_text(segments: List[Dict[str, Any]]) -> str:
    """
    Paragraph break when gap between previous end and next start >= PAUSE_PARAGRAPH_SECONDS.
    Otherwise join with spaces.
    """
    if not segments:
        return ""

    paras: List[str] = []
    cur: List[str] = []
    prev_end: Optional[float] = None

    for seg in segments:
        txt = seg["text"].strip()
        if not txt:
            continue

        start = seg.get("start")
        end = seg.get("end")

        if prev_end is not None and isinstance(start, (int, float)) and isinstance(prev_end, (int, float)):
            gap = float(start) - float(prev_end)
            if gap >= PAUSE_PARAGRAPH_SECONDS and cur:
                paras.append(" ".join(cur).strip())
                cur = []

        cur.append(txt)
        if isinstance(end, (int, float)):
            prev_end = float(end)

    if cur:
        paras.append(" ".join(cur).strip())

    # Gentle cleanup
    out = "\n\n".join(p for p in paras if p)
    out = out.replace("  ", " ").strip()
    return out


def word_count(text: str) -> int:
    words = [word for word in text.replace("\n", " ").split(" ") if word.strip()]
    return len(words)


def strip_existing_footer(text: str) -> str:
    if FOOTER_START in text and FOOTER_END in text:
        pre = text.split(FOOTER_START, 1)[0].rstrip()
        return pre
    return text.rstrip()


def write_transcript(output_path: Path, body: str, wc: int) -> None:
    body = strip_existing_footer(body).strip()

    footer_lines = [
        "",
        FOOTER_START,
        f"<h6>Word count: {wc}</h6>",
        f"Transcribed locally with whisper.cpp (Metal). More info: {REPO_URL}",
        FOOTER_END,
        "",
    ]
    final = body + "\n" + "\n".join(footer_lines)
    output_path.write_text(final, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_file", help="Path to input audio/video (mp3/m4a/aac/mov/...)")
    ap.add_argument("output_dir", help="Directory to write transcripts")
    ap.add_argument("--model", default=str(Path.home() / "_00_GIT_MASTER/transcription-to-blog" / "_MODELS" / "ggml-medium.bin"))
    ap.add_argument("--language", default="en")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--best-of", type=int, default=5)
    args = ap.parse_args()

    audio_path = Path(args.audio_file).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "_LOGS"
    log_file = log_dir / "transcribe.log"
    state_file = log_dir / "transcribe_state.json"

    model_path = Path(args.model).expanduser().resolve()

    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    ffmpeg_bin, whisper_bin = resolve_bins()

    # Checksum-based skip
    audio_sha = sha256_file(audio_path)
    settings_key = f"model={model_path.name}|lang={args.language}|threads={args.threads}|beam={args.beam_size}|best={args.best_of}"
    run_sha = hashlib.sha256(f"{audio_sha}|{settings_key}".encode("utf-8")).hexdigest()

    state = load_state(state_file)
    state.setdefault("runs", {})
    prev = state["runs"].get(str(audio_path))
    if prev and prev.get("run_sha") == run_sha:
        _log(log_file, f"SKIP (checksum match): {audio_path.name}")
        return

    out_txt = out_dir / f"{audio_path.stem}.txt"

    # Convert to wav in same folder as input (temporary)
    wav_path = audio_path.with_suffix(".wav")
    _log(log_file, f"Converting to wav: {audio_path.name} -> {wav_path.name}")
    try:
        convert_to_wav(ffmpeg_bin, audio_path, wav_path)
    except subprocess.CalledProcessError as e:
        _log(log_file, f"ERROR converting with ffmpeg: {e}")
        raise

    out_prefix = out_dir / audio_path.stem

    _log(log_file, "Backend: whisper-cpp (Homebrew) | Metal GPU (Apple Silicon)")
    json_path = None
    try:
        json_path, txt_path = run_whisper(
            whisper_bin=whisper_bin,
            model_path=model_path,
            wav_path=wav_path,
            out_prefix=out_prefix,
            threads=args.threads,
            beam_size=args.beam_size,
            best_of=args.best_of,
            language=args.language,
            try_json=True,
        )
    finally:
        # Always delete temp wav
        try:
            if wav_path.exists():
                wav_path.unlink()
        except Exception:
            pass

    # Prefer JSON-based formatting
    body = ""
    if json_path and json_path.exists():
        segs = parse_whisper_json(json_path)
        body = build_paragraph_text(segs)
    if not body:
        # Fallback: use txt output from whisper-cli as-is.
        txt_fallback = out_prefix.with_suffix(".txt")
        body = txt_fallback.read_text(encoding="utf-8", errors="ignore").strip() if txt_fallback.exists() else ""

    wc = word_count(body)
    write_transcript(out_txt, body, wc)

    # Update state
    state["runs"][str(audio_path)] = {
        "run_sha": run_sha,
        "audio_sha": audio_sha,
        "settings": settings_key,
        "output": str(out_txt),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_state(state_file, state)

    _log(log_file, f"Transcript written: {out_txt}")


if __name__ == "__main__":
    main()
