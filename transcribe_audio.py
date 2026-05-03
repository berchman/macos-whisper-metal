#!/usr/bin/env python3
"""
transcribe_audio.py (Beta 0.6)

Local transcription pipeline for macOS (Apple Silicon):

- Watches/runner calls this script for a single file at a time.
- Uses ffmpeg to convert non-wav inputs to 16kHz mono PCM WAV (temp file)
- Uses whisper.cpp via Homebrew `whisper-cli` with Metal acceleration
- Writes: <output_dir>/<stem>.txt
- Writes JSON sidecar (if supported) to: <output_dir>/_sidecars/<stem>.json
  (keeps the transcript folder clean: TXT only at top level)

Formatting goals:
- No inline timestamps in TXT (for now)
- Better paragraphing using JSON segments when available (pause gap >= 1.5s)
- Append a footer block containing:
  - word count
  - transcribed-with tagline + repo link
  - wrapped in markers so WP publisher can strip it by default

Idempotency:
- State file in _LOGS prevents reprocessing identical inputs/settings
- Footer is checksum-safe (won’t double-append)

NOTE:
- If input is a .wav, we do NOT run ffmpeg conversion by default (avoids overwrite issues).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import textwrap
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
    # Capture output so launchd logs don’t explode; still errors out on non-zero.
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
    want_json: bool = True,
) -> Tuple[Optional[Path], Path]:
    """
    Runs whisper-cli.

    Always generates TXT: out_prefix.txt

    If want_json and binary supports it, also generates JSON:
      out_prefix.json

    Returns (json_path_or_None, txt_path)
    """
    txt_path = out_prefix.with_suffix(".txt")

    json_dir = out_prefix.parent / "json_files"
    json_dir.mkdir(parents=True, exist_ok=True)

    json_prefix = json_dir / out_prefix.name
    json_path = json_prefix.with_suffix(".json")

    cmd_txt = [
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

    cmd_json = [
        whisper_bin,
        "-m", str(model_path),
        "-f", str(wav_path),
        "--language", language,
        "--output-json",
        "--threads", str(threads),
        "--beam-size", str(beam_size),
        "--best-of", str(best_of),
        "--output-file", str(json_prefix),
    ]

    # Try JSON first (so we can paragraph better), but don’t die if unsupported.
    json_ok = False
    if want_json:
        try:
            subprocess.run(cmd_json, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            json_ok = json_path.exists()
        except subprocess.CalledProcessError:
            json_ok = False

    # Always generate TXT (this is the contract)
    subprocess.run(cmd_txt, check=True)

    return (json_path if json_ok else None), txt_path


def parse_timestamp_seconds(value: Any) -> Optional[float]:
    """
    Accept numeric seconds or whisper.cpp timestamp strings like 00:00:11,000.
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        pass

    match = re.fullmatch(r"(?:(\d+):)?(\d{2}):(\d{2})(?:[,.](\d{1,3}))?", value)
    if not match:
        return None

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis_raw = match.group(4) or "0"
    millis = int(millis_raw.ljust(3, "0")[:3])
    return (hours * 3600) + (minutes * 60) + seconds + (millis / 1000.0)


def parse_whisper_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Expect whisper.cpp-style json containing segments with start/end and text.
    Tolerates a few shapes.
    """
    data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))

    segments: Any = []
    if isinstance(data, dict):
        if "segments" in data and isinstance(data["segments"], list):
            segments = data["segments"]
        elif "transcription" in data and isinstance(data["transcription"], list):
            segments = data["transcription"]

    if not isinstance(segments, list):
        return []

    out: List[Dict[str, Any]] = []
    for s in segments:
        if not isinstance(s, dict):
            continue

        text = str(s.get("text", "")).strip()
        if not text:
            continue

        start: Optional[float] = None
        end: Optional[float] = None

        if "t0" in s and "t1" in s:
            start = parse_timestamp_seconds(s["t0"])
            end = parse_timestamp_seconds(s["t1"])
            # heuristic: if large, likely centiseconds
            if start is not None and end is not None and (start > 1000 or end > 1000):
                start /= 100.0
                end /= 100.0
        elif "start" in s and "end" in s:
            start = parse_timestamp_seconds(s["start"])
            end = parse_timestamp_seconds(s["end"])
        elif "timestamps" in s and isinstance(s["timestamps"], dict):
            start = parse_timestamp_seconds(s["timestamps"].get("from"))
            end = parse_timestamp_seconds(s["timestamps"].get("to"))
        elif "offsets" in s and isinstance(s["offsets"], dict):
            start_offset = parse_timestamp_seconds(s["offsets"].get("from"))
            end_offset = parse_timestamp_seconds(s["offsets"].get("to"))
            if start_offset is not None and end_offset is not None:
                start = start_offset / 1000.0
                end = end_offset / 1000.0

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
        txt = seg.get("text", "").strip()
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

    out = "\n\n".join(p for p in paras if p).replace("  ", " ").strip()
    return out

def format_srt_timestamp(seconds: float) -> str:
    ms = int(seconds * 1000)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms = ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def wrap_text_srt(text: str, width: int = 35) -> List[str]:
    return textwrap.wrap(text, width=width)


def generate_srt_from_segments(segments: List[Dict[str, Any]]) -> str:
    entries = []
    buffer = []
    start_time = None
    end_time = None

    def flush():
        nonlocal buffer, start_time, end_time
        if not buffer:
            return
        text = " ".join(buffer).strip()
        lines = wrap_text_srt(text)

        if lines:
            entries.append({
                "start": start_time,
                "end": end_time,
                "lines": lines
            })

        buffer = []
        start_time = None
        end_time = None

    for seg in segments:
        txt = seg["text"].strip()
        if not txt:
            continue

        seg_start = seg.get("start")
        seg_end = seg.get("end")
        if not isinstance(seg_start, (int, float)) or not isinstance(seg_end, (int, float)):
            continue

        if start_time is None:
            start_time = float(seg_start)

        buffer.append(txt)
        end_time = float(seg_end)

        # flush if too long
        if len(" ".join(buffer)) > 70:
            flush()

    flush()

    # build final SRT string
    srt_lines = []
    for i, e in enumerate(entries, start=1):
        srt_lines.append(str(i))
        srt_lines.append(
            f"{format_srt_timestamp(e['start'])} --> {format_srt_timestamp(e['end'])}"
        )
        srt_lines.extend(e["lines"])
        srt_lines.append("")  # blank line

    return "\n".join(srt_lines)

def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def strip_existing_footer(text: str) -> str:
    if FOOTER_START in text and FOOTER_END in text:
        return text.split(FOOTER_START, 1)[0].rstrip()
    return text.rstrip()


def write_transcript(output_path: Path, body: str, wc: int) -> None:
    body = strip_existing_footer(body).strip()

    footer_lines = [
        "",
        FOOTER_START,
        f"Word count: {wc}",
        f"Transcribed locally with whisper.cpp (Metal). More info: {REPO_URL}",
        FOOTER_END,
        "",
    ]
    final = body + "\n" + "\n".join(footer_lines)
    output_path.write_text(final, encoding="utf-8")


def move_json_sidecar(json_path: Path, out_dir: Path) -> Path:
    """
    Move JSON to <out_dir>/_sidecars/<name>.json
    """
    sidecar_dir = out_dir / "_sidecars"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    dest = sidecar_dir / json_path.name
    try:
        if dest.exists():
            dest.unlink()
        json_path.replace(dest)
    except Exception:
        # If move fails, just keep it where it is; caller can still parse it.
        return json_path
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_file", help="Path to input audio/video (mp3/m4a/aac/mov/wav/...)")
    ap.add_argument("output_dir", help="Directory to write transcripts")
    ap.add_argument(
        "--model",
        default=str(Path.home() / "_00_GIT_MASTER" / "00_TAWK2TEXT" / "_models" / "ggml-medium.bin"),
    )
    ap.add_argument("--language", default="en")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--best-of", type=int, default=5)
    ap.add_argument("--no-json", action="store_true", help="Disable JSON generation (txt only).")
    ap.add_argument("--srt", action="store_true", help="Generate SRT subtitles")
    args = ap.parse_args()

    audio_path = Path(args.audio_file).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "_logs"
    log_file = log_dir / "transcribe.log"
    state_file = log_dir / "transcribe_state.json"
    enable_json = True  # keep segments for formatting + future features
    model_path = Path(args.model).expanduser().resolve()

    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    ffmpeg_bin, whisper_bin = resolve_bins()

    # Checksum-based skip (prevents repeated processing)
    audio_sha = sha256_file(audio_path)
    settings_key = f"model={model_path.name}|lang={args.language}|threads={args.threads}|beam={args.beam_size}|best={args.best_of}|json={not args.no_json}"
    run_sha = hashlib.sha256(f"{audio_sha}|{settings_key}".encode("utf-8")).hexdigest()

    state = load_state(state_file)
    state.setdefault("runs", {})
    prev = state["runs"].get(str(audio_path))
    if prev and prev.get("run_sha") == run_sha:
        _log(log_file, f"SKIP (checksum match): {audio_path.name}")
        return

    out_txt = out_dir / f"{audio_path.stem}.txt"
    out_prefix = out_dir / audio_path.stem
    json_path = None
    txt_path = None
    
    # Prepare WAV path
    created_temp_wav = False
    wav_path = audio_path

    temp_wav_created = False

    # Decide final WAV path
    if audio_path.suffix.lower() == ".wav":
        wav_path = audio_path
    else:
        # Convert into FINAL_DIR (or a dedicated temp dir if you prefer)
        wav_path = (script_dir / "_wav_files" / f"{audio_path.stem}.wav").resolve()
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            _log(log_file, f"Converting to wav: {audio_path.name} -> {wav_path.name}")
            convert_to_wav(ffmpeg_bin, audio_path, wav_path)
            temp_wav_created = True
        except subprocess.CalledProcessError as e:
            _log(log_file, f"ERROR converting with ffmpeg: {e}")
            raise

    # Run whisper
    _log(log_file, "Backend: whisper-cpp (Homebrew) | Metal GPU (Apple Silicon)")
    json_path: Optional[Path] = None
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
            want_json=(not args.no_json),
        )
    except subprocess.CalledProcessError as e:
        _log(log_file, f"ERROR running whisper-cli: {e}")
        raise
    finally:
        # Delete temp wav ONLY if we created it (and regardless of whisper success)
        if temp_wav_created:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass
            
            
    # If JSON exists, move it into sidecars folder
    if json_path and json_path.exists():
        json_path = move_json_sidecar(json_path, out_dir)

    # Build body: JSON paragraphs if possible, else TXT fallback
    body = ""
    segs = []

    if json_path and json_path.exists():
        segs = parse_whisper_json(json_path)
        body = build_paragraph_text(segs)

        if args.srt and segs:
            srt_output = generate_srt_from_segments(segs)
            srt_path = out_prefix.with_suffix(".srt")
            srt_path.write_text(srt_output, encoding="utf-8")

    # Fallback if JSON failed or produced empty body
    if not body:
        txt_fallback = out_prefix.with_suffix(".txt")
        if txt_fallback.exists():
            body = txt_fallback.read_text(encoding="utf-8", errors="ignore").strip()
        else:
            body = ""

    wc = word_count(body)
    write_transcript(out_txt, body, wc)

    # Update state (idempotency)
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
