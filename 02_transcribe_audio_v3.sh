#!/opt/homebrew/bin/bash

# launchd-safe PATH (Homebrew + system)
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# Resolve base directory (where this script lives)
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

INBOX_DIR="$BASE_DIR/_00_To_Be_Processed"
PROCESSED_DIR="$BASE_DIR/_01_Processed"
TRANSCRIPT_DIR="$BASE_DIR/_02_Transcripts"
LOGS_DIR="$BASE_DIR/_LOGS"
LOG_FILE="$LOGS_DIR/transcription_log_v3.txt"

mkdir -p "$LOGS_DIR"

echo "Watching for audio files in: $INBOX_DIR"
echo "Log file: $LOG_FILE"

FSWATCH="/opt/homebrew/bin/fswatch"

while true; do
  "$FSWATCH" -0 -e "\.DS_Store$" "$INBOX_DIR" | while IFS= read -r -d $'\0' file; do

    # Only act on regular files
    [ -f "$file" ] || continue

      ext="${file##*.}"
      ext="$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')"

    case "$ext" in
      wav|mp3|m4a|aac|mov)
        ;;
      *)
        continue
        ;;
    esac

    filename=$(basename "$file")
    echo "Detected audio file: $filename" | tee -a "$LOG_FILE"

    mv "$file" "$PROCESSED_DIR/$filename"
    echo "Moved to processed folder." | tee -a "$LOG_FILE"

    echo "Starting transcription..." | tee -a "$LOG_FILE"
    "$BASE_DIR/.venv/bin/python" "$BASE_DIR/transcribe_audio.py" \
      "$PROCESSED_DIR/$filename" \
      "$TRANSCRIPT_DIR" \
      --clean-fillers \
      2>&1 | tee -a "$LOG_FILE"

    echo "Finished transcription for $filename" | tee -a "$LOG_FILE"
    echo "--------------------------------------" | tee -a "$LOG_FILE"

  done

  echo "$(date) fswatch exited, restarting…" >> "$LOG_FILE"
  sleep 2
done
