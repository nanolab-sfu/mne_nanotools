#!/usr/bin/env bash
set -euo pipefail

# ------------------ CONFIG ------------------
ROOT_DIR="/Volumes/A_Flores/temp_san/some_BRSHN"
TXT_FILE="${ROOT_DIR}/subjects_sessions.txt"

SCRIPT="/Users/isaant/Documents/PosDoc/Projects/generic_mne_pipelines/generic_taskfree_MEGIN.py"

TASK="rest"
RUN=1
PREFER="digFiltered"

# Optional: log file
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ------------------ CHECKS ------------------
[[ ! -f "$TXT_FILE" ]] && { echo "❌ TXT file not found: $TXT_FILE"; exit 1; }
[[ ! -f "$SCRIPT" ]] && { echo "❌ Script not found: $SCRIPT"; exit 1; }

# ------------------ LOOP ------------------
while IFS= read -r line; do
    # Skip empty lines or comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    subject_id=$(echo "$line" | awk '{print $1}')
    session=$(echo "$line" | awk '{print $2}')

    echo "=============================================="
    echo "▶ Processing $subject_id | $session"
    echo "=============================================="

    python "$SCRIPT" \
        --root_dir "$ROOT_DIR" \
        --subject_id "$subject_id" \
        --session "$session" \
        --task "$TASK" \
        --run "$RUN" \
        --prefer "$PREFER" \
        > "${LOG_DIR}/${subject_id}_${session}.log" 2>&1

    echo "✔ Finished $subject_id $session"

done < "$TXT_FILE"

echo "✅ All subjects processed."