#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Defaults (override in /etc/nanolab_backup.conf)
# -----------------------
BACKUP_ROOT="${BACKUP_ROOT:-/mnt/backup}"
HOME_SRC="${HOME_SRC:-/home}"
DATASETS_SRC="${DATASETS_SRC:-/usr/share/data-sets}"

# Minimum free space required on backup filesystem (in GiB)
MIN_FREE_GIB="${MIN_FREE_GIB:-200}"

# Email alerts (optional; configure in /etc/nanolab_backup.conf)
ALERT_EMAIL_TO="${ALERT_EMAIL_TO:-}"
ALERT_EMAIL_FROM="${ALERT_EMAIL_FROM:-nanolab-backup@$(hostname -f)}"
ALERT_SUBJECT_PREFIX="${ALERT_SUBJECT_PREFIX:-[NANOLAB BACKUP]}"

# Scheduling logic: 14 days
PERIOD_SEC=$((14 * 24 * 60 * 60))

HOME_DST="${BACKUP_ROOT}/home"
DATASETS_DST="${BACKUP_ROOT}/data-sets"
STATE_DIR="${BACKUP_ROOT}/.backup_state"
LOCK_FILE="${BACKUP_ROOT}/.backup_lock"

# rsync options:
# --delete (mirrors) not included since we want to keep a copy of the deleted files.
# This is safe ONLY if BACKUP_ROOT is correct + mounted.
RSYNC_OPTS=(-aH --numeric-ids --one-file-system)

# -----------------------
# Load config if present
# -----------------------
CONF="/etc/nanolab_backup.conf"
if [[ -f "$CONF" ]]; then
  # shellcheck disable=SC1090
  source "$CONF"
fi

# -----------------------
# Helpers
# -----------------------
log() {
  echo "[$(date '+%F %T')] $*"
}

send_alert() {
  local message="$1"
  if [[ -z "${ALERT_EMAIL_TO}" ]]; then
    log "ALERT_EMAIL_TO not set; skipping email alert."
    return 0
  fi

  local subject="${ALERT_SUBJECT_PREFIX} $(hostname -s) backup FAILED"
  if command -v mail >/dev/null 2>&1; then
    printf "%s\n" "$message" | mail -s "$subject" -r "$ALERT_EMAIL_FROM" "$ALERT_EMAIL_TO" || true
  elif command -v sendmail >/dev/null 2>&1; then
    {
      echo "From: ${ALERT_EMAIL_FROM}"
      echo "To: ${ALERT_EMAIL_TO}"
      echo "Subject: ${subject}"
      echo
      echo "$message"
    } | sendmail -t || true
  else
    log "No mail/sendmail found; cannot send alert email."
  fi
}

die() {
  local msg="$1"
  log "ERROR: $msg"
  send_alert "$msg"
  exit 1
}

need_root() {
  [[ $EUID -eq 0 ]] || die "Must run as root."
}

# Ensure BACKUP_ROOT is a mount point and writable
assert_backup_mounted() {
  if ! mountpoint -q "$BACKUP_ROOT"; then
    die "Backup root is NOT a mountpoint: $BACKUP_ROOT (aborting to avoid rsync-to-root-disk disasters)."
  fi
  [[ -w "$BACKUP_ROOT" ]] || die "Backup root not writable: $BACKUP_ROOT"
}

# Check free space on the filesystem containing BACKUP_ROOT
assert_free_space() {
  local avail_kb
  avail_kb="$(df -Pk "$BACKUP_ROOT" | awk 'NR==2 {print $4}')"
  [[ -n "$avail_kb" ]] || die "Could not determine free space for $BACKUP_ROOT"

  # Convert KB -> GiB (integer)
  local avail_gib=$(( avail_kb / 1024 / 1024 ))
  if (( avail_gib < MIN_FREE_GIB )); then
    die "Insufficient free space on backup disk: ${avail_gib} GiB available < ${MIN_FREE_GIB} GiB required."
  fi
  log "Free space OK: ${avail_gib} GiB available (min required ${MIN_FREE_GIB} GiB)."
}

ensure_paths() {
  mkdir -p "$HOME_DST" "$DATASETS_DST" "$STATE_DIR"
}

# Best-effort last login epoch seconds for a user; 0 if unknown/never
user_last_login_epoch() {
  local user="$1"
  local last_line last_field epoch

  if command -v lastlog >/dev/null 2>&1; then
    last_line="$(lastlog -u "$user" 2>/dev/null | tail -n 1 || true)"
    if echo "$last_line" | grep -qi "Never logged in"; then
      echo 0; return
    fi
    last_field="$(echo "$last_line" | sed -E 's/^'"$user"'\s+[^ ]+\s+[^ ]+\s+//')"
    epoch="$(date -d "$last_field" +%s 2>/dev/null || echo 0)"
    echo "$epoch"; return
  fi

  if command -v last >/dev/null 2>&1; then
    last_line="$(last -F "$user" 2>/dev/null | head -n 1 || true)"
    [[ -z "$last_line" ]] && { echo 0; return; }
    echo "$last_line" | grep -qi "wtmp begins" && { echo 0; return; }
    last_field="$(echo "$last_line" | sed -E 's/.*  ([A-Z][a-z]{2} [A-Z][a-z]{2} .*)$/\1/')"
    epoch="$(date -d "$last_field" +%s 2>/dev/null || echo 0)"
    echo "$epoch"; return
  fi

  echo 0
}

last_backup_epoch() {
  local user="$1"
  local stamp="${STATE_DIR}/${user}.last_backup"
  [[ -f "$stamp" ]] && cat "$stamp" 2>/dev/null || echo 0
}

set_last_backup_epoch() {
  local user="$1"
  date +%s > "${STATE_DIR}/${user}.last_backup"
}

should_backup_user() {
  local user="$1"
  local now last_bk last_login

  now="$(date +%s)"
  last_bk="$(last_backup_epoch "$user")"
  last_login="$(user_last_login_epoch "$user")"

  # Never backed up -> yes
  if [[ "$last_bk" -le 0 ]]; then
    return 0
  fi

  # Older than 14d -> yes
  if (( now - last_bk >= PERIOD_SEC )); then
    return 0
  fi

  # Last login older than 14d (known) -> yes (backup immediately)
  if [[ "$last_login" -gt 0 ]] && (( now - last_login >= PERIOD_SEC )); then
    return 0
  fi

  return 1
}

list_home_users() {
  find "$HOME_SRC" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort
}

backup_home_user() {
  local user="$1"
  local src="${HOME_SRC}/${user}/"
  local dst="${HOME_DST}/${user}/"

  [[ -d "$src" ]] || { log "SKIP: ${src} missing"; return; }
  mkdir -p "$dst"

  log "HOME backup: user=${user} ${src} -> ${dst}"
  rsync "${RSYNC_OPTS[@]}" "$src" "$dst"
  set_last_backup_epoch "$user"
  log "DONE: HOME user=${user}"
}

backup_datasets() {
  local src="${DATASETS_SRC}/"
  local dst="${DATASETS_DST}/"

  [[ -d "$DATASETS_SRC" ]] || { log "SKIP: ${DATASETS_SRC} missing"; return; }
  mkdir -p "$dst"

  log "DATASETS backup: ${src} -> ${dst}"
  rsync "${RSYNC_OPTS[@]}" "$src" "$dst"
  log "DONE: DATASETS"
}

# -----------------------
# Main (with non-overlap lock)
# -----------------------
need_root
assert_backup_mounted
assert_free_space
ensure_paths

# Lock: if another backup is running, exit cleanly
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "Another backup is already running; exiting."
  exit 0
fi

log "=== Backup started on $(hostname -f) ==="
log "BACKUP_ROOT=$BACKUP_ROOT"

# Home per user
while IFS= read -r user; do
  if should_backup_user "$user"; then
    backup_home_user "$user"
  else
    log "SKIP: user=${user} (recent backup + active within 14 days)"
  fi
done < <(list_home_users)

# Datasets
backup_datasets

log "=== Backup finished ==="