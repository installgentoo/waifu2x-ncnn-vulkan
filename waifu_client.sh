#!/usr/bin/env bash
set -euo pipefail

WAIFU_BIN=${WAIFU_BIN:-waifu2x-ncnn-vulkan}
WAIFU_STATE_DIR=${WAIFU_STATE_DIR:-/tmp/waifu2x-daemon}
WAIFU_IDLE_TIMEOUT=${WAIFU_IDLE_TIMEOUT:-360}

usage() {
  cat <<USAGE
Usage: waifu_client.sh -i input -o output [-n noise] [-s scale] [-f format]
USAGE
}

INPUT=""
OUTPUT=""
NOISE=1
SCALE=2
FORMAT="png"
STATUS_PIPE=""

while getopts ":i:o:n:s:f:h" opt; do
  case "$opt" in
    i) INPUT="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    n) NOISE="$OPTARG" ;;
    s) SCALE="$OPTARG" ;;
    f) FORMAT="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Missing argument for -$OPTARG" >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

[[ -n "$INPUT" && -n "$OUTPUT" ]] || { usage >&2; exit 2; }
INPUT=$(realpath -- "$INPUT")
OUTPUT=$(realpath -m -- "$OUTPUT")
case "$NOISE" in -1|0|1|2|3) ;; *) echo "Invalid -n: $NOISE" >&2; exit 2;; esac
case "$SCALE" in 1|2|4|8|16|32) ;; *) echo "Invalid -s: $SCALE" >&2; exit 2;; esac
case "$FORMAT" in png|jpg|webp) ;; *) echo "Invalid -f: $FORMAT" >&2; exit 2;; esac

mkdir -p "$WAIFU_STATE_DIR"
PIPE="$WAIFU_STATE_DIR/cmd.fifo"
LOCK="$WAIFU_STATE_DIR/control.lock"
SUPERVISOR_PID_FILE="$WAIFU_STATE_DIR/supervisor.pid"
LAST_USE_FILE="$WAIFU_STATE_DIR/last_use"
STARTUP_CFG_FILE="$WAIFU_STATE_DIR/startup.cfg"
DAEMON_LOG="$WAIFU_STATE_DIR/daemon.log"

is_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }
read_pid() { [[ -s "$1" ]] && cat "$1" || true; }

cleanup_client_state() {
  [[ -n "$STATUS_PIPE" ]] && rm -f "$STATUS_PIPE"
}

trap cleanup_client_state EXIT
stop_server_unlocked() {
  local spid
  spid=$(read_pid "$SUPERVISOR_PID_FILE")
  is_alive "$spid" && kill "$spid" 2>/dev/null || true
  rm -f "$SUPERVISOR_PID_FILE"
}

start_server_unlocked() {
  rm -f "$PIPE"
  mkfifo "$PIPE"

  (
    cd "$WAIFU_STATE_DIR"

    _daemon_pid=""
    _cleanup() {
      [[ -n "$_daemon_pid" ]] && kill "$_daemon_pid" 2>/dev/null || true
      rm -f "$SUPERVISOR_PID_FILE"
    }
    trap _cleanup EXIT TERM INT

    # Hold fifo open to prevent EOF when short-lived writers close.
    exec 9<>"$PIPE"

    "$WAIFU_BIN" -D "$PIPE" -n "$NOISE" -s "$SCALE" -f "$FORMAT" >/dev/null 2>>"$DAEMON_LOG" &
    _daemon_pid=$!

    # Watchdog: idle-shutdown loop
    while :; do
      sleep 5
      [[ -e "$LAST_USE_FILE" ]] || continue
      now=$(date +%s)
      last=$(date +%s -r "$LAST_USE_FILE" 2>/dev/null || echo 0)
      if (( now - last >= WAIFU_IDLE_TIMEOUT )); then
        exec 201>"$LOCK"
        flock 201
        # Re-check after acquiring lock — a client may have sneaked in between
        # the idle check above and us acquiring the lock
        now=$(date +%s)
        last=$(date +%s -r "$LAST_USE_FILE" 2>/dev/null || echo 0)
        if (( now - last >= WAIFU_IDLE_TIMEOUT )); then
          break
        fi
        flock -u 201
      fi
    done
    # EXIT trap handles daemon kill and pid file cleanup
  ) &
  echo $! > "$SUPERVISOR_PID_FILE"

  printf 'n=%s\ns=%s\nf=%s\n' "$NOISE" "$SCALE" "$FORMAT" > "$STARTUP_CFG_FILE"
  touch "$LAST_USE_FILE"
}

wait_for_status() {
  local status
  IFS= read -r status < "$STATUS_PIPE" || return 1
  [[ "$status" == "OK" ]] && return 0
  [[ "$status" == "FAIL" ]] && return 2
  return 1
}

exec 200>"$LOCK"
flock 200

spid=$(read_pid "$SUPERVISOR_PID_FILE")
alive=0
is_alive "$spid" && alive=1 || true
originally_alive=$alive

if (( alive == 1 )) && [[ -s "$STARTUP_CFG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$STARTUP_CFG_FILE"
  if [[ "${n:-}" != "$NOISE" || \
        ( "${s:-}" == "1" && "$SCALE" != "1" ) || \
        ( "${s:-}" != "1" && "$SCALE" == "1" ) || \
        "${f:-}" != "$FORMAT" ]]; then
    echo "Warning: restarting daemon, scale/noise incompatible!"
    alive=0
  fi
fi

if (( alive == 0 )); then
  stop_server_unlocked
  (( originally_alive == 0 )) && rm -f "$DAEMON_LOG"
  start_server_unlocked
fi

STATUS_PIPE="$WAIFU_STATE_DIR/status.$$.$(uuidgen).fifo"
rm -f "$STATUS_PIPE"
mkfifo "$STATUS_PIPE"
{
  printf '%s\0' -i "$INPUT" -o "$OUTPUT" -n "$NOISE" -s "$SCALE" -f "$FORMAT" -p "$STATUS_PIPE"
  printf '\0'
} > "$PIPE"
touch "$LAST_USE_FILE"

flock -u 200

if ! wait_for_status; then
  echo "waifu_client.sh: daemon reported failure for output: $OUTPUT" >&2
  exit 1
fi
