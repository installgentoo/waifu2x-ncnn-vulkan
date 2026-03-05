#!/usr/bin/env bash
set -euo pipefail

WAIFU_BIN=${WAIFU_BIN:-waifu2x-ncnn-vulkan}
WAIFU_STATE_DIR=${WAIFU_STATE_DIR:-/tmp/waifu2x-daemon}
WAIFU_IDLE_TIMEOUT=${WAIFU_IDLE_TIMEOUT:-120}
WAIFU_WAIT_TIMEOUT=${WAIFU_WAIT_TIMEOUT:-300}

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
case "$NOISE" in -1|0|1|2|3) ;; *) echo "Invalid -n: $NOISE" >&2; exit 2;; esac
case "$SCALE" in 1|2|4|8|16|32) ;; *) echo "Invalid -s: $SCALE" >&2; exit 2;; esac
case "$FORMAT" in png|jpg|webp) ;; *) echo "Invalid -f: $FORMAT" >&2; exit 2;; esac

mkdir -p "$WAIFU_STATE_DIR"
PIPE="$WAIFU_STATE_DIR/cmd.fifo"
LOCK="$WAIFU_STATE_DIR/control.lock"
DAEMON_PID_FILE="$WAIFU_STATE_DIR/daemon.pid"
KEEPER_PID_FILE="$WAIFU_STATE_DIR/keeper.pid"
WATCHDOG_PID_FILE="$WAIFU_STATE_DIR/watchdog.pid"
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
  local dpid kpid
  dpid=$(read_pid "$DAEMON_PID_FILE")
  kpid=$(read_pid "$KEEPER_PID_FILE")
  is_alive "$dpid" && kill "$dpid" 2>/dev/null || true
  is_alive "$kpid" && kill "$kpid" 2>/dev/null || true
  rm -f "$DAEMON_PID_FILE" "$KEEPER_PID_FILE"
}

start_server_unlocked() {
  rm -f "$PIPE"
  mkfifo "$PIPE"

  # Keep fifo open to prevent EOF exit when short-lived writers close.
  ( exec 9<>"$PIPE"; while :; do sleep 600; done ) &
  echo $! > "$KEEPER_PID_FILE"

  "$WAIFU_BIN" -D "$PIPE" -n "$NOISE" -s "$SCALE" -f "$FORMAT" >>"$DAEMON_LOG" 2>&1 &
  echo $! > "$DAEMON_PID_FILE"

  printf 'n=%s\ns=%s\nf=%s\n' "$NOISE" "$SCALE" "$FORMAT" > "$STARTUP_CFG_FILE"
  touch "$LAST_USE_FILE"
}

ensure_watchdog_unlocked() {
  local wpid
  wpid=$(read_pid "$WATCHDOG_PID_FILE")
  if is_alive "$wpid"; then
    return
  fi

  (
    while :; do
      sleep 5
      [[ -e "$LAST_USE_FILE" ]] || continue
      now=$(date +%s)
      last=$(date +%s -r "$LAST_USE_FILE" 2>/dev/null || echo 0)
      idle=$((now - last))
      if (( idle >= WAIFU_IDLE_TIMEOUT )); then
        exec 201>"$LOCK"
        flock 201
        stop_server_unlocked
        rm -f "$WATCHDOG_PID_FILE"
        exit 0
      fi
    done
  ) &
  echo $! > "$WATCHDOG_PID_FILE"
}

wait_for_status() {
  local status
  if ! status=$(timeout "${WAIFU_WAIT_TIMEOUT}s" bash -ceu 'IFS= read -r line < "$1"; printf "%s" "$line"' _ "$STATUS_PIPE"); then
    return 1
  fi

  [[ "$status" == "OK" ]] && return 0
  [[ "$status" == "FAIL" ]] && return 2
  return 1
}


exec 200>"$LOCK"
flock 200

dpid=$(read_pid "$DAEMON_PID_FILE")
alive=0
is_alive "$dpid" && alive=1 || true

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
  start_server_unlocked
fi

ensure_watchdog_unlocked

STATUS_PIPE="$WAIFU_STATE_DIR/status.$$.${RANDOM}.fifo"
rm -f "$STATUS_PIPE"
mkfifo "$STATUS_PIPE"

{
  printf '%s\0' -i "$INPUT" -o "$OUTPUT" -n "$NOISE" -s "$SCALE" -f "$FORMAT" -p "$STATUS_PIPE"
  printf '\0'
} > "$PIPE"
touch "$LAST_USE_FILE"

flock -u 200

if wait_for_status; then
  :
else
  rc=$?
  if (( rc == 2 )); then
    echo "waifu_client.sh: daemon reported failure for output: $OUTPUT" >&2
  else
    echo "waifu_client.sh: timed out waiting for daemon status: $OUTPUT" >&2
  fi
  exit 1
fi
