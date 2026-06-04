#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PID=""
OUTPUT_DIR=""
WARMUP_CMD=""
WORKLOAD_CMD="python3 scripts/run_perf.py --level simple --case in128_out128_c1"
MSPROF_BIN="${MSPROF_BIN:-msprof}"
ATTACH_DELAY_SECONDS=2
STOP_DELAY_SECONDS=3
EXPORT_PROFILE=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/capture_xllm_profile.sh --pid <xllm_parent_pid> [options]

Options:
  --pid PID                 xLLM parent process PID to attach.
  --output-dir DIR          Output directory. Default: profiling/<timestamp>_xllm_profile.
  --warmup-cmd CMD          Command to run before msprof starts.
  --workload-cmd CMD        Command to run inside the profiling window.
                            Default: python3 scripts/run_perf.py --level simple --case in128_out128_c1
  --msprof PATH             msprof binary. Default: msprof or $MSPROF_BIN.
  --attach-delay-seconds N  Seconds to wait after msprof attach. Default: 2.
  --stop-delay-seconds N    Seconds to wait after sending stop. Default: 3.
  --no-export               Do not run msprof --export after capture.
  -h, --help                Show this help.

Before starting xLLM, make sure the service inherits PROFILING_MODE=dynamic.
This repository's scripts/launch_xllm.sh exports it for xLLM.
EOF
}

log() {
  local message
  message="$(printf '[%(%Y-%m-%dT%H:%M:%SZ)T] %s\n' -1 "$*")"
  printf '%s\n' "$message"
  if [[ -n "${CAPTURE_LOG:-}" ]]; then
    printf '%s\n' "$message" >> "$CAPTURE_LOG"
  fi
}

die() {
  log "ERROR: $*"
  exit 1
}

json_escape() {
  sed \
    -e 's/\\/\\\\/g' \
    -e 's/"/\\"/g' \
    -e 's/	/\\t/g'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid)
      PID="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --warmup-cmd)
      WARMUP_CMD="${2:-}"
      shift 2
      ;;
    --workload-cmd)
      WORKLOAD_CMD="${2:-}"
      shift 2
      ;;
    --msprof)
      MSPROF_BIN="${2:-}"
      shift 2
      ;;
    --attach-delay-seconds)
      ATTACH_DELAY_SECONDS="${2:-}"
      shift 2
      ;;
    --stop-delay-seconds)
      STOP_DELAY_SECONDS="${2:-}"
      shift 2
      ;;
    --no-export)
      EXPORT_PROFILE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$PID" ]] || { usage; exit 1; }
[[ "$PID" =~ ^[0-9]+$ ]] || die "--pid must be numeric: $PID"
kill -0 "$PID" 2>/dev/null || die "process not found: $PID"
command -v "$MSPROF_BIN" >/dev/null 2>&1 || die "msprof not found: $MSPROF_BIN"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT_DIR/profiling/$(date -u +%Y%m%d_%H%M%S)_xllm_profile"
elif [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi

if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  die "output directory is not empty: $OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"
CAPTURE_LOG="$OUTPUT_DIR/capture.log"
MSPROF_LOG="$OUTPUT_DIR/msprof.log"
EXPORT_LOG="$OUTPUT_DIR/export.log"
WORKLOAD_LOG="$OUTPUT_DIR/workload.log"
WARMUP_LOG="$OUTPUT_DIR/warmup.log"

PIPE_FILE="${TMPDIR:-/tmp}/xllm_msprof_pipe_$$"
MSPROF_PID=""
PIPE_OPEN=0

cleanup() {
  if [[ "$PIPE_OPEN" == "1" ]]; then
    printf 'quit\n' >&3 2>/dev/null || true
    exec 3>&- 2>/dev/null || true
    PIPE_OPEN=0
  fi
  if [[ -n "$MSPROF_PID" ]]; then
    wait "$MSPROF_PID" 2>/dev/null || true
    MSPROF_PID=""
  fi
  rm -f "$PIPE_FILE"
}
trap cleanup EXIT INT TERM

run_logged_command() {
  local name="$1"
  local command="$2"
  local logfile="$3"
  local rc

  log "Running $name: $command"
  set +e
  (
    cd "$ROOT_DIR"
    bash -lc "$command"
  ) 2>&1 | tee "$logfile"
  rc=${PIPESTATUS[0]}
  set -e
  log "$name exit_code=$rc"
  return "$rc"
}

snapshot_npu() {
  local output="$1"
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info > "$output" 2>&1 || true
  fi
}

write_manifest() {
  local status="$1"
  local latest_prof="${2:-}"
  local workbench_commit xllm_commit pid_cmd workload_json warmup_json status_json latest_json

  workbench_commit="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  xllm_commit="$(git -C "$ROOT_DIR/code/xllm" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  pid_cmd="$(ps -p "$PID" -o args= 2>/dev/null || true)"

  workload_json="$(printf '%s' "$WORKLOAD_CMD" | json_escape)"
  warmup_json="$(printf '%s' "$WARMUP_CMD" | json_escape)"
  status_json="$(printf '%s' "$status" | json_escape)"
  latest_json="$(printf '%s' "$latest_prof" | json_escape)"
  pid_cmd="$(printf '%s' "$pid_cmd" | json_escape)"

  cat > "$OUTPUT_DIR/manifest.json" <<EOF
{
  "framework": "xllm",
  "task": "profiling",
  "status": "$status_json",
  "pid": $PID,
  "pid_command": "$pid_cmd",
  "workbench_commit": "$workbench_commit",
  "xllm_commit": "$xllm_commit",
  "workload_cmd": "$workload_json",
  "warmup_cmd": "$warmup_json",
  "msprof": "$MSPROF_BIN",
  "output_dir": "$OUTPUT_DIR",
  "latest_prof_dir": "$latest_json"
}
EOF
}

log "xLLM profiling capture"
log "  pid=$PID"
log "  output_dir=$OUTPUT_DIR"
log "  msprof=$MSPROF_BIN"
log "  warmup_cmd=${WARMUP_CMD:-none}"
log "  workload_cmd=$WORKLOAD_CMD"
log "  export=$EXPORT_PROFILE"

snapshot_npu "$OUTPUT_DIR/npu_smi_before.txt"
write_manifest "RUNNING" ""

if [[ -n "$WARMUP_CMD" ]]; then
  run_logged_command "warmup" "$WARMUP_CMD" "$WARMUP_LOG" || die "warmup command failed"
fi

rm -f "$PIPE_FILE"
mkfifo "$PIPE_FILE"

log "Starting msprof dynamic attach"
"$MSPROF_BIN" \
  --dynamic=on \
  --output="$OUTPUT_DIR" \
  --model-execution=on \
  --runtime-api=on \
  --aicpu=on \
  --pid="$PID" < "$PIPE_FILE" > "$MSPROF_LOG" 2>&1 &
MSPROF_PID=$!

exec 3>"$PIPE_FILE"
PIPE_OPEN=1
sleep "$ATTACH_DELAY_SECONDS"

log "Sending msprof start"
printf 'start\n' >&3

set +e
run_logged_command "workload" "$WORKLOAD_CMD" "$WORKLOAD_LOG"
WORKLOAD_RC=$?
set -e

log "Sending msprof stop"
printf 'stop\n' >&3
sleep "$STOP_DELAY_SECONDS"
printf 'quit\n' >&3 2>/dev/null || true
exec 3>&-
PIPE_OPEN=0
wait "$MSPROF_PID" 2>/dev/null || true
MSPROF_PID=""

LATEST_PROF=""
for _ in {1..10}; do
  LATEST_PROF="$(ls -td "$OUTPUT_DIR"/PROF_* 2>/dev/null | head -1 || true)"
  if [[ -n "$LATEST_PROF" ]]; then
    break
  fi
  sleep 1
done
[[ -n "$LATEST_PROF" ]] || die "msprof did not create a PROF_* directory under $OUTPUT_DIR"

if (( EXPORT_PROFILE == 1 )); then
  log "Exporting profiling data: $LATEST_PROF"
  "$MSPROF_BIN" --export=on --output="$LATEST_PROF" 2>&1 | tee "$EXPORT_LOG"
fi

snapshot_npu "$OUTPUT_DIR/npu_smi_after.txt"

if (( WORKLOAD_RC != 0 )); then
  write_manifest "INCONCLUSIVE" "$LATEST_PROF"
  die "workload command failed with exit code $WORKLOAD_RC"
fi

write_manifest "PASS" "$LATEST_PROF"
log "Profiling capture completed"
log "  latest_prof_dir=$LATEST_PROF"
if [[ -d "$LATEST_PROF/mindstudio_profiler_output" ]]; then
  log "  exported_dir=$LATEST_PROF/mindstudio_profiler_output"
else
  log "  exported_dir=missing"
fi
