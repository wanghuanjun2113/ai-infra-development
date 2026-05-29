#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/development.yaml"

FRAMEWORK="xllm"
REPO_PATH=""
SERVER_BIN=""
MODEL_NAME=""
MODEL_PATH=""
DRAFT_MODEL_PATH=""
VISIBLE_DEVICES="auto"
START_DEVICE=0
START_PORT=18150
NNODES=2
MASTER_NODE_ADDR="127.0.0.1:22345"
HCCL_IF_BASE_PORT=43433
LOG_DIR="log"
RUN_DIR=""
MAX_MEMORY_UTILIZATION=0.7
MAX_TOKENS_PER_BATCH=32768
MAX_SEQS_PER_BATCH=16
BLOCK_SIZE=128
COMMUNICATION_BACKEND="lccl"
MAX_TOKENS_PER_CHUNK_FOR_PREFILL=256
MAX_CONCURRENT_REQUESTS=30
NUM_SPECULATIVE_TOKENS=2
GIT_BRANCH="unknown"
GIT_COMMIT="unknown"
LAUNCH_LOG=""
HEALTHCHECK_TIMEOUT_SECONDS=1800
HEALTHCHECK_INTERVAL_SECONDS=5
SMOKE_TEST_TIMEOUT_SECONDS=120
DEPLOY_STATUS="INCONCLUSIVE"
HEALTHCHECK_STATUS="NOT_RUN"
SMOKE_TEST_STATUS="NOT_RUN"
SERVICE_MODEL_ID=""

log() {
  local message
  message="$(printf '[%(%Y-%m-%dT%H:%M:%SZ)T] %s\n' -1 "$*")"
  printf '%s\n' "$message"
  if [[ -n "${LAUNCH_LOG:-}" ]]; then
    printf '%s\n' "$message" >> "$LAUNCH_LOG"
  fi
}

die() {
  local message
  message="$(printf '[%(%Y-%m-%dT%H:%M:%SZ)T] ERROR: %s\n' -1 "$*")"
  printf '%s\n' "$message" >&2
  if [[ -n "${LAUNCH_LOG:-}" ]]; then
    printf '%s\n' "$message" >> "$LAUNCH_LOG"
  fi
  exit 1
}

yaml_get() {
  local path="$1"
  local default_value="${2:-}"

  awk -v want="$path" -v default_value="$default_value" '
    function trim(s) {
      sub(/^[[:space:]]+/, "", s)
      sub(/[[:space:]]+$/, "", s)
      return s
    }
    function unquote(s) {
      s = trim(s)
      if ((s ~ /^".*"$/) || (s ~ /^\047.*\047$/)) {
        s = substr(s, 2, length(s) - 2)
      }
      return s
    }
    /^[[:space:]]*($|#)/ { next }
    {
      line = $0
      indent = 0
      while (substr(line, indent + 1, 1) == " ") {
        indent++
      }
      level = int(indent / 2)
      content = trim(line)
      if (content !~ /^[^:]+:/) {
        next
      }

      key = content
      sub(/:.*/, "", key)
      key = trim(key)

      value = content
      sub(/^[^:]+:[[:space:]]*/, "", value)

      stack[level] = key
      for (i in stack) {
        if (i > level) {
          delete stack[i]
        }
      }

      full = stack[0]
      for (i = 1; i < level; i++) {
        full = full "." stack[i]
      }
      if (level > 0) {
        full = full "." key
      }

      if (value != "" && full == want) {
        print unquote(value)
        found = 1
        exit
      }
    }
    END {
      if (!found && default_value != "") {
        print default_value
      }
    }
  ' "$CONFIG_FILE"
}

yaml_first_child() {
  local path="$1"

  awk -v want="$path" '
    function trim(s) {
      sub(/^[[:space:]]+/, "", s)
      sub(/[[:space:]]+$/, "", s)
      return s
    }
    /^[[:space:]]*($|#)/ { next }
    {
      line = $0
      indent = 0
      while (substr(line, indent + 1, 1) == " ") {
        indent++
      }
      level = int(indent / 2)
      content = trim(line)
      if (content !~ /^[^:]+:/) {
        next
      }

      key = content
      sub(/:.*/, "", key)
      key = trim(key)

      value = content
      sub(/^[^:]+:[[:space:]]*/, "", value)

      stack[level] = key
      for (i in stack) {
        if (i > level) {
          delete stack[i]
        }
      }

      parent = stack[0]
      for (i = 1; i < level; i++) {
        parent = parent "." stack[i]
      }

      if (level > 0 && parent == want) {
        print key
        exit
      }
    }
  ' "$CONFIG_FILE"
}

abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$path"
  fi
}

count_visible_devices() {
  local devices="$1"
  if [[ -z "$devices" ]]; then
    echo 0
  else
    awk -F',' '{print NF}' <<< "$devices"
  fi
}

json_escape() {
  sed \
    -e 's/\\/\\\\/g' \
    -e 's/"/\\"/g' \
    -e 's/	/\\t/g'
}

csv_contains() {
  local csv="$1"
  local value="$2"
  local item

  IFS=',' read -r -a items <<< "$csv"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ "$item" == "$value" ]]; then
      return 0
    fi
  done
  return 1
}

source_env_file() {
  local file="$1"
  local nounset_was_enabled=0

  case "$-" in
    *u*) nounset_was_enabled=1; set +u ;;
  esac

  # shellcheck disable=SC1090
  source "$file"

  if (( nounset_was_enabled == 1 )); then
    set -u
  fi
}

load_config_defaults() {
  log "Reading config: $CONFIG_FILE"
  [[ -f "$CONFIG_FILE" ]] || die "config file not found: $CONFIG_FILE"

  FRAMEWORK="$(yaml_get current_framework xllm)"
  REPO_PATH="$(abs_path "$(yaml_get frameworks.xllm.path code/xllm)")"
  SERVER_BIN="$REPO_PATH/build/xllm/core/server/xllm"

  MODEL_NAME="$(yaml_get deploy.xllm.model "")"
  if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="$(yaml_get benchmark.model "")"
  fi
  if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="$(yaml_first_child models)"
  fi

  if [[ -n "$MODEL_NAME" ]]; then
    MODEL_PATH="$(yaml_get "models.$MODEL_NAME.path" "")"
    DRAFT_MODEL_PATH="$(yaml_get "models.$MODEL_NAME.draft_model_path" "")"
  fi

  VISIBLE_DEVICES="$(yaml_get deploy.xllm.visible_devices auto)"
  START_DEVICE="$(yaml_get deploy.xllm.start_device 0)"
  START_PORT="$(yaml_get deploy.xllm.start_port 18150)"
  NNODES="$(yaml_get deploy.xllm.nnodes 2)"
  MASTER_NODE_ADDR="$(yaml_get deploy.xllm.master_node_addr 127.0.0.1:22345)"
  HCCL_IF_BASE_PORT="$(yaml_get deploy.xllm.hccl_if_base_port 43433)"
  MAX_MEMORY_UTILIZATION="$(yaml_get deploy.xllm.max_memory_utilization 0.7)"
  MAX_TOKENS_PER_BATCH="$(yaml_get deploy.xllm.max_tokens_per_batch 32768)"
  MAX_SEQS_PER_BATCH="$(yaml_get deploy.xllm.max_seqs_per_batch 16)"
  BLOCK_SIZE="$(yaml_get deploy.xllm.block_size 128)"
  COMMUNICATION_BACKEND="$(yaml_get deploy.xllm.communication_backend lccl)"
  MAX_TOKENS_PER_CHUNK_FOR_PREFILL="$(yaml_get deploy.xllm.max_tokens_per_chunk_for_prefill 256)"
  MAX_CONCURRENT_REQUESTS="$(yaml_get deploy.xllm.max_concurrent_requests 30)"
  NUM_SPECULATIVE_TOKENS="$(yaml_get deploy.xllm.speculative_tokens 2)"

  if git -C "$REPO_PATH" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_BRANCH="$(git -C "$REPO_PATH" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    GIT_COMMIT="$(git -C "$REPO_PATH" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  fi

  if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$ROOT_DIR/runs/deploy/$(date -u +%Y%m%d_%H%M%S)_xllm_$GIT_COMMIT"
  fi
  LOG_DIR="$RUN_DIR"
}

print_config_summary() {
  log "Resolved launch config:"
  log "  framework=$FRAMEWORK"
  log "  repo=$REPO_PATH"
  log "  branch=$GIT_BRANCH commit=$GIT_COMMIT"
  log "  server_bin=$SERVER_BIN"
  log "  model_name=$MODEL_NAME"
  log "  model_path=$MODEL_PATH"
  log "  draft_model_path=$DRAFT_MODEL_PATH"
  log "  visible_devices=$VISIBLE_DEVICES"
  log "  start_device=$START_DEVICE start_port=$START_PORT nnodes=$NNODES"
  log "  master_node_addr=$MASTER_NODE_ADDR hccl_if_base_port=$HCCL_IF_BASE_PORT"
  log "  healthcheck_timeout_seconds=$HEALTHCHECK_TIMEOUT_SECONDS"
  log "  smoke_test_timeout_seconds=$SMOKE_TEST_TIMEOUT_SECONDS"
  log "  run_dir=$RUN_DIR"
}

select_idle_devices() {
  local need="$1"
  local npu_smi_file="$2"
  local idle_devices=""
  local busy_devices=""
  local detail=""
  local chip_id phy_id aicore hbm_used hbm_total threshold busy

  [[ -s "$npu_smi_file" ]] || die "npu-smi output is empty: $npu_smi_file"

  busy_devices="$(
    awk '
      /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]*[0-9]+[[:space:]]+\|/ {
        line = $0
        gsub(/[|]/, " ", line)
        n = split(line, f, /[[:space:]]+/)
        delete v
        c = 0
        for (i = 1; i <= n; i++) {
          if (f[i] != "") {
            v[++c] = f[i]
          }
        }
        if (c >= 3 && v[1] ~ /^[0-9]+$/ && v[2] ~ /^[0-9]+$/ && v[3] ~ /^[0-9]+$/) {
          phy_id = v[1] * 2 + v[2]
          if (!seen[phy_id]++) {
            if (out != "") {
              out = out ","
            }
            out = out phy_id
          }
        }
      }
      END { print out }
    ' "$npu_smi_file"
  )"

  while read -r chip_id phy_id aicore hbm_used hbm_total; do
    threshold=$((hbm_total / 5))
    if (( threshold < 8192 )); then
      threshold=8192
    fi
    busy=0
    if csv_contains "$busy_devices" "$phy_id"; then
      busy=1
    fi

    if [[ -n "$detail" ]]; then
      detail="$detail, "
    fi
    detail="${detail}${phy_id}(aicore=${aicore},hbm=${hbm_used}/${hbm_total},busy=${busy})"

    if (( busy == 0 && aicore <= 5 && hbm_used <= threshold )) && ! csv_contains "$idle_devices" "$phy_id"; then
      if [[ -z "$idle_devices" ]]; then
        idle_devices="$phy_id"
      else
        idle_devices="$idle_devices,$phy_id"
      fi
    fi
  done < <(
    awk '
      /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]+[0-9A-Fa-f:.]+[[:space:]]+\|/ {
        line = $0
        gsub(/[|\/]/, " ", line)
        n = split(line, f, /[[:space:]]+/)
        delete v
        c = 0
        for (i = 1; i <= n; i++) {
          if (f[i] != "") {
            v[++c] = f[i]
          }
        }
        if (c >= 8 && v[1] ~ /^[0-9]+$/ && v[2] ~ /^[0-9]+$/ && v[4] ~ /^[0-9]+$/ && v[7] ~ /^[0-9]+$/ && v[8] ~ /^[0-9]+$/) {
          print v[1], v[2], v[4], v[7], v[8]
        }
      }
    ' "$npu_smi_file"
  )

  if (( "$(count_visible_devices "$idle_devices")" < need )); then
    die "not enough idle NPU devices: need $need, idle ${idle_devices:-[]}; all cards: ${detail:-none}"
  fi

  awk -F',' -v need="$need" '
    {
      for (i = 1; i <= NF && i <= need; i++) {
        if (i > 1) {
          printf ","
        }
        printf "%s", $i
      }
      print ""
    }
  ' <<< "$idle_devices"
}

write_manifest() {
  local manifest="$LOG_DIR/manifest.json"
  local repo_json branch_json commit_json model_json model_path_json draft_json bin_json visible_json start_device_json start_port_json nnodes_json master_json run_json
  local status_json health_json smoke_json service_model_json

  repo_json="$(printf '%s' "$REPO_PATH" | json_escape)"
  branch_json="$(printf '%s' "$GIT_BRANCH" | json_escape)"
  commit_json="$(printf '%s' "$GIT_COMMIT" | json_escape)"
  model_json="$(printf '%s' "$MODEL_NAME" | json_escape)"
  model_path_json="$(printf '%s' "$MODEL_PATH" | json_escape)"
  draft_json="$(printf '%s' "$DRAFT_MODEL_PATH" | json_escape)"
  bin_json="$(printf '%s' "$SERVER_BIN" | json_escape)"
  visible_json="$(printf '%s' "$VISIBLE_DEVICES" | json_escape)"
  start_device_json="$(printf '%s' "$START_DEVICE" | json_escape)"
  start_port_json="$(printf '%s' "$START_PORT" | json_escape)"
  nnodes_json="$(printf '%s' "$NNODES" | json_escape)"
  master_json="$(printf '%s' "$MASTER_NODE_ADDR" | json_escape)"
  run_json="$(printf '%s' "$RUN_DIR" | json_escape)"
  status_json="$(printf '%s' "$DEPLOY_STATUS" | json_escape)"
  health_json="$(printf '%s' "$HEALTHCHECK_STATUS" | json_escape)"
  smoke_json="$(printf '%s' "$SMOKE_TEST_STATUS" | json_escape)"
  service_model_json="$(printf '%s' "$SERVICE_MODEL_ID" | json_escape)"

  cat > "$manifest" <<EOF
{
  "branch": "$branch_json",
  "commit": "$commit_json",
  "draft_model_path": "$draft_json",
  "framework": "xllm",
  "master_node_addr": "$master_json",
  "model": "$model_json",
  "model_path": "$model_path_json",
  "nnodes": "$nnodes_json",
  "repo": "$repo_json",
  "run_dir": "$run_json",
  "server_bin": "$bin_json",
  "service_model_id": "$service_model_json",
  "start_device": "$start_device_json",
  "start_port": "$start_port_json",
  "status": "$status_json",
  "healthcheck_status": "$health_json",
  "smoke_test_status": "$smoke_json",
  "task": "deploy",
  "visible_devices": "$visible_json"
}
EOF
}

write_report() {
  cat > "$LOG_DIR/report.md" <<EOF
# xllm Deploy Report

## Verdict

$DEPLOY_STATUS

## Context

- Framework: xllm
- Repo: $REPO_PATH
- Branch: $GIT_BRANCH
- Commit: $GIT_COMMIT
- Model: $MODEL_PATH
- Draft model: $DRAFT_MODEL_PATH
- Visible devices: $VISIBLE_DEVICES
- Start port: $START_PORT
- Nodes: $NNODES
- Run directory: $RUN_DIR
- Health check: $HEALTHCHECK_STATUS
- Smoke test: $SMOKE_TEST_STATUS
- Service model id: ${SERVICE_MODEL_ID:-unknown}
- Health check log: $LOG_DIR/healthcheck.log
- Smoke test log: $LOG_DIR/smoke_test.log
EOF
}

setup_environment() {
  local include_dir torch_dir

  log "Setting xllm runtime environment"

  for include_dir in /usr/local/include/python* /usr/include/python*; do
    if [[ -f "$include_dir/Python.h" ]]; then
      PYTHON_INCLUDE_PATH="$include_dir"
      case "$include_dir" in
        /usr/local/include/*) PYTHON_LIB_PATH="/usr/local/lib" ;;
        /usr/include/*) PYTHON_LIB_PATH="/usr/lib64" ;;
      esac
      export PYTHON_INCLUDE_PATH PYTHON_LIB_PATH
      break
    fi
  done

  for torch_dir in \
    /usr/local/lib/python*/site-packages/torch \
    /usr/local/lib64/python*/site-packages/torch \
    /usr/lib/python*/site-packages/torch \
    /usr/lib64/python*/site-packages/torch; do
    if [[ -d "$torch_dir" ]]; then
      PYTORCH_INSTALL_PATH="$torch_dir"
      export PYTORCH_INSTALL_PATH
      export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
      break
    fi
  done

  if [[ -z "${PYTHON_INCLUDE_PATH:-}" ]]; then
    log "WARN: Python include path not found from common shell globs"
  fi

  if [[ -z "${PYTORCH_INSTALL_PATH:-}" ]]; then
    log "WARN: PyTorch install path not found from common shell globs"
  fi

  export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
  export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:${LD_LIBRARY_PATH:-}

  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source_env_file /usr/local/Ascend/ascend-toolkit/set_env.sh
  else
    log "WARN: Ascend toolkit env file not found"
  fi

  if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source_env_file /usr/local/Ascend/nnal/atb/set_env.sh
  else
    log "WARN: ATB env file not found"
  fi

  export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
  export XLLM_PROFILING=1
  export ASDOPS_LOG_TO_STDOUT=1
  export ASDOPS_LOG_LEVEL=0
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export NPU_MEMORY_FRACTION=0.90
  export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
  export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
  export OMP_NUM_THREADS=12
  export HCCL_CONNECT_TIMEOUT=7200
  export INF_NAN_MODE_ENABLE=0
  export INF_NAN_MODE_FORCE_DISABLE=1
  export PROFILING_MODE=dynamic
  export HCCL_IF_BASE_PORT="$HCCL_IF_BASE_PORT"
  export TRITON_BINARY_PATH="$REPO_PATH/third_party/torch_npu_ops/triton_npu/binary"

  log "Environment ready: ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  log "Triton NPU binary path: TRITON_BINARY_PATH=$TRITON_BINARY_PATH"
}

parse_config_arg_first() {
  local arg_i next_i
  for (( arg_i=1; arg_i<=$#; arg_i++ )); do
    if [[ "${!arg_i}" == "--config" ]]; then
      next_i=$((arg_i + 1))
      CONFIG_FILE="${!next_i}"
      if [[ "$CONFIG_FILE" != /* ]]; then
        CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
      fi
      break
    fi
  done
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG_FILE="$2"
        if [[ "$CONFIG_FILE" != /* ]]; then
          CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
        fi
        shift 2
        ;;
      --server-bin) SERVER_BIN="$2"; shift 2 ;;
      --model) MODEL_PATH="$2"; shift 2 ;;
      --draft-model) DRAFT_MODEL_PATH="$2"; shift 2 ;;
      --visible-devices) VISIBLE_DEVICES="$2"; shift 2 ;;
      --start-device) START_DEVICE="$2"; shift 2 ;;
      --start-port) START_PORT="$2"; shift 2 ;;
      --nnodes) NNODES="$2"; shift 2 ;;
      --master-node-addr) MASTER_NODE_ADDR="$2"; shift 2 ;;
      --hccl-if-base-port) HCCL_IF_BASE_PORT="$2"; shift 2 ;;
      --log-dir) RUN_DIR="$2"; LOG_DIR="$2"; shift 2 ;;
      --run-dir) RUN_DIR="$2"; LOG_DIR="$2"; shift 2 ;;
      --max-memory-utilization) MAX_MEMORY_UTILIZATION="$2"; shift 2 ;;
      --max-tokens-per-batch) MAX_TOKENS_PER_BATCH="$2"; shift 2 ;;
      --max-seqs-per-batch) MAX_SEQS_PER_BATCH="$2"; shift 2 ;;
      --block-size) BLOCK_SIZE="$2"; shift 2 ;;
      --communication-backend) COMMUNICATION_BACKEND="$2"; shift 2 ;;
      --max-tokens-per-chunk-for-prefill) MAX_TOKENS_PER_CHUNK_FOR_PREFILL="$2"; shift 2 ;;
      --max-concurrent-requests) MAX_CONCURRENT_REQUESTS="$2"; shift 2 ;;
      --num-speculative-tokens) NUM_SPECULATIVE_TOKENS="$2"; shift 2 ;;
      --healthcheck-timeout-seconds) HEALTHCHECK_TIMEOUT_SECONDS="$2"; shift 2 ;;
      --healthcheck-interval-seconds) HEALTHCHECK_INTERVAL_SECONDS="$2"; shift 2 ;;
      --smoke-test-timeout-seconds) SMOKE_TEST_TIMEOUT_SECONDS="$2"; shift 2 ;;
      -h|--help)
        cat <<'USAGE'
Usage: scripts/launch_xllm.sh [options]

By default this script reads development.yaml, creates a deploy run directory,
selects idle NPU devices, exports xllm runtime variables, and starts xllm.

Common options:
  --config PATH
  --server-bin PATH
  --model PATH
  --draft-model PATH
  --visible-devices IDS|auto
  --start-device N
  --start-port PORT
  --nnodes N
  --master-node-addr HOST:PORT
  --run-dir DIR
  --healthcheck-timeout-seconds SECONDS
  --smoke-test-timeout-seconds SECONDS
USAGE
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
  done

  if [[ "$RUN_DIR" != /* ]]; then
    RUN_DIR="$ROOT_DIR/$RUN_DIR"
    LOG_DIR="$RUN_DIR"
  fi
}

validate_config() {
  [[ "$FRAMEWORK" == "xllm" ]] || die "current_framework must be xllm, got: $FRAMEWORK"
  [[ -n "$SERVER_BIN" ]] || die "missing server binary path"
  [[ -n "$MODEL_PATH" ]] || die "missing model path; configure benchmark.model or deploy.xllm.model"
  [[ -x "$SERVER_BIN" ]] || die "xllm server binary not found or not executable: $SERVER_BIN; build xllm first with: python setup.py build"
  [[ "$NNODES" =~ ^[0-9]+$ ]] && (( NNODES > 0 )) || die "nnodes must be a positive integer: $NNODES"
  [[ "$START_PORT" =~ ^[0-9]+$ ]] || die "start_port must be an integer: $START_PORT"
  [[ "$START_DEVICE" =~ ^[0-9]+$ ]] || die "start_device must be an integer: $START_DEVICE"
  [[ "$HEALTHCHECK_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] && (( HEALTHCHECK_TIMEOUT_SECONDS > 0 )) || die "healthcheck timeout must be a positive integer: $HEALTHCHECK_TIMEOUT_SECONDS"
  [[ "$HEALTHCHECK_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] && (( HEALTHCHECK_INTERVAL_SECONDS > 0 )) || die "healthcheck interval must be a positive integer: $HEALTHCHECK_INTERVAL_SECONDS"
  [[ "$SMOKE_TEST_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] && (( SMOKE_TEST_TIMEOUT_SECONDS > 0 )) || die "smoke test timeout must be a positive integer: $SMOKE_TEST_TIMEOUT_SECONDS"
}

start_service() {
  local i port device log_file pid

  log "Starting $NNODES xllm process(es)"
  : > "$LOG_DIR/pids.txt"

  for (( i=0; i<NNODES; i++ )); do
    port=$((START_PORT + i))
    device=$((START_DEVICE + i))
    log_file="$LOG_DIR/node_$i.log"

    CMD=(
      "$SERVER_BIN"
      --model "$MODEL_PATH"
      --devices "npu:$device"
      --port "$port"
      --master_node_addr "$MASTER_NODE_ADDR"
      --nnodes "$NNODES"
      --max_memory_utilization "$MAX_MEMORY_UTILIZATION"
      --max_tokens_per_batch "$MAX_TOKENS_PER_BATCH"
      --max_seqs_per_batch "$MAX_SEQS_PER_BATCH"
      --block_size "$BLOCK_SIZE"
      --communication_backend "$COMMUNICATION_BACKEND"
      --enable_prefix_cache=false
      --enable_chunked_prefill=false
      --max_tokens_per_chunk_for_prefill "$MAX_TOKENS_PER_CHUNK_FOR_PREFILL"
      --enable_schedule_overlap=true
      --enable_graph=true
      --node_rank "$i"
      --enable_shm=0
      --task generate
      --max_concurrent_requests "$MAX_CONCURRENT_REQUESTS"
      --backend llm
    )

    if [[ -n "$DRAFT_MODEL_PATH" && "$NUM_SPECULATIVE_TOKENS" != "0" ]]; then
      CMD+=(
        --draft_model "$DRAFT_MODEL_PATH"
        --draft_devices "npu:$device"
        --num_speculative_tokens "$NUM_SPECULATIVE_TOKENS"
      )
    elif [[ -n "$DRAFT_MODEL_PATH" ]]; then
      log "MTP disabled for node_$i: draft_model_path is set but num_speculative_tokens=0"
    fi

    log "Starting node_$i: port=$port device=npu:$device log=$log_file"
    printf 'Command:' > "$log_file"
    printf ' %q' "${CMD[@]}" >> "$log_file"
    printf '\n\n' >> "$log_file"

    nohup "${CMD[@]}" >> "$log_file" 2>&1 &
    pid="$!"
    echo "$pid" >> "$LOG_DIR/pids.txt"
    log "node_$i pid=$pid"
  done

  log "Started $NNODES xllm process(es). Logs: $LOG_DIR"
  log "Health check URL: http://127.0.0.1:$START_PORT/v1/models"
}

all_processes_alive() {
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    kill -0 "$pid" >/dev/null 2>&1 || return 1
  done < "$LOG_DIR/pids.txt"
}

extract_service_model_id() {
  local models_file="$1"
  local parsed=""

  parsed="$(
    sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$models_file" | head -n 1
  )"

  if [[ -n "$parsed" ]]; then
    SERVICE_MODEL_ID="$parsed"
  elif [[ -n "$MODEL_NAME" ]]; then
    SERVICE_MODEL_ID="$MODEL_NAME"
  else
    SERVICE_MODEL_ID="$(basename "$MODEL_PATH")"
  fi
}

wait_for_healthcheck() {
  local url="http://127.0.0.1:$START_PORT/v1/models"
  local deadline=$((SECONDS + HEALTHCHECK_TIMEOUT_SECONDS))
  local attempt=0
  local output="$LOG_DIR/healthcheck.log"
  local body="$LOG_DIR/models.json"
  local tmp="$LOG_DIR/healthcheck.tmp"

  command -v curl >/dev/null 2>&1 || die "curl command not found"

  log "Waiting for service readiness: url=$url timeout=${HEALTHCHECK_TIMEOUT_SECONDS}s interval=${HEALTHCHECK_INTERVAL_SECONDS}s"
  : > "$output"

  while (( SECONDS < deadline )); do
    attempt=$((attempt + 1))
    if ! all_processes_alive; then
      HEALTHCHECK_STATUS="FAIL"
      DEPLOY_STATUS="FAIL"
      write_manifest
      write_report
      die "xllm process exited before health check passed; inspect $LOG_DIR/node_*.log"
    fi

    if curl -fsS --max-time 10 "$url" > "$tmp" 2>&1; then
      mv "$tmp" "$body"
      {
        printf 'url=%s\n' "$url"
        printf 'attempt=%s\n' "$attempt"
        printf 'status=PASS\n\n'
        cat "$body"
        printf '\n'
      } > "$output"
      HEALTHCHECK_STATUS="PASS"
      extract_service_model_id "$body"
      log "Health check PASS: url=$url attempt=$attempt output=$output"
      log "Service model id for smoke test: $SERVICE_MODEL_ID"
      return 0
    fi

    {
      printf 'url=%s\n' "$url"
      printf 'attempt=%s\n' "$attempt"
      printf 'status=WAITING\n\n'
      cat "$tmp"
      printf '\n'
    } > "$output"
    rm -f "$tmp"
    log "Health check waiting: attempt=$attempt elapsed=${SECONDS}s"
    sleep "$HEALTHCHECK_INTERVAL_SECONDS"
  done

  HEALTHCHECK_STATUS="FAIL"
  DEPLOY_STATUS="FAIL"
  write_manifest
  write_report
  die "health check timeout after ${HEALTHCHECK_TIMEOUT_SECONDS}s: $url; latest output: $output"
}

run_smoke_test() {
  local url="http://127.0.0.1:$START_PORT/v1/chat/completions"
  local output="$LOG_DIR/smoke_test.log"
  local payload="$LOG_DIR/smoke_test.request.json"
  local response="$LOG_DIR/smoke_test.response.json"
  local escaped_model

  escaped_model="$(printf '%s' "$SERVICE_MODEL_ID" | json_escape)"

  cat > "$payload" <<EOF
{
  "model": "$escaped_model",
  "max_tokens": 8,
  "temperature": 0,
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": "hello xllm"
    }
  ]
}
EOF

  log "Running smoke test: POST $url model=$SERVICE_MODEL_ID timeout=${SMOKE_TEST_TIMEOUT_SECONDS}s"
  if curl -fsS --max-time "$SMOKE_TEST_TIMEOUT_SECONDS" \
      -H "Content-Type: application/json" \
      -d @"$payload" \
      "$url" > "$response" 2> "$output"; then
    {
      printf 'url=%s\n' "$url"
      printf 'request=%s\n' "$payload"
      printf 'response=%s\n' "$response"
      printf 'status=PASS\n\n'
      cat "$response"
      printf '\n'
    } >> "$output"
    SMOKE_TEST_STATUS="PASS"
    DEPLOY_STATUS="PASS"
    log "Smoke test PASS: output=$output response=$response"
    return 0
  fi

  SMOKE_TEST_STATUS="FAIL"
  DEPLOY_STATUS="FAIL"
  {
    printf 'url=%s\n' "$url"
    printf 'request=%s\n' "$payload"
    printf 'response=%s\n' "$response"
    printf 'status=FAIL\n'
  } >> "$output"
  write_manifest
  write_report
  die "smoke test failed: output=$output response=$response"
}

print_runtime_summary() {
  log "Launch summary:"
  log "  status=$DEPLOY_STATUS"
  log "  healthcheck_status=$HEALTHCHECK_STATUS"
  log "  smoke_test_status=$SMOKE_TEST_STATUS"
  log "  service_url=http://127.0.0.1:$START_PORT"
  log "  models_url=http://127.0.0.1:$START_PORT/v1/models"
  log "  chat_url=http://127.0.0.1:$START_PORT/v1/chat/completions"
  log "  service_model_id=${SERVICE_MODEL_ID:-unknown}"
  log "  run_dir=$RUN_DIR"
  log "  pids_file=$LOG_DIR/pids.txt"
  log "  visible_devices_file=$LOG_DIR/visible_devices.txt"
  log "  healthcheck_log=$LOG_DIR/healthcheck.log"
  log "  smoke_test_log=$LOG_DIR/smoke_test.log"
  log "  node_logs=$LOG_DIR/node_*.log"
}

main() {
  parse_config_arg_first "$@"
  load_config_defaults
  parse_args "$@"
  validate_config

  mkdir -p "$RUN_DIR"
  LAUNCH_LOG="$LOG_DIR/launch.log"
  : > "$LAUNCH_LOG"
  print_config_summary

  log "Capturing NPU status: $LOG_DIR/npu-smi.before.txt"
  command -v npu-smi >/dev/null 2>&1 || die "npu-smi command not found"
  npu-smi info > "$LOG_DIR/npu-smi.before.txt" 2>&1 || die "failed to run npu-smi info"

  if [[ "$VISIBLE_DEVICES" == "auto" ]]; then
    log "Selecting idle NPU devices for nnodes=$NNODES"
    VISIBLE_DEVICES="$(select_idle_devices "$NNODES" "$LOG_DIR/npu-smi.before.txt")"
  else
    log "Using configured visible devices: $VISIBLE_DEVICES"
  fi

  if (( "$(count_visible_devices "$VISIBLE_DEVICES")" < NNODES )); then
    die "visible devices are not enough for nnodes: visible=$VISIBLE_DEVICES nnodes=$NNODES"
  fi

  echo "$VISIBLE_DEVICES" > "$LOG_DIR/visible_devices.txt"
  log "Selected visible devices: $VISIBLE_DEVICES"

  write_manifest
  write_report
  log "Wrote deploy artifacts: manifest.json report.md visible_devices.txt"

  setup_environment
  start_service
  wait_for_healthcheck
  run_smoke_test
  write_manifest
  write_report
  log "Updated deploy artifacts: manifest.json report.md healthcheck.log smoke_test.log"
  print_runtime_summary
}

main "$@"
