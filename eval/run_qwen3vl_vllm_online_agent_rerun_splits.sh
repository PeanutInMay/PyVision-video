#!/usr/bin/env bash
set -e

USER_ARGS=("$@")
RUN_ARGS=()
SPLITS_ARG="videomme,vstar,hrbench,math"

for arg in "${USER_ARGS[@]}"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash eval/run_qwen3vl_vllm_online_agent_rerun_splits.sh [options passed to python eval]

Script-only options:
  --splits videomme,vstar,hrbench,math   Select splits to run. Example: --splits hrbench

Common python options to pass through:
  --resume
  --concurrency N
  --request-timeout-seconds N
  --rerun-from-results PATH
  --rerun-mode both|tool_errors|incorrect

Default GPU layout:
  videomme -> GPU 0, TP=1, port 8021
  vstar    -> GPU 1, TP=1, port 8022
  hrbench  -> GPU 2,3, TP=2, port 8023
  math     -> GPU 4,5, TP=2, port 8024
EOF
    exit 0
  fi
done

idx=0
while [[ "${idx}" -lt "${#USER_ARGS[@]}" ]]; do
  arg="${USER_ARGS[$idx]}"
  if [[ "${arg}" == "--splits" && $((idx + 1)) -lt ${#USER_ARGS[@]} ]]; then
    SPLITS_ARG="${USER_ARGS[$((idx + 1))]}"
    idx=$((idx + 2))
    continue
  elif [[ "${arg}" == --splits=* ]]; then
    SPLITS_ARG="${arg#--splits=}"
    idx=$((idx + 1))
    continue
  fi
  RUN_ARGS+=("${arg}")
  idx=$((idx + 1))
done

set --

source ~/.bashrc
source /share/home/sxjiang/miniconda3/bin/activate
conda activate papo-q3

set -- "${RUN_ARGS[@]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/share/home/sxjiang/model/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-thinking-8b}"
HOST="${HOST:-127.0.0.1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/share/home/sxjiang/zhzhu/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/eval/results/qwen3vl_vllm_online_agent/rerun_no_lvb_splits}"
RERUN_FROM_RESULTS="${RERUN_FROM_RESULTS:-${REPO_DIR}/eval/results/qwen3vl_vllm_online_agent/qwen3vl_vllm_online_agent_no_lvb.jsonl}"
VIDEO_INITIAL_FRAMES="${VIDEO_INITIAL_FRAMES:-64}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
MAX_TURNS="${MAX_TURNS:-10}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-900}"

for ((idx = 0; idx < ${#RUN_ARGS[@]}; idx++)); do
  arg="${RUN_ARGS[$idx]}"
  if [[ "${arg}" == "--video-initial-frames" && $((idx + 1)) -lt ${#RUN_ARGS[@]} ]]; then
    VIDEO_INITIAL_FRAMES="${RUN_ARGS[$((idx + 1))]}"
  elif [[ "${arg}" == --video-initial-frames=* ]]; then
    VIDEO_INITIAL_FRAMES="${arg#--video-initial-frames=}"
  elif [[ "${arg}" == "--concurrency" && $((idx + 1)) -lt ${#RUN_ARGS[@]} ]]; then
    CONCURRENCY="${RUN_ARGS[$((idx + 1))]}"
  elif [[ "${arg}" == --concurrency=* ]]; then
    CONCURRENCY="${arg#--concurrency=}"
  fi
done

mkdir -p "${OUTPUT_DIR}"

SERVER_PIDS=()
CLIENT_PIDS=()

cleanup() {
  for pid in "${CLIENT_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

split_devices() {
  case "$1" in
    videomme) echo "0" ;;
    vstar) echo "1" ;;
    hrbench) echo "2,3" ;;
    math) echo "4,5" ;;
    *) echo "Unknown split $1" >&2; exit 1 ;;
  esac
}

split_port() {
  case "$1" in
    videomme) echo "8021" ;;
    vstar) echo "8022" ;;
    hrbench) echo "8023" ;;
    math) echo "8024" ;;
    *) echo "Unknown split $1" >&2; exit 1 ;;
  esac
}

split_tp() {
  case "$1" in
    videomme|vstar) echo "1" ;;
    hrbench|math) echo "2" ;;
    *) echo "Unknown split $1" >&2; exit 1 ;;
  esac
}

split_datasets() {
  case "$1" in
    videomme) echo "videomme" ;;
    vstar) echo "vstar" ;;
    hrbench) echo "hrbench4k,hrbench8k" ;;
    math) echo "mathvista,mathvision" ;;
    *) echo "Unknown split $1" >&2; exit 1 ;;
  esac
}

start_server() {
  local split="$1"
  local devices="$(split_devices "${split}")"
  local port="$(split_port "${split}")"
  local tp="$(split_tp "${split}")"
  local log_path="${OUTPUT_DIR}/server_${split}_${port}.log"

  CUDA_VISIBLE_DEVICES="${devices}" vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${port}" \
    --tensor-parallel-size "${tp}" \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
    --limit-mm-per-prompt '{"image": 16, "video": 1}' \
    --media-io-kwargs "{\"video\": {\"num_frames\": ${VIDEO_INITIAL_FRAMES}}}" \
    >"${log_path}" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("${pid}")
  echo "Started ${split} server pid=${pid}; devices=${devices}; tp=${tp}; port=${port}; log=${log_path}"
}

wait_server() {
  local split="$1"
  local port="$(split_port "${split}")"
  local ready_url="http://${HOST}:${port}/v1/models"
  local log_path="${OUTPUT_DIR}/server_${split}_${port}.log"
  local ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "${ready_url}" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "${ready}" != "1" ]]; then
    echo "Timed out waiting for ${split} server. Last log lines:" >&2
    tail -n 80 "${log_path}" >&2 || true
    exit 1
  fi
  echo "${split} server is ready: ${ready_url}"
}

start_client() {
  local split="$1"
  shift
  local port="$(split_port "${split}")"
  local datasets="$(split_datasets "${split}")"
  local output_path="${OUTPUT_DIR}/ours_rerun_${split}.jsonl"
  local client_log="${OUTPUT_DIR}/client_${split}.log"

  python "${SCRIPT_DIR}/qwen3vl_vllm_online_agent_eval.py" \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --openai-base-url "http://${HOST}:${port}/v1" \
    --openai-api-key EMPTY \
    --output-dir "${OUTPUT_DIR}" \
    --output-path "${output_path}" \
    --datasets "${datasets}" \
    --rerun-from-results "${RERUN_FROM_RESULTS}" \
    --rerun-mode both \
    --temperature 0.6 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-turns "${MAX_TURNS}" \
    --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}" \
    --video-initial-frames "${VIDEO_INITIAL_FRAMES}" \
    --concurrency "${CONCURRENCY}" \
    "$@" \
    >"${client_log}" 2>&1 &
  local pid=$!
  CLIENT_PIDS+=("${pid}")
  echo "Started ${split} client pid=${pid}; datasets=${datasets}; output=${output_path}; log=${client_log}"
}

SPLITS=()
IFS=',' read -ra REQUESTED_SPLITS <<<"${SPLITS_ARG}"
for split in "${REQUESTED_SPLITS[@]}"; do
  split="${split// /}"
  [[ -z "${split}" ]] && continue
  case "${split}" in
    videomme|vstar|hrbench|math) SPLITS+=("${split}") ;;
    *) echo "Unknown split '${split}'. Valid splits: videomme,vstar,hrbench,math" >&2; exit 1 ;;
  esac
done

if [[ "${#SPLITS[@]}" -eq 0 ]]; then
  echo "No valid splits selected from --splits=${SPLITS_ARG}" >&2
  exit 1
fi

for split in "${SPLITS[@]}"; do
  start_server "${split}"
done

for split in "${SPLITS[@]}"; do
  wait_server "${split}"
done

for split in "${SPLITS[@]}"; do
  start_client "${split}" "$@"
done

FAILED=0
for pid in "${CLIENT_PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAILED=1
  fi
done

if [[ "${FAILED}" != "0" ]]; then
  echo "At least one rerun client failed. Check logs under ${OUTPUT_DIR}." >&2
  exit 1
fi

echo "All rerun splits finished. Results are under ${OUTPUT_DIR}."
