#!/usr/bin/env bash
set -e

USER_ARGS=("$@")
RUN_ARGS=()
BASELINES="direct,perception_tool,perception_inline"

idx=0
while [[ "${idx}" -lt "${#USER_ARGS[@]}" ]]; do
  arg="${USER_ARGS[$idx]}"
  if [[ "${arg}" == "--baselines" && $((idx + 1)) -lt ${#USER_ARGS[@]} ]]; then
    BASELINES="${USER_ARGS[$((idx + 1))]}"
    idx=$((idx + 2))
    continue
  elif [[ "${arg}" == --baselines=* ]]; then
    BASELINES="${arg#--baselines=}"
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
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/share/home/sxjiang/zhzhu/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/eval/results/qwen3vl_vllm_online_baselines}"
VIDEO_INITIAL_FRAMES="${VIDEO_INITIAL_FRAMES:-32}"
CONCURRENCY="${CONCURRENCY:-32}"

for ((i = 0; i < ${#RUN_ARGS[@]}; i++)); do
  arg="${RUN_ARGS[$i]}"
  if [[ "${arg}" == "--video-initial-frames" && $((i + 1)) -lt ${#RUN_ARGS[@]} ]]; then
    VIDEO_INITIAL_FRAMES="${RUN_ARGS[$((i + 1))]}"
  elif [[ "${arg}" == --video-initial-frames=* ]]; then
    VIDEO_INITIAL_FRAMES="${arg#--video-initial-frames=}"
  elif [[ "${arg}" == "--concurrency" && $((i + 1)) -lt ${#RUN_ARGS[@]} ]]; then
    CONCURRENCY="${RUN_ARGS[$((i + 1))]}"
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

baseline_enabled() {
  local needle="$1"
  IFS=',' read -ra selected <<<"${BASELINES}"
  for item in "${selected[@]}"; do
    if [[ "${item// /}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

baseline_port() {
  case "$1" in
    direct) echo "8011" ;;
    perception_tool) echo "8012" ;;
    perception_inline) echo "8013" ;;
    *) echo "Unknown baseline $1" >&2; exit 1 ;;
  esac
}

baseline_devices() {
  case "$1" in
    direct) echo "0,1" ;;
    perception_tool) echo "2,3" ;;
    perception_inline) echo "4,5" ;;
    *) echo "Unknown baseline $1" >&2; exit 1 ;;
  esac
}

start_server() {
  local baseline="$1"
  local port="$2"
  local devices="$3"
  local log_path="${OUTPUT_DIR}/server_${baseline}_${port}.log"

  CUDA_VISIBLE_DEVICES="${devices}" vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${port}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
    --limit-mm-per-prompt '{"image": 16, "video": 1}' \
    --media-io-kwargs "{\"video\": {\"num_frames\": ${VIDEO_INITIAL_FRAMES}}}" \
    >"${log_path}" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("${pid}")
  echo "Started ${baseline} vLLM server pid=${pid}; devices=${devices}; port=${port}; log=${log_path}"
}

wait_server() {
  local baseline="$1"
  local port="$2"
  local ready_url="http://${HOST}:${port}/v1/models"
  local log_path="${OUTPUT_DIR}/server_${baseline}_${port}.log"
  local ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "${ready_url}" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "${ready}" != "1" ]]; then
    echo "Timed out waiting for ${baseline} server at ${ready_url}. Last log lines:" >&2
    tail -n 80 "${log_path}" >&2 || true
    exit 1
  fi
  echo "${baseline} server is ready: ${ready_url}"
}

start_client() {
  local baseline="$1"
  local port="$2"
  shift 2
  local output_path="${OUTPUT_DIR}/baseline_${baseline}.jsonl"
  local client_log="${OUTPUT_DIR}/client_${baseline}.log"
  python "${SCRIPT_DIR}/qwen3vl_vllm_online_baseline_eval.py" \
    --baseline "${baseline}" \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --openai-base-url "http://${HOST}:${port}/v1" \
    --openai-api-key EMPTY \
    --output-dir "${OUTPUT_DIR}" \
    --temperature 0.6 \
    --max-new-tokens 8192 \
    --max-turns 6 \
    --video-initial-frames "${VIDEO_INITIAL_FRAMES}" \
    --concurrency "${CONCURRENCY}" \
    "$@" \
    --output-path "${output_path}" \
    >"${client_log}" 2>&1 &
  local pid=$!
  CLIENT_PIDS+=("${pid}")
  echo "Started ${baseline} client pid=${pid}; output=${output_path}; log=${client_log}"
}

SELECTED_BASELINES=()
for baseline in direct perception_tool perception_inline; do
  if baseline_enabled "${baseline}"; then
    SELECTED_BASELINES+=("${baseline}")
  fi
done

if [[ "${#SELECTED_BASELINES[@]}" -eq 0 ]]; then
  echo "No valid baselines selected from --baselines=${BASELINES}" >&2
  exit 1
fi

for baseline in "${SELECTED_BASELINES[@]}"; do
  start_server "${baseline}" "$(baseline_port "${baseline}")" "$(baseline_devices "${baseline}")"
done

for baseline in "${SELECTED_BASELINES[@]}"; do
  wait_server "${baseline}" "$(baseline_port "${baseline}")"
done

for baseline in "${SELECTED_BASELINES[@]}"; do
  start_client "${baseline}" "$(baseline_port "${baseline}")" "$@"
done

FAILED=0
for pid in "${CLIENT_PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAILED=1
  fi
done

if [[ "${FAILED}" != "0" ]]; then
  echo "At least one baseline client failed. Check logs under ${OUTPUT_DIR}." >&2
  exit 1
fi

echo "All selected baselines finished. Results are under ${OUTPUT_DIR}."
