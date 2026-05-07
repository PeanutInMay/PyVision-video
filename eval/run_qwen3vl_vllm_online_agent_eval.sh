#!/usr/bin/env bash
set -e

USER_ARGS=("$@")
set --

source ~/.bashrc
source /share/home/sxjiang/miniconda3/bin/activate
conda activate papo-q3

set -- "${USER_ARGS[@]}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/share/home/sxjiang/model/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-thinking-8b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/share/home/sxjiang/zhzhu/dataset}"
SERVER_LOG="${SERVER_LOG:-${REPO_DIR}/eval/results/qwen3vl_vllm_online_agent/server_${PORT}.log}"
VIDEO_INITIAL_FRAMES="${VIDEO_INITIAL_FRAMES:-64}"

for ((idx = 0; idx < ${#USER_ARGS[@]}; idx++)); do
  arg="${USER_ARGS[$idx]}"
  if [[ "${arg}" == "--video-initial-frames" && $((idx + 1)) -lt ${#USER_ARGS[@]} ]]; then
    VIDEO_INITIAL_FRAMES="${USER_ARGS[$((idx + 1))]}"
  elif [[ "${arg}" == --video-initial-frames=* ]]; then
    VIDEO_INITIAL_FRAMES="${arg#--video-initial-frames=}"
  fi
done

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${HOST}:${PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

mkdir -p "$(dirname "${SERVER_LOG}")"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
  --limit-mm-per-prompt '{"image": 16, "video": 1}' \
  --media-io-kwargs "{\"video\": {\"num_frames\": ${VIDEO_INITIAL_FRAMES}}}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Started vLLM server pid=${SERVER_PID}; log=${SERVER_LOG}"

READY_URL="http://${HOST}:${PORT}/v1/models"
SERVER_READY=0
for _ in $(seq 1 180); do
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    echo "vLLM server is ready: ${READY_URL}"
    SERVER_READY=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM server exited before ready. Last log lines:" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done

if [[ "${SERVER_READY}" != "1" ]]; then
  echo "Timed out waiting for vLLM server. Last log lines:" >&2
  tail -n 80 "${SERVER_LOG}" >&2 || true
  exit 1
fi

python "${SCRIPT_DIR}/qwen3vl_vllm_online_agent_eval.py" \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --openai-base-url "${OPENAI_BASE_URL}" \
  --openai-api-key "${OPENAI_API_KEY}" \
  --output-dir "${REPO_DIR}/eval/results/qwen3vl_vllm_online_agent" \
  --temperature 0.6 \
  --max-new-tokens 16384 \
  --max-turns 10 \
  --video-initial-frames "${VIDEO_INITIAL_FRAMES}" \
  --concurrency 16 \
  "$@"
