#!/usr/bin/env bash
set -e

USER_ARGS=("$@")
set --

source ~/.bashrc
source /share/home/sxjiang/miniconda3/bin/activate
conda activate papo-q3

set -- "${USER_ARGS[@]}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${SCRIPT_DIR}/qwen3vl_vllm_offline_agent_eval.py" \
  --model-path /share/home/sxjiang/model/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking \
  --vstar-path /share/home/sxjiang/zhzhu/dataset/vstar_bench/test_questions.jsonl \
  --videomme-path /share/home/sxjiang/zhzhu/dataset/VideoMME/eval_template_copy.json \
  --output-dir "${REPO_DIR}/eval/results/qwen3vl_vllm_offline_agent" \
  --temperature 0.6 \
  --max-new-tokens 8192 \
  --max-turns 10 \
  --video-initial-frames 64 \
  --tensor-parallel-size 4 \
  "$@"
