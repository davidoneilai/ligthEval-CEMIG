#!/usr/bin/env bash
set -euo pipefail

MODELS_FILE="${MODELS_FILE:-configs/models.txt}"
TASKS_FILE="${TASKS_FILE:-configs/evals.txt}"
CUSTOM_TASKS_FILE="${CUSTOM_TASKS_FILE:-tasks/custom_energy_eval.py}"

EVAL_IMAGE="${EVAL_IMAGE:-david/vllm-cu130-qwen-nightly}"
OUT_ROOT="${OUT_ROOT:-outputs_lighteval_user}"
HF_TOKEN="${HF_TOKEN:-}"

DTYPE="${DTYPE:-bfloat16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TP_SIZE="${TP_SIZE:-1}"
GPU_DEVICE="${GPU_DEVICE:-4}"

mkdir -p "$OUT_ROOT"

while IFS= read -r MODEL || [[ -n "$MODEL" ]]; do
  MODEL="$(printf "%s" "$MODEL" | tr -d '\r' | xargs)"
  [[ -z "$MODEL" ]] && continue
  [[ "$MODEL" =~ ^# ]] && continue

  slug="$(printf "%s" "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/: .' '_')"
  model_out="${OUT_ROOT}/${slug}"
  mkdir -p "$model_out"

  if find "$model_out" -type f -name 'results_*.json' | grep -q .; then
    echo "[SKIP] $MODEL já possui results_*.json"
    continue
  fi

  echo "[RUN] MODEL=$MODEL TP_SIZE=$TP_SIZE"

  docker run --rm \
    --gpus "\"device=${GPU_DEVICE}\"" \
    --user "$(id -u):$(id -g)" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HOME=/tmp \
    -e USER=worker \
    -e LOGNAME=worker \
    -e USERNAME=worker \
    -e XDG_CACHE_HOME=/tmp/xdg-cache \
    -e HF_HOME=/workspace/.cache_user/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/workspace/.cache_user/huggingface/hub \
    -e HF_DATASETS_CACHE=/workspace/.cache_user/huggingface/datasets \
    -e PIP_CACHE_DIR=/workspace/.cache_user/pip \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    -e TRITON_CACHE_DIR=/tmp/triton \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -w /workspace \
    -v "$(pwd):/workspace" \
    "$EVAL_IMAGE" \
    bash -lc "
      mkdir -p \
        /tmp/.cache \
        /tmp/xdg-cache \
        /tmp/torchinductor \
        /tmp/triton \
        /workspace/.cache_user/huggingface/hub \
        /workspace/.cache_user/huggingface/datasets \
        /workspace/.cache_user/pip \
        /workspace/${model_out} && \
      lighteval vllm \
        \"model_name=${MODEL},dtype=${DTYPE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_length=${MAX_MODEL_LEN},tensor_parallel_size=${TP_SIZE},data_parallel_size=1,pipeline_parallel_size=1,trust_remote_code=True\" \
        \"./${TASKS_FILE}\" \
        --custom-tasks \"./${CUSTOM_TASKS_FILE}\" \
        --output-dir \"/workspace/${model_out}\" \
        --save-details
    "

  echo "[DONE] $MODEL"
done < "$MODELS_FILE"