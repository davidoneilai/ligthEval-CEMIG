## Execução

### Buildar container
```bash
docker build -f Dockerfile -t vllm-ligtheval-cu130 .
```

### Task genérica

```bash
docker run --rm -it \
  --gpus '"device=XXX"' \
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
  -v "$(pwd):/workspace" \
  -w /workspace \
  vllm-LigthEval-cu130 \
  bash -lc '
    mkdir -p \
      /tmp/.cache \
      /tmp/xdg-cache \
      /tmp/torchinductor \
      /tmp/triton \
      /workspace/.cache_user/huggingface/hub \
      /workspace/.cache_user/huggingface/datasets \
      /workspace/.cache_user/pip \
      /workspace/test_lighteval_out && \
    lighteval vllm \
      "model_name=Qwen/Qwen3.5-0.8B,dtype=bfloat16,gpu_memory_utilization=0.35,max_model_length=2048,tensor_parallel_size=1,data_parallel_size=1,pipeline_parallel_size=1,trust_remote_code=True" \
      "ifeval" \
      --output-dir /workspace/test_lighteval_out \
      --save-details
  '
```

### Task regulatório-cemig

```bash
docker run --rm -it \
  --gpus '"device=XXX"' \
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
  -v "$(pwd):/workspace" \
  -w /workspace \
  vllm-LigthEval-cu130 \
  bash -lc '
    mkdir -p \
      /tmp/.cache \
      /tmp/xdg-cache \
      /tmp/torchinductor \
      /tmp/triton \
      /workspace/.cache_user/huggingface/hub \
      /workspace/.cache_user/huggingface/datasets \
      /workspace/.cache_user/pip \
      /workspace/test_energy_eval && \
    lighteval vllm \
      "model_name=Qwen/Qwen3.5-0.8B,dtype=bfloat16,gpu_memory_utilization=0.35,max_model_length=2048,tensor_parallel_size=1,data_parallel_size=1,pipeline_parallel_size=1,trust_remote_code=True" \
      "energy_eval|0" \
      --custom-tasks /workspace/tasks/custom_energy_eval.py \
      --output-dir /workspace/test_energy_eval \
      --save-details
  '
```

### Scripts

Se preferir, execute o script via bash (ex.: `bash scripts/run_lighteval_vllm.sh`), mas confira antes as variáveis de ambiente, principalmente `HF_TOKEN`, `GPU_DEVICE` e `EVAL_IMAGE`.

Para rodar em lote com `models.txt` e `evals.txt` explicitamente:

```bash
HF_TOKEN="$HF_TOKEN" \
GPU_DEVICE=0 \
EVAL_IMAGE=vllm-LigthEval-cu130 \
MODELS_FILE=configs/models.txt \
TASKS_FILE=configs/evals.txt \
CUSTOM_TASKS_FILE=tasks/custom_energy_eval.py \
OUT_ROOT=outputs_lighteval_user \
bash scripts/run_lighteval_vllm.sh
```

Formato exato com `docker run` (equivalente ao lote, lendo `models.txt` e usando `evals.txt`):

```bash
while IFS= read -r MODEL || [[ -n "$MODEL" ]]; do
  MODEL="$(printf "%s" "$MODEL" | tr -d '\r' | xargs)"
  [[ -z "$MODEL" ]] && continue
  [[ "$MODEL" =~ ^# ]] && continue

  slug="$(printf "%s" "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/: .' '_')"
  OUT_DIR="outputs_lighteval_user/${slug}"

  docker run --rm -it \
    --gpus '"device=0"' \
    -e MODEL="$MODEL" \
    -e OUT_DIR="$OUT_DIR" \
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
    -v "$(pwd):/workspace" \
    -w /workspace \
    vllm-LigthEval-cu130 \
    bash -lc '
      mkdir -p \
        /tmp/.cache \
        /tmp/xdg-cache \
        /tmp/torchinductor \
        /tmp/triton \
        /workspace/.cache_user/huggingface/hub \
        /workspace/.cache_user/huggingface/datasets \
        /workspace/.cache_user/pip \
        "/workspace/${OUT_DIR}" && \
      lighteval vllm \
        "model_name=${MODEL},dtype=bfloat16,gpu_memory_utilization=0.35,max_model_length=8192,tensor_parallel_size=1,data_parallel_size=1,pipeline_parallel_size=1,trust_remote_code=True" \
        "./configs/evals.txt" \
        --custom-tasks ./tasks/custom_energy_eval.py \
        --output-dir "/workspace/${OUT_DIR}" \
        --save-details
    '
done < configs/models.txt
```

Para inspecionar cada linha do benchmark em `outputs/`:

```bash
docker run --rm -it \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vllm-LigthEval-cu130 \
  python scripts/export_energy_outputs.py \
    --details-root test_energy_eval/details \
    --out-dir outputs
```

Arquivos gerados:
- `outputs/<timestamp>/rows.jsonl`: todas as linhas com prompt, alternativas, gold, predição e logprobs.
- `outputs/<timestamp>/wrong_only.jsonl`: somente linhas erradas.
- `outputs/<timestamp>/summary.json`: resumo da acurácia calculada a partir das linhas.

