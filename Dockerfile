FROM vllm/vllm-openai:v0.18.1-cu130

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    XDG_CACHE_HOME=/workspace/.cache \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python-is-python3 \
        git curl wget jq vim nano rsync \
        build-essential pkg-config cmake && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/.cache/pip \
             /workspace/.cache/huggingface \
             /workspace/.cache/torchinductor \
             /workspace/.cache/triton

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
        "git+https://github.com/huggingface/lighteval.git" \
        ray \
        more-itertools \
        langdetect \
        sentencepiece \
        immutabledict \
        "nltk>=3.9.1" \
        matplotlib \
        pandas

RUN python3 - <<'PY'
import importlib.metadata as md
import torch
print("torch:", torch.__version__)
print("vllm:", md.version("vllm"))
print("lighteval:", md.version("lighteval"))
PY

RUN python3 - <<'PY'
import pathlib
import lighteval.tasks.tasks.tiny_benchmarks as tb

target = pathlib.Path(tb.__file__).parent / "tinyBenchmarks.pkl"
print("tinyBenchmarks path:", target)
print("exists:", target.exists())
PY

RUN chmod a+r /usr/local/lib/python3.12/dist-packages/lighteval/tasks/tasks/tinyBenchmarks.pkl

WORKDIR /workspace
ENTRYPOINT []
CMD ["/bin/bash"]
