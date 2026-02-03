#!/usr/bin/env bash
set -euo pipefail

# Usage: bash setup.sh [--prefix /abs/conda/path]
# Creates three conda envs and installs project requirements:
#  - paper-reviewer (Python 3.8, requirements.txt)
#  - cof (Python 3.8, pip-only install from requirements_cof.txt with PyTorch handled)
#  - qwen3-py39 (Python 3.9, requirements_qwen3_py39.txt)

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
REQ_MAIN="$PROJECT_ROOT/requirements.txt"
REQ_COF="$PROJECT_ROOT/requirements_cof.txt"
REQ_QWEN3="$PROJECT_ROOT/requirements_qwen3_py39.txt"

ENV_MAIN="paper-reviewer"
ENV_COF="cof"
ENV_QWEN3="qwen3-py39"

PY_MAIN="3.8"
PY_QWEN3="3.9"

CONDA_PREFIX_PATH=""

while [[ ${1:-} != "" ]]; do
  case "$1" in
    --prefix)
      shift
      CONDA_PREFIX_PATH="$1"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift
done

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing requirements file: $f" >&2
    exit 1
  fi
}

activate_conda_env() {
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$1"
}

create_env_if_needed() {
  local env_name="$1"
  local py_ver="$2"
  if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "[INFO] Environment $env_name already exists. Skipping creation."
  else
    if [[ -n "$CONDA_PREFIX_PATH" ]]; then
      conda create -y -p "$CONDA_PREFIX_PATH/$env_name" python="$py_ver"
    else
      conda create -y -n "$env_name" python="$py_ver"
    fi
  fi
}

install_main_requirements() {
  local env_name="$ENV_MAIN"
  echo "[INFO] Installing requirements for $env_name from $REQ_MAIN"
  activate_conda_env "$env_name"
  pip install --no-input --upgrade pip
  pip install --no-input -r "$REQ_MAIN"
  python -V
  python -c "import sys; print('✅ Env OK:', '$env_name', sys.version)"
  conda deactivate
}

install_qwen3_requirements() {
  local env_name="$ENV_QWEN3"
  echo "[INFO] Installing requirements for $env_name from $REQ_QWEN3"
  activate_conda_env "$env_name"
  pip install --no-input --upgrade pip
  pip install --no-input -r "$REQ_QWEN3"
  python -V
  python -c "import sys; print('✅ Env OK:', '$env_name', sys.version)"
  conda deactivate
}

install_cof_stack() {
  local env_name="$ENV_COF"
  echo "[INFO] Installing CoF stack for $env_name (pip-only)"
  activate_conda_env "$env_name"

  pip install --no-input --upgrade pip

  # Try PyTorch cu113 wheels first (fast path); fallback to CPU wheels if unavailable
  set +e
  echo "[INFO] Installing PyTorch 1.10 cu113 wheels (attempt)"
  pip install --no-input --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.10.0+cu113 torchvision==0.11.0+cu113 || PYTORCH_FAIL=$?
  if [[ ${PYTORCH_FAIL:-0} -ne 0 ]]; then
    echo "[WARN] cu113 wheels not available; falling back to CPU wheels"
    pip install --no-input --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==1.10.2+cpu torchvision==0.11.3+cpu || \
      pip install --no-input torch==1.10.2 torchvision==0.11.3
  fi
  set -e

  # Additional key deps (from your reference), via pip
  pip install --no-input \
    sentencepiece \
    transformers==4.12.5 \
    typing_extensions==4.4.0 \
    ujson \
    hydra-core==1.1 \
    hydra-submitit-launcher==1.0.1 \
    pytorch-lightning \
    jsonlines \
    zmq \
    pytrec_eval \
    beir \
    dpr || true

  # Optional spaCy and English model; do not fail if model download is blocked
  set +e
  pip install --no-input spacy==3.3.1
  python -m spacy download en_core_web_sm --direct --timeout 60
  if [[ $? -ne 0 ]]; then
    echo "[WARN] spaCy model download failed; skipping en_core_web_sm"
  fi
  set -e

  # Install remaining requirements from requirements_cof.txt, filtering torch* and spaCy model wheel
  echo "[INFO] Installing remaining requirements from $REQ_COF (filtered)"
  TMP_REQ="$(mktemp)"
  grep -v -E '^(torch|torchvision|torchaudio)\b' "$REQ_COF" | \
    grep -v -E '^en-core-web-sm\s*@\s*https://github.com/explosion/spacy-models/' > "$TMP_REQ"
  pip install --no-input -r "$TMP_REQ" || true
  rm -f "$TMP_REQ"

  python -V
  python -c "import sys; import torch, transformers; print('✅ Env OK:', '$env_name', sys.version); print('torch', torch.__version__); print('transformers', transformers.__version__)"
  conda deactivate
}

# Validate requirement files
require_file "$REQ_MAIN"
require_file "$REQ_COF"
require_file "$REQ_QWEN3"

# Create/Update environments
create_env_if_needed "$ENV_MAIN" "$PY_MAIN"
install_main_requirements

create_env_if_needed "$ENV_COF" "$PY_MAIN"
install_cof_stack

create_env_if_needed "$ENV_QWEN3" "$PY_QWEN3"
install_qwen3_requirements

echo "[DONE] All environments prepared."
