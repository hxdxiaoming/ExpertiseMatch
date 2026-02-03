#!/usr/bin/env bash
set -euo pipefail

# Orchestrate experiments per model name
# Usage examples:
#  TPMS:     bash run_experiment.sh --model tpms --dataset stelmakh --config configs/tpms.yml
#  BM25:     bash run_experiment.sh --model bm25 --dataset stelmakh --config configs/bm25.yml
#  SPECTER:  bash run_experiment.sh --model specter --dataset stelmakh --config configs/embedding/specter_profile_max_k10.yml
#  Qwen3:    bash run_experiment.sh --model qwen3 --dataset SIGIR --config configs/embedding/qwen3_profile_max.yml
#  CoF:      bash run_experiment.sh --model cof --dataset stelmakh --config_name rc_chain_max
# Notes:
#  - For embedding models (scibert, specter, scincl, specter2-*, coco-dr, gru), offline_preprocessing runs first, then generate+evaluate
#  - Qwen3: run offline_preprocessing + generate + evaluate all in qwen3-py39 to avoid numpy cache issues
#  - CoF: run get_paper_emb + chain in cof env; then evaluate in paper-reviewer

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

MODEL=""
DATASET=""
CONFIG_FILE=""
ENCODER=""
COF_CONFIG_NAME=""
RANKING_FILE_OVERRIDE=""

while [[ $# > 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --config) CONFIG_FILE="$2"; shift 2;;
    --encoder) ENCODER="$2"; shift 2;;
    --config_name) COF_CONFIG_NAME="$2"; shift 2;;
    --ranking_file) RANKING_FILE_OVERRIDE="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$MODEL" || -z "$DATASET" ]]; then
  echo "Usage: --model <tpms|bm25|scibert|specter|scincl|specter2-base|specter2-adhoc|specter2-proximity|specter2-classification|coco-dr|qwen3|gru|cof> --dataset <name> [--config <file>] [--encoder <name>] [--config_name <cof_name>] [--ranking_file <path>]" >&2
  exit 1
fi

# Helpers
activate_env() {
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$1"
}

get_task_type() {
  python - "$DATASET" <<'PY'
import json, os, sys
name=sys.argv[1]
name_l=name.lower()
meta=os.path.join('data', name_l, f'{name_l}_meta.json')
try:
    with open(meta,'r') as f:
        m=json.load(f)
    t=(m.get('task_type') or '').strip().lower()
    if t in ('paper-centric','paper_centric','paper'): print('paper-centric')
    elif t in ('reviewer-centric','reviewer_centric','reviewer'): print('reviewer-centric')
    else: print('reviewer-centric')
except Exception:
    print('reviewer-centric')
PY
}

resolve_default_config() {
  local m="$1"; local found=""
  case "$m" in
    tpms) found="configs/tpms.yml" ;;
    bm25) found="configs/bm25.yml" ;;
    scibert) found="configs/embedding/scibert_profile_max_k10.yml" ;;
    specter) found="configs/embedding/specter_profile_max_k10.yml" ;;
    scincl) found="configs/embedding/scincl_profile_max_k10.yml" ;;
    coco-dr) found="configs/embedding/coco_dr_profile_max_k10.yml" ;;
    gru) found="configs/gru.yml" ;;
    specter2-base) found="configs/embedding/specter2base_profile_max_k10.yml" ;;
    specter2-adhoc) found="configs/embedding/specter2adhoc_profile_max_k10.yml" ;;
    specter2-proximity) found="configs/embedding/specter2proximity_profile_max_k10.yml" ;;
    specter2-classification) found="configs/embedding/specter2classification_profile_max_k10.yml" ;;
    qwen3) found="configs/embedding/qwen3_profile_max.yml" ;;
  esac
  if [[ -n "$found" && -f "$found" ]]; then
    CONFIG_FILE="$found"; return 0
  fi
  local kw="$m"
  case "$m" in
    coco-dr) kw="coco" ;;
    specter2-*) kw="specter2" ;;
  esac
  if compgen -G "configs/**/${kw}*.yml" > /dev/null; then
    CONFIG_FILE="$(ls -1 configs/**/${kw}*.yml 2>/dev/null | head -n1)"; return 0
  fi
  if compgen -G "configs/**/${kw}*.yaml" > /dev/null; then
    CONFIG_FILE="$(ls -1 configs/**/${kw}*.yaml 2>/dev/null | head -n1)"; return 0
  fi
  return 1
}

capture_ranking_from_stdout() {
  local ranking=""
  while IFS= read -r line; do
    echo "$line"
    if [[ "$line" == *"Results saved to:"* ]]; then
      ranking=$(echo "$line" | sed -E 's/^.*Results saved to: *//')
      echo "$ranking" > .last_ranking_path.txt
    fi
  done
}

find_latest_cof_ranking() {
  python - "$DATASET" <<'PY'
import os, sys, glob
name=sys.argv[1].lower()
paths=glob.glob('results/**/*.json', recursive=True)
paths=[p for p in paths if name in p.lower() and 'ranking' in os.path.basename(p).lower()]
if not paths:
    sys.exit(1)
latest=max(paths, key=lambda p: os.path.getmtime(p))
print(latest)
PY
}

model_lc="$(echo "$MODEL" | tr 'A-Z' 'a-z')"
flow=""
case "$model_lc" in
  tpms|bm25) flow="direct";;
  qwen3) flow="qwen3"; ENCODER="${ENCODER:-qwen3}";;
  cof) flow="cof";;
  scibert|specter|scincl|coco-dr|gru) flow="embedding"; ENCODER="${ENCODER:-$model_lc}";;
  specter2-base) flow="embedding"; ENCODER="${ENCODER:-specter2base}";;
  specter2-adhoc) flow="embedding"; ENCODER="${ENCODER:-specter2adhoc}";;
  specter2-proximity) flow="embedding"; ENCODER="${ENCODER:-specter2proximity}";;
  specter2-classification) flow="embedding"; ENCODER="${ENCODER:-specter2classification}";;
  *) echo "Unsupported model: $MODEL" >&2; exit 1;;
 esac

run_direct() {
  if [[ -z "$CONFIG_FILE" ]]; then
    if ! resolve_default_config "$model_lc"; then
      echo "[ERROR] $MODEL requires --config <file>, and no default config was found." >&2; exit 1
    fi
  fi
  activate_env paper-reviewer
  python run_model_wrapper.py --mode generate --dataset_name "$DATASET" --config_file "$CONFIG_FILE" | capture_ranking_from_stdout
  local rf="${RANKING_FILE_OVERRIDE:-$(cat .last_ranking_path.txt)}"
  python run_model_wrapper.py --mode evaluate --dataset_name "$DATASET" --ranking_file "$rf"
}

run_embedding() {
  if [[ -z "$CONFIG_FILE" ]]; then
    if ! resolve_default_config "$model_lc"; then
      echo "[ERROR] $MODEL requires --config <file>, and no default config was found." >&2; exit 1
    fi
  fi
  activate_env paper-reviewer
  python offline_preprocessing.py --dataset "$DATASET" --embedding_models "$ENCODER"
  python run_model_wrapper.py --mode generate --dataset_name "$DATASET" --config_file "$CONFIG_FILE" | capture_ranking_from_stdout
  local rf="${RANKING_FILE_OVERRIDE:-$(cat .last_ranking_path.txt)}"
  python run_model_wrapper.py --mode evaluate --dataset_name "$DATASET" --ranking_file "$rf"
}

run_qwen3() {
  if [[ -z "$CONFIG_FILE" ]]; then
    if ! resolve_default_config "$model_lc"; then
      echo "[ERROR] qwen3 requires --config <file>, and no default config was found." >&2; exit 1
    fi
  fi
  activate_env qwen3-py39
  python offline_preprocessing.py --dataset "$DATASET" --embedding_models qwen3
  python run_model_wrapper.py --mode generate --dataset_name "$DATASET" --config_file "$CONFIG_FILE" | capture_ranking_from_stdout
  local rf="${RANKING_FILE_OVERRIDE:-$(cat .last_ranking_path.txt)}"
  python run_model_wrapper.py --mode evaluate --dataset_name "$DATASET" --ranking_file "$rf"
  conda deactivate
}

run_cof() {
  local task; task=$(get_task_type)
  if [[ -z "$COF_CONFIG_NAME" ]]; then
    if [[ "$task" == "paper-centric" ]]; then COF_CONFIG_NAME="pc_chain_max"; else COF_CONFIG_NAME="rc_chain_max"; fi
    echo "[INFO] CoF --config_name not provided, using default: $COF_CONFIG_NAME"
  fi
  activate_env cof
  python CoF-main/get_paper_emb.py --dataset "$DATASET" || python CoF-main/get_paper_emb.py --dataset_name "$DATASET" || true
  if [[ "$task" == "paper-centric" ]]; then
    python CoF-main/chain_of_factors_to_results.py --dataset "$DATASET" --config_name "$COF_CONFIG_NAME" || \
    python CoF-main/chain_of_factors_to_results.py --dataset_name "$DATASET" --config_name "$COF_CONFIG_NAME"
  else
    python CoF-main/chain_of_factors_reviewer_to_results.py --dataset "$DATASET" --config_name "$COF_CONFIG_NAME" || \
    python CoF-main/chain_of_factors_reviewer_to_results.py --dataset_name "$DATASET" --config_name "$COF_CONFIG_NAME"
  fi
  conda deactivate
  activate_env paper-reviewer
  local rf; if [[ -n "$RANKING_FILE_OVERRIDE" ]]; then rf="$RANKING_FILE_OVERRIDE"; else rf="$(find_latest_cof_ranking || true)"; fi
  if [[ -z "${rf:-}" || ! -f "$rf" ]]; then
    echo "[WARN] Could not auto-locate CoF ranking file. Provide --ranking_file <path> to evaluate. Skipping evaluation." >&2
    return 0
  fi
  python run_model_wrapper.py --mode evaluate --dataset_name "$DATASET" --ranking_file "$rf"
}

case "$flow" in
  direct)    run_direct ;;
  embedding) run_embedding ;;
  qwen3)     run_qwen3 ;;
  cof)       run_cof ;;
  *) echo "Unknown flow: $flow" >&2; exit 1;;
 esac
