# EXPERMATCH: A Unified Benchmark for Bidirectional and Cross‑Domain Expertise Matching

This repository contains the official codebase for a unified expertise matching benchmark and system. It supports bidirectional matching (Paper‑to‑Reviewer and Reviewer‑to‑Paper) and cross‑domain matching (Project‑to‑Researcher), provides standardized generation and evaluation pipelines, and includes strong classical, embedding, and multi‑factor baselines.

## Overview
Expertise matching is fundamental to academic peer review and industrial innovation. Prior work has focused mostly on a single direction (paper‑centric), academic domain only, and fragmented evaluation settings. EXPERMATCH addresses these gaps with:
- Bidirectional tasks: Paper‑to‑Reviewer (P‑to‑R) and Reviewer‑to‑Paper (R‑to‑P)
- Cross‑domain task: Project‑to‑Researcher (J‑to‑R)
- Unified datasets, metrics, and evaluation
- A broad suite of models from classic IR to SOTA dense encoders and multi‑factor models

Our experiments reveal a clear quality‑efficiency trade‑off: dense embedding models yield the best quality but with higher latency; a context‑aware multi‑factor model (CoF) achieves competitive quality with substantially lower latency. Model choice should be guided by task and operational constraints.

## Tasks
We formulate matching as ranking. Given a query q and candidates C={c1..cn}, define a scoring function f(q,c) and rank candidates by relevance. We evaluate in zero‑shot settings.
- P‑to‑R (paper‑centric): query is a paper (title+abstract); candidates are reviewers (publication profiles)
- R‑to‑P (reviewer‑centric): query is a reviewer; candidates are papers
- J‑to‑R (cross‑domain): query is a project; candidates are researchers

Datasets declare their task type in `data/<dataset>/<dataset>_meta.json` and the pipeline adapts automatically.

## Datasets (public examples)
- SIGIR (P‑to‑R): 73 papers; aspect‑based graded labels, with raw/soft/hard variants
- Stelmakh (R‑to‑P): 58 reviewers; 1‑5 self‑assessed expertise scores; raw/soft/hard variants
- Additional P‑to‑R datasets (KDD, SciRepEval, NIPS) can be integrated following the same format
- J‑to‑R (industrial) is supported by the pipeline design; private data is not released

See `data/<dataset>/` for structure and `meta.json` for `task_type`.

## Data and Model Setup (数据集与模型放置说明)

**Folder structure.** The repo keeps full directory layout for `data/`, `models/`, and `results/`. Only lightweight files (configs, small datasets, `.gitkeep`) are tracked; large model weights and experiment outputs are not.

### Datasets (数据集)

- **Included (structure + meta only):** The repository keeps the folder structure and `*_meta.json` for SIGIR, KDD, NIPS, SciRepEval, Stelmakh under `data/<name>/`. The actual dataset files (`*_papers*.json`, `*_queries_*.json`, `*_reviewers*.json`) are not in the repo; place them in the same directories following the naming pattern so the pipeline can find them (see the linked benchmark or data sources for downloads).
- **Adding a new dataset:** Create a directory `data/<DatasetName>/` and place:
  - `<DatasetName>_meta.json` (with `task_type`: `p2r`, `r2p`, or `j2r`),
  - test papers/queries/reviewers JSONs following the same naming pattern as existing datasets.
- **Large or external data:** If you use datasets that are not in the repo, put them under `data/<DatasetName>/` with the same file names and structure so `run_experiment.sh` can find them.

### Models (模型)

Pre-trained model weights are **not** stored in the repository. Only the directory structure under `models/` is versioned (via `.gitkeep`). You need to place each model in the correct path before running:

| Model / baseline | Path under `models/` |
|------------------|------------------------|
| COCO-DR          | `models/coco-dr/base-msmarco/` |
| CoF (BiomedBERT) | `models/CoF/BiomedNLP-BiomedBERT-base-uncased-abstract/` |
| CoF checkpoint   | `models/CoF/cof.ckpt` |
| GRU              | `models/gru/model_checkpoint.pth` and `models/gru/tokenizer/` |
| SciBERT          | `models/scibert/` |
| SPECTER / SciNCL / Qwen3 | `models/sentence-transformers/specter/`, `scincl/`, `qwen3-0.6B/` |
| SPECTER2 base    | `models/specter2/base/` |
| SPECTER2 adapters| `models/specter2/adapters/adhoc/`, `proximity/`, `classification/` |

Download the official checkpoints from the respective sources (Hugging Face, author repos, etc.) and extract or copy them into the paths above so that configs in `configs/` point to the correct locations. Experiment outputs go to `results/` (not tracked); the folder is created automatically when you run experiments.

**Recommended Hugging Face model sources:**
- SPECTER: [`allenai/specter`](https://huggingface.co/allenai/specter)
- SPECTER2: [`allenai/specter2`](https://huggingface.co/allenai/specter2)
- Qwen3-Embedding-0.6B: [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- SciBERT (scivocab, uncased): [`allenai/scibert_scivocab_uncased`](https://huggingface.co/allenai/scibert_scivocab_uncased)
- SciNCL: [`malteos/scincl`](https://huggingface.co/malteos/scincl)
- COCO-DR (MS MARCO): [`OpenMatch/cocodr-base-msmarco`](https://huggingface.co/OpenMatch/cocodr-base-msmarco)
- BiomedBERT encoder for CoF: [`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)

**GRU model:**
- Pretrained GRU checkpoint (`model_checkpoint.pth`) used in the EXPERMATCH paper can be downloaded from Google Drive: [`gru.zip`](https://drive.google.com/file/d/1_PJYdLkBRkISH-B8sceh33bJpIuHeg4B/view?usp=sharing). After downloading and extracting, place `model_checkpoint.pth` under `models/gru/` and the tokenizer files under `models/gru/tokenizer/`.
- Alternatively, you can reproduce and finetune your own GRU model by following the methodology in the COLING 2025 demo paper *“Autonomous Machine Learning-Based Peer Reviewer Selection System”* ([link](https://aclanthology.org/2025.coling-demos.20/)).

**CoF checkpoint:**
- CoF multi-factor model weights (`cof.ckpt`) can be downloaded from Google Drive: [model.zip](https://drive.google.com/file/d/1n4fV6-K18V1nuLPGVBTbxsU78KDCnPLd/view). After downloading and extracting, place `cof.ckpt` under `models/CoF/`.

#### CoF-main internal model paths

The original CoF code under `CoF-main/` expects its own local model directory:

- **BiomedBERT encoder (for CoF-main scripts)**: `CoF-main/model/BiomedNLP-BiomedBERT-base-uncased-abstract/`  
  (you can copy the checkpoint downloaded from [`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) into this folder)
- **CoF checkpoint (for CoF-main chains)**: `CoF-main/model/cof.ckpt`  
  (you can reuse the same `cof.ckpt` as in `models/CoF/cof.ckpt`)

When running CoF via `run_experiment.sh` or CoF-main scripts, make sure the above paths exist so that `CoF-main/get_paper_emb.py` can resolve the local encoder and checkpoint correctly.

## Metrics
We report across graded and binary settings:
- Graded (raw): NDCG@k (primary), Kendall’s Tau, Spearman’s Rho
- Binary (soft/hard): MAP (primary), P@k, Recall@k, MRR@k, AUC‑ROC

## Models
We include a comprehensive suite of baselines:
- Classic IR: BM25, TPMS
- Scientific‑domain embeddings: SciBERT, SPECTER, SciNCL, SPECTER2 (Base, Adhoc, Proximity, Classification)
- General‑purpose embeddings: COCO‑DR, Qwen3‑Embedding‑0.6B
- Advanced: GRU (lightweight supervised), CoF (multi‑factor: semantic/topic/citation)

All embedding models use unified profile aggregation and score aggregation strategies (mean, max, weighted). CoF supports paper‑centric and reviewer‑centric chains with standardized outputs.

## Quality–Efficiency Trade‑off (Key Findings)
- Classic IR (BM25/TPMS): fastest by far; lower quality; strong baseline
- Embedding models: highest quality; higher latency (20–30× classic IR)
- CoF: competitive quality at much lower latency than dense encoders; a balanced middle‑ground

## Environments
We separate environments for robustness and compatibility:
- `paper-reviewer` (Python 3.8): default for BM25, TPMS, scientific embeddings, SPECTER2, COCO‑DR, GRU
- `qwen3-py39` (Python 3.9): Qwen3 end‑to‑end (offline + generate + evaluate) to avoid NumPy cache incompatibility
- `cof`: CoF (pip‑based installation)

One‑click setup:
```bash
bash setup.sh
```
Notes:
- If `en_core_web_sm` fails to download (network limits), skip or install wheel manually later:
  ```bash
  pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0-py3-none-any.whl
  ```

## Quick Start
Run standardized experiments via a single orchestrator:
- BM25 (auto default config):
  ```bash
  bash run_experiment.sh --model bm25 --dataset SIGIR
  ```
- TPMS:
  ```bash
  bash run_experiment.sh --model tpms --dataset SIGIR
  ```
- Embedding (e.g., SPECTER):
  ```bash
  bash run_experiment.sh --model specter --dataset SIGIR
  ```
- Qwen3 (runs entirely in qwen3 env):
  ```bash
  bash run_experiment.sh --model qwen3 --dataset SIGIR
  ```
- CoF (auto choose paper/reviewer chain from meta.json; default config_name if omitted):
  ```bash
  bash run_experiment.sh --model cof --dataset stelmakh
  ```

Outputs are written to `results/**/<method>/**/_ranking.json` and `_evaluation.json`.

## Orchestrator: `run_experiment.sh`
Supported models:
`tpms`, `bm25`, `scibert`, `specter`, `scincl`, `specter2-base`, `specter2-adhoc`, `specter2-proximity`, `specter2-classification`, `coco-dr`, `qwen3`, `gru`, `cof`.

Behavior:
- TPMS/BM25: generate + evaluate in `paper-reviewer`
- Embeddings/GRU: offline_preprocessing → generate → evaluate in `paper-reviewer`
- Qwen3: offline + generate + evaluate entirely in `qwen3-py39`
- CoF: run `CoF-main/get_paper_emb.py` and the appropriate chain in `cof`; then evaluate in `paper-reviewer`

Defaults:
- If `--config` is omitted, a default config is auto‑resolved (see `configs/`)
- CoF `--config_name` defaults to `pc_chain_max` (paper‑centric) or `rc_chain_max` (reviewer‑centric)

Examples:
```bash
# With defaults
bash run_experiment.sh --model specter --dataset SIGIR

# Custom config
bash run_experiment.sh --model scibert --dataset SIGIR --config configs/embedding/scibert_profile_mean_k25.yml

# CoF with explicit config_name
bash run_experiment.sh --model cof --dataset stelmakh --config_name rc_chain_max
```

## Reproducing Results
1) Install environments: `bash setup.sh`
2) Sanity check:
   - `bash run_experiment.sh --model bm25 --dataset SIGIR`
   - `bash run_experiment.sh --model specter --dataset SIGIR`
3) Full runs (SIGIR or stelmakh):
   - TPMS, BM25, SPECTER, SciBERT, SciNCL, SPECTER2 (Base/Adhoc/Proximity/Classification), COCO‑DR, GRU, Qwen3, CoF

All methods have been validated end‑to‑end on SIGIR and stelmakh (generate + evaluate).

## Third‑Party Code and Attribution
All code under the `CoF-main/` directory is adapted from the official repository of the paper “Chain‑of‑Factors Paper‑Reviewer Matching” (WWW 2025). We made minimal modifications to integrate it into this benchmark, standardize outputs (ranking JSONs compatible with `run_model_wrapper.py`), and streamline running scripts. Please cite the original work when using CoF‑related components.

## Troubleshooting
- NumPy cache error (`numpy._core` not found): occurs when loading cached embeddings across different Python/NumPy. Qwen3 is run fully in `qwen3-py39` to avoid cross‑env cache issues.
- spaCy model download failure: optional; install later via the wheel URL above.
- CoF outputs: chain scripts write standardized ranking JSONs, which are then evaluated via the unified evaluator.

## Citation
If you find this repository helpful, please cite the benchmark:
```bibtex
@inproceedings{expermatch2025,
  title   = {EXPERMATCH: A Unified Benchmark for Bidirectional and Cross-Domain Expertise Matching},
  author  = {Anonymous},
  year    = {2025},
  note    = {Code: https://github.com/hxdxiaoming/ExpertiseMatch}
}
```

And cite the CoF paper when using the `CoF-main/` components:
```bibtex
@inproceedings{cof2025,
  title   = {Chain-of-Factors Paper-Reviewer Matching},
  author  = {Zhang, Y. and Shen, Y. and Kang, S. and Chen, X. and Jin, B. and Han, J.},
  booktitle= {Proceedings of the Web Conference (WWW)},
  year    = {2025}
}
```

## License
MIT License. See `LICENSE` for details.
