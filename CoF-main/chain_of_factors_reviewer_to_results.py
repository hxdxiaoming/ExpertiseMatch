#!/usr/bin/env python3
import os
import json
import time
import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm


def _read_jsonl_ids(file_path: str, id_key: str) -> List[str]:
	ids = []
	with open(file_path, 'r', encoding='utf-8') as f:
		for line in f:
			if not line.strip():
				continue
			obj = json.loads(line)
			ids.append(obj[id_key])
	return ids


def _resolve_path_candidates(dataset: str) -> Dict[str, str]:
	"""Resolve possible paths for data and embeddings (prefer CoF-main, fallback to root)."""
	root = Path(__file__).resolve().parents[1]
	cof_dir = Path(__file__).resolve().parent
	paths = {}

	# data files (papers/reviewers/queries)
	paths['papers_cof'] = str(cof_dir / f'data/{dataset}_papers_test.json')
	paths['reviewers_cof'] = str(cof_dir / f'data/{dataset}_reviewers_test.json')
	paths['queries_raw_cof'] = str(cof_dir / f'data/{dataset}_queries_test_raw.json')
	paths['queries_soft_cof'] = str(cof_dir / f'data/{dataset}_queries_test_soft.json')
	paths['queries_hard_cof'] = str(cof_dir / f'data/{dataset}_queries_test_hard.json')

	paths['papers_root'] = str(root / f'data/{dataset}/{dataset}_papers.json')
	paths['reviewers_root'] = str(root / f'data/{dataset}/{dataset}_reviewers.json')
	paths['queries_raw_root'] = str(root / f'data/{dataset}/{dataset}_queries_test_raw.json')
	paths['queries_soft_root'] = str(root / f'data/{dataset}/{dataset}_queries_test_soft.json')
	paths['queries_hard_root'] = str(root / f'data/{dataset}/{dataset}_queries_test_hard.json')

	# embeddings (prefer CoF-main)
	paths['emb_semantic'] = str(cof_dir / f'embedding/{dataset}_paper_emb_semantic.txt')
	paths['emb_topic'] = str(cof_dir / f'embedding/{dataset}_paper_emb_topic.txt')
	paths['emb_citation'] = str(cof_dir / f'embedding/{dataset}_paper_emb_citation.txt')

	return paths


def _first_existing(*candidates: str) -> str:
	for p in candidates:
		if p and os.path.exists(p):
			return p
	return ''


def load_embeddings(dataset: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
	paths = _resolve_path_candidates(dataset)
	papers_file = _first_existing(paths['papers_cof'], paths['papers_root'])
	if not papers_file:
		raise FileNotFoundError(f"papers file not found for dataset {dataset}")

	sem_path = paths['emb_semantic']
	topic_path = paths['emb_topic']
	cite_path = paths['emb_citation']
	if not (os.path.exists(sem_path) and os.path.exists(topic_path) and os.path.exists(cite_path)):
		raise FileNotFoundError("embedding files not found in CoF-main/embedding")

	paper2emb_R, paper2emb_L, paper2emb_C = {}, {}, {}
	with open(papers_file, 'r', encoding='utf-8') as fin1, \
		 open(sem_path, 'r', encoding='utf-8') as fin2, \
		 open(topic_path, 'r', encoding='utf-8') as fin3, \
		 open(cite_path, 'r', encoding='utf-8') as fin4:
		print('Getting paper embeddings...')
		for line1, line2, line3, line4 in tqdm(zip(fin1, fin2, fin3, fin4)):
			if not line1.strip():
				continue
			data1 = json.loads(line1)
			paper = data1['paper']

			emb_R = np.array([float(x) for x in line2.strip().split()])
			paper2emb_R[paper] = emb_R

			emb_L = np.array([float(x) for x in line3.strip().split()])
			paper2emb_L[paper] = emb_L

			emb_C = np.array([float(x) for x in line4.strip().split()])
			paper2emb_C[paper] = emb_C

	return paper2emb_R, paper2emb_L, paper2emb_C


def load_reviewers(dataset: str) -> Tuple[Dict[str, List[str]], List[str]]:
	paths = _resolve_path_candidates(dataset)
	reviewers_file = _first_existing(paths['reviewers_cof'], paths['reviewers_root'])
	if not reviewers_file:
		raise FileNotFoundError(f"reviewers file not found for dataset {dataset}")

	reviewer2papers = {}
	with open(reviewers_file, 'r', encoding='utf-8') as fin:
		print('Getting reviewer profiles...')
		for line in tqdm(fin):
			if not line.strip():
				continue
			obj = json.loads(line)
			reviewer2papers[obj['reviewer']] = obj['papers']

	all_reviewers = sorted(reviewer2papers.keys())
	return reviewer2papers, all_reviewers


def select_queries_file(dataset: str, qrel_priority: List[str]) -> str:
	paths = _resolve_path_candidates(dataset)
	cands = []
	for typ in qrel_priority:
		if typ == 'raw':
			cands.append(paths['queries_raw_cof'])
			cands.append(paths['queries_raw_root'])
		elif typ == 'soft':
			cands.append(paths['queries_soft_cof'])
			cands.append(paths['queries_soft_root'])
		elif typ == 'hard':
			cands.append(paths['queries_hard_cof'])
			cands.append(paths['queries_hard_root'])
		else:
			continue
	qfile = _first_existing(*cands)
	if not qfile:
		raise FileNotFoundError(f"No queries file found for dataset {dataset} with priority {qrel_priority}")
	return qfile


def _aggregate(scores: List[float], mode: str) -> float:
	if not scores:
		return 0.0
	if mode == 'max':
		return float(np.max(scores))
	else:
		return float(np.mean(scores))


def compute_reviewer_centric_rankings(dataset: str,
		paper2emb_R: Dict[str, np.ndarray],
		paper2emb_L: Dict[str, np.ndarray],
		paper2emb_C: Dict[str, np.ndarray],
		reviewer2papers: Dict[str, List[str]],
		all_reviewers: List[str],
		queries_file: str,
		topn1_ratio: float,
		topn2_ratio: float,
		aggregation: str) -> Dict[str, Dict]:
	"""Reviewer-centric three-stage ratio filtering with aggregation per factor.
	Returns ranking_lists keyed by reviewer_id.
	"""
	# Load all candidate papers IDs from embeddings dict (keys are consistent with papers file order)
	all_paper_ids = sorted(paper2emb_R.keys())

	# Read queries (we only need query ids for consistency/stats; logic scores papers vs reviewer)
	query_ids = _read_jsonl_ids(queries_file, 'query_id')
	print(f"Using {len(query_ids)} queries from: {queries_file}")

	ranking_lists = {}
	for reviewer_id in tqdm(all_reviewers, desc='Scoring reviewers'):
		reviewer_papers = reviewer2papers.get(reviewer_id, [])
		if not reviewer_papers:
			# still produce empty candidate list
			ranking_lists[reviewer_id] = {"query_type": "reviewer", "candidates": [], "total_candidates": 0}
			continue

		# Stage 1: semantic over all papers with aggregation against reviewer's papers
		sem_scores = {}
		for pid in all_paper_ids:
			qvec = paper2emb_R.get(pid)
			if qvec is None:
				continue
			vals = []
			for rp in reviewer_papers:
				rpvec = paper2emb_R.get(rp)
				if rpvec is not None:
					vals.append(float(np.dot(qvec, rpvec)))
			sem_scores[pid] = _aggregate(vals, aggregation)

		# keep topN by ratio
		n1 = max(1, int(len(sem_scores) * topn1_ratio))
		stage1 = sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)[:n1]
		candidates1 = [pid for pid, _ in stage1]

		# Stage 2: topic on candidates1
		topic_scores = {}
		for pid in candidates1:
			qvec = paper2emb_L.get(pid)
			if qvec is None:
				continue
			vals = []
			for rp in reviewer_papers:
				rpvec = paper2emb_L.get(rp)
				if rpvec is not None:
					vals.append(float(np.dot(qvec, rpvec)))
			topic_scores[pid] = _aggregate(vals, aggregation)

		n2 = max(1, int(len(topic_scores) * topn2_ratio))
		stage2 = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:n2]
		candidates2 = [pid for pid, _ in stage2]

		# Stage 3: citation on candidates2 (final scoring sum of three factors)
		final_scores = {}
		for pid in candidates2:
			sem = sem_scores.get(pid, 0.0)
			topv = topic_scores.get(pid, 0.0)
			qvec = paper2emb_C.get(pid)
			if qvec is None:
				cite = 0.0
			else:
				vals = []
				for rp in reviewer_papers:
					rpvec = paper2emb_C.get(rp)
					if rpvec is not None:
						vals.append(float(np.dot(qvec, rpvec)))
				cite = _aggregate(vals, aggregation)
			final_scores[pid] = float(sem + topv + cite)

		cands = [{"id": pid, "score": float(s)} for pid, s in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)]
		ranking_lists[reviewer_id] = {
			"query_type": "reviewer",
			"candidates": cands,
			"total_candidates": len(cands)
		}

	return ranking_lists


def save_ranking_file(dataset: str, config_name: str, ranking_lists: dict,
		total_papers: int, total_reviewers: int) -> str:
	dataset_lower = dataset.lower()
	repo_root_results = Path(__file__).resolve().parents[1] / 'results'
	output_path = repo_root_results / f"cof/{dataset}/{config_name}/{dataset_lower}_{config_name}_ranking.json"
	output_path.parent.mkdir(parents=True, exist_ok=True)

	experiment_config = {
		"config_file": f"generated_by_chain_of_factors_reviewer_to_results:{config_name}",
		"dataset": dataset,
		"experiment_name": config_name,
		"method_type": "CoF_ReviewerChain",
		"config": {
			"matcher_class": "CoFReviewerChain",
			"matcher_config": {
				"topn1_ratio": None,
				"topn2_ratio": None,
				"aggregation": None
			}
		}
	}

	data_info = {
		"total_papers": total_papers,
		"total_reviewers": total_reviewers,
		"papers_scored": total_reviewers,  # reviewer-centric: queries == reviewers
		"task_type": "reviewer-centric",
		"qrel_format": "closed"
	}

	efficiency_metrics = {
		"offline_time_seconds": 0.0,
		"online_latency_ms_per_query": 0.0,
		"total_online_time_seconds": 0.0,
		"total_experiment_time_seconds": 0.0,
		"memory_usage_mb": 0.0,
		"method_type": "CoF_ReviewerChain",
		"debug_info": {
			"papers_to_score": total_papers,
			"total_reviewers": total_reviewers,
			"task_type": "reviewer-centric",
			"is_cached": False,
			"offline_memory_mb": 0.0,
			"online_memory_mb": 0.0
		}
	}

	result = {
		"ranking_format_version": "1.0",
		"experiment_config": experiment_config,
		"data_info": data_info,
		"efficiency_metrics": efficiency_metrics,
		"ranking_lists": ranking_lists,
		"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
	}

	with open(output_path, 'w', encoding='utf-8') as f:
		json.dump(result, f, indent=2, ensure_ascii=False)

	print(f"✅ Ranking saved to: {output_path}")
	return str(output_path)


def main():
	parser = argparse.ArgumentParser(description='Reviewer-centric Chain-of-Factors with ratio filtering to results')
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--config_name')
	parser.add_argument('--topn1_ratio', type=float, default=0.3, help='first-stage keep ratio (semantic)')
	parser.add_argument('--topn2_ratio', type=float, default=0.7, help='second-stage keep ratio (topic)')
	parser.add_argument('--aggregation', choices=['max', 'mean'], default='max', help='aggregation across reviewer papers')
	parser.add_argument('--qrel_priority', type=str, default='raw,soft,hard')
	args = parser.parse_args()

	# auto-generate config_name if not provided
	if not args.config_name:
		def fmt_ratio(x: float) -> str:
			# 0.1 -> 010, 0.05 -> 005
			val = int(round(x * 100))
			return f"{val:03d}"
		args.config_name = f"rc_chain_{args.aggregation}_n1{fmt_ratio(args.topn1_ratio)}_n2{fmt_ratio(args.topn2_ratio)}"

	paper2emb_R, paper2emb_L, paper2emb_C = load_embeddings(args.dataset)
	reviewer2papers, all_reviewers = load_reviewers(args.dataset)
	queries_file = select_queries_file(args.dataset, [x.strip() for x in args.qrel_priority.split(',') if x.strip()])

	ranking_lists = compute_reviewer_centric_rankings(
		args.dataset,
		paper2emb_R,
		paper2emb_L,
		paper2emb_C,
		reviewer2papers,
		all_reviewers,
		queries_file,
		args.topn1_ratio,
		args.topn2_ratio,
		args.aggregation
	)

	save_ranking_file(
		dataset=args.dataset,
		config_name=args.config_name,
		ranking_lists=ranking_lists,
		total_papers=len(paper2emb_R),
		total_reviewers=len(all_reviewers)
	)


if __name__ == '__main__':
	main()
