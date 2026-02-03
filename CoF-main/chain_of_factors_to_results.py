#!/usr/bin/env python3
import os
import json
import time
import gc
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 保持与原实现一致的依赖
# import pytrec_eval  # 仅用于后续可能的本地评估（当前只生成排序文件，不做评估）

# 可选的内存测量（与 run_model_wrapper 风格一致）
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

def get_memory_usage_mb() -> float:
    """获取当前进程的内存使用量(MB)"""
    if not PSUTIL_AVAILABLE:
        return 0.0
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # 转换为MB
    except Exception:
        return 0.0

def measure_with_memory(func, *args, **kwargs):
    """
    执行函数并测量内存使用量
    
    Returns:
        (函数结果, 执行时间, 峰值内存使用量MB)
    """
    # 清理垃圾回收
    gc.collect()
    
    # 记录开始状态
    memory_before = get_memory_usage_mb()
    start_time = time.time()
    
    # 执行函数
    result = func(*args, **kwargs)
    
    # 记录结束状态
    end_time = time.time()
    memory_after = get_memory_usage_mb()
    
    execution_time = end_time - start_time
    peak_memory = max(memory_before, memory_after)
    
    return result, execution_time, peak_memory

def calculate_correct_latency(online_time: float, num_papers: int, num_reviewers: int, task_type: str = "paper-centric") -> float:
    """
    计算正确的在线延迟
    
    Args:
        online_time: 总在线时间(秒)
        num_papers: 论文数量
        num_reviewers: 审稿人数量
        task_type: 任务类型 ('paper-centric' 或 'reviewer-centric')
    
    Returns:
        每查询延迟(毫秒)
    """
    if task_type == 'paper-centric':
        # 每个论文需要与所有审稿人匹配
        total_queries = num_papers * num_reviewers
    elif task_type == 'reviewer-centric':
        # 每个审稿人需要与所有论文匹配
        total_queries = num_reviewers * num_papers
    else:
        # 默认情况，假设是论文数量
        total_queries = num_papers

    if total_queries == 0:
        return 0.0

    return (online_time * 1000) / total_queries


def load_embeddings(dataset: str):
    paper2emb_R = {}
    paper2emb_L = {}
    paper2emb_C = {}

    papers_path = f'data/{dataset}_papers_test.json'
    sem_path = f'embedding/{dataset}_paper_emb_semantic.txt'
    topic_path = f'embedding/{dataset}_paper_emb_topic.txt'
    cite_path = f'embedding/{dataset}_paper_emb_citation.txt'

    with open(papers_path) as fin1, \
         open(sem_path) as fin2, \
         open(topic_path) as fin3, \
         open(cite_path) as fin4:
        print('Getting paper embeddings...')
        for idx, (line1, line2, line3, line4) in enumerate(tqdm(zip(fin1, fin2, fin3, fin4))):
            data1 = json.loads(line1)
            paper = data1['paper']

            emb_R = np.array([float(x) for x in line2.strip().split()])
            emb_L = np.array([float(x) for x in line3.strip().split()])
            emb_C = np.array([float(x) for x in line4.strip().split()])

            paper2emb_R[paper] = emb_R
            paper2emb_L[paper] = emb_L
            paper2emb_C[paper] = emb_C

    return paper2emb_R, paper2emb_L, paper2emb_C


def load_reviewers(dataset: str):
    reviewer2papers = {}
    paper2reviewers = defaultdict(set)

    reviewers_path = f'data/{dataset}_reviewers_test.json'
    with open(reviewers_path) as fin:
        print('Getting reviewer profiles...')
        for line in tqdm(fin):
            data = json.loads(line)
            reviewer = data['reviewer']
            papers = data['papers']
            reviewer2papers[reviewer] = papers
            for paper in papers:
                paper2reviewers[paper].add(reviewer)

    all_reviewer_ids = list(reviewer2papers.keys())
    return reviewer2papers, paper2reviewers, all_reviewer_ids


def select_queries_file(dataset: str) -> str:
    # 优先 soft，其次 raw；去掉 hard
    soft = f'data/{dataset}_queries_test_soft.json'
    raw = f'data/{dataset}_queries_test_raw.json'
    if os.path.exists(soft):
        print(f"Using queries file: {soft}")
        return soft
    if os.path.exists(raw):
        print(f"Using queries file: {raw}")
        return raw
    raise FileNotFoundError(f"No queries file found for dataset {dataset} (expected soft or raw)")


def compute_ranking_lists_core(paper2emb_R, paper2emb_L, paper2emb_C, reviewer2papers, paper2reviewers, all_reviewer_ids, topn1: int, topn2: int, topk: int, queries_path: str):
    """核心计算函数（不包含数据加载，与run_model_wrapper一致）"""

    # 动态确定候选截断
    if topn1 > 0:
        topn1 = int(len(paper2reviewers) / topn1)
    else:
        topn1 = 1
    if topn2 > 0:
        topn2 = int(len(paper2reviewers) / topn2)
    else:
        topn2 = 1

    ranking_lists = {}

    print('Generate ranking lists...')
    with open(queries_path) as fin:
        for line in tqdm(fin):
            data = json.loads(line)
            query = data['query_id']
            q_emb_R = paper2emb_R[query]
            q_emb_C = paper2emb_C[query]
            q_emb_L = paper2emb_L[query]

            # 第一阶段：按R相似挑选候选paper
            p_score = {}
            for paper in paper2reviewers:
                p_emb_R = paper2emb_R[paper]
                p_score[paper] = float(np.dot(q_emb_R, p_emb_R))
            p_score_sorted = sorted(p_score.items(), key=lambda x: x[1], reverse=True)[:topn1]
            candidates_papers = [x[0] for x in p_score_sorted]

            # 第二阶段：按C相似精排paper
            p_score = {}
            for paper in candidates_papers:
                p_emb_C = paper2emb_C[paper]
                p_score[paper] = float(np.dot(q_emb_C, p_emb_C))
            p_score_sorted = sorted(p_score.items(), key=lambda x: x[1], reverse=True)[:topn2]

            # 聚合到reviewer分数
            reviewer_scores_list = defaultdict(list)
            for k, _ in p_score_sorted:
                for reviewer in paper2reviewers[k]:
                    p_emb_R = paper2emb_R[k]
                    p_emb_C = paper2emb_C[k]
                    p_emb_L = paper2emb_L[k]
                    score = float(np.dot(q_emb_R, p_emb_R) + np.dot(q_emb_C, p_emb_C) + np.dot(q_emb_L, p_emb_L))
                    reviewer_scores_list[reviewer].append(score)

            # 最终reviewer打分（取topk均值）
            reviewer_final_scores = {}
            for reviewer_id in all_reviewer_ids:
                if reviewer_id in reviewer_scores_list:
                    v = reviewer_scores_list[reviewer_id]
                    v_sorted = sorted(v, reverse=True)[:topk]
                    reviewer_final_scores[reviewer_id] = float(sum(v_sorted) / len(v_sorted))
                else:
                    reviewer_final_scores[reviewer_id] = 0.0

            # 生成run_model_wrapper兼容的ranking条目
            candidates = [
                {"id": reviewer_id, "score": reviewer_final_scores[reviewer_id]}
                for reviewer_id in all_reviewer_ids
            ]
            candidates.sort(key=lambda x: x["score"], reverse=True)

            ranking_lists[query] = {
                "query_type": "paper",
                "candidates": candidates,
                "total_candidates": len(candidates)
            }

    return ranking_lists, len(paper2emb_R), len(reviewer2papers)


def compute_ranking_lists(dataset: str, topn1: int, topn2: int, topk: int, queries_path: str):
    """原始计算函数（包含数据加载，保持向后兼容）"""
    paper2emb_R, paper2emb_L, paper2emb_C = load_embeddings(dataset)
    reviewer2papers, paper2reviewers, all_reviewer_ids = load_reviewers(dataset)
    return compute_ranking_lists_core(paper2emb_R, paper2emb_L, paper2emb_C, 
                                    reviewer2papers, paper2reviewers, all_reviewer_ids,
                                    topn1, topn2, topk, queries_path)


def compute_ranking_lists_with_timing(dataset: str, topn1: int, topn2: int, topk: int, queries_path: str):
    """带精确时间测量的计算排序列表函数（与run_model_wrapper一致，不包含embedding加载时间）"""
    # 先加载数据（不计时，与run_model_wrapper一致）
    paper2emb_R, paper2emb_L, paper2emb_C = load_embeddings(dataset)
    reviewer2papers, paper2reviewers, all_reviewer_ids = load_reviewers(dataset)
    
    def _compute_ranking_lists():
        return compute_ranking_lists_core(paper2emb_R, paper2emb_L, paper2emb_C, 
                                        reviewer2papers, paper2reviewers, all_reviewer_ids,
                                        topn1, topn2, topk, queries_path)
    
    # 使用 measure_with_memory 进行精确测量（只测量核心计算，不包含数据加载）
    result, online_time, peak_memory_mb = measure_with_memory(_compute_ranking_lists)
    ranking_lists, total_papers, total_reviewers = result
    
    return ranking_lists, total_papers, total_reviewers, online_time, peak_memory_mb


def save_ranking_file(dataset: str, config_name: str, ranking_lists: dict, total_papers: int, total_reviewers: int, efficiency_metrics: dict, method_type: str = "CoF_Original"):
    dataset_lower = dataset.lower()
    # 输出到主目录results（脚本位于 CoF-main，下一级为项目根目录）
    repo_root_results = Path(__file__).resolve().parents[1] / 'results'
    output_path = repo_root_results / f"cof/{dataset}/{config_name}/{dataset_lower}_{config_name}_ranking.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 组装与run_model_wrapper兼容的结构
    experiment_config = {
        "config_file": f"generated_by_chain_of_factors_to_results:{config_name}",
        "dataset": dataset,
        "experiment_name": config_name,
        "method_type": method_type,
        "config": {
            "matcher_class": "CoFOriginalMatcher",
            "matcher_config": {
                "topn1": None,
                "topn2": None,
                "topk": None
            }
        }
    }

    data_info = {
        "total_papers": total_papers,
        "total_reviewers": total_reviewers,
        "papers_scored": len(ranking_lists),
        "task_type": "paper-centric",
        "qrel_format": "closed"
    }

    # 填充效率指标
    efficiency_metrics = efficiency_metrics or {
        "offline_time_seconds": 0.0,
        "online_latency_ms_per_query": 0.0,
        "total_online_time_seconds": 0.0,
        "total_experiment_time_seconds": 0.0,
        "memory_usage_mb": 0.0,
        "method_type": method_type,
        "debug_info": {
            "papers_to_score": data_info["papers_scored"],
            "total_reviewers": total_reviewers,
            "task_type": data_info["task_type"],
            "total_queries_computed": data_info["papers_scored"] * total_reviewers,
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
    parser = argparse.ArgumentParser(description='Generate run_model_wrapper-compatible ranking file from CoF logic')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config_name', required=True, help='用于results路径中的配置名称')
    parser.add_argument('--topn1', type=int, default=50)
    parser.add_argument('--topn2', type=int, default=100)
    parser.add_argument('--topk', type=int, default=3)
    args = parser.parse_args()

    queries_path = select_queries_file(args.dataset)

    # 使用改进的时间测量方法
    print("🎯 Computing ranking lists with precise timing...")
    ranking_lists, total_papers, total_reviewers, online_time, peak_memory_mb = compute_ranking_lists_with_timing(
        args.dataset, args.topn1, args.topn2, args.topk, queries_path
    )
    
    # 计算正确的延迟
    num_queries = len(ranking_lists)
    latency_ms = calculate_correct_latency(online_time, num_queries, total_reviewers, "paper-centric")

    efficiency_metrics = {
        "offline_time_seconds": 0.0,
        "online_latency_ms_per_query": latency_ms,
        "total_online_time_seconds": online_time,
        "total_experiment_time_seconds": online_time,
        "memory_usage_mb": peak_memory_mb,
        "method_type": "CoF_Original",
        "debug_info": {
            "papers_to_score": num_queries,
            "total_reviewers": total_reviewers,
            "task_type": "paper-centric",
            "total_queries_computed": num_queries * total_reviewers,
            "is_cached": False,
            "offline_memory_mb": 0.0,  # 简化处理，CoF没有离线阶段
            "online_memory_mb": peak_memory_mb
        }
    }

    save_ranking_file(
        dataset=args.dataset,
        config_name=args.config_name,
        ranking_lists=ranking_lists,
        total_papers=total_papers,
        total_reviewers=total_reviewers,
        efficiency_metrics=efficiency_metrics,
        method_type="CoF_Original"
    )


if __name__ == '__main__':
    main()
