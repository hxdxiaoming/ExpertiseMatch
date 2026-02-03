#!/usr/bin/env python3
"""
run_model_wrapper.py - 统一的实验执行脚本

支持两种模式：
1. --mode generate: 只生成排序列表
2. --mode evaluate: 从排序列表生成评估指标
3. --mode full: 完整流程（生成+评估）

支持所有方法：BM25、Embedding等
"""

import argparse
import json
import os
import sys
import time
import gc
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# 添加内存监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available, memory usage will not be measured")

# 添加项目路径
sys.path.append('.')

from data_loader import DataLoader
from evaluation import MetricsCalculator
from utils.detailed_results import get_detailed_results_handler

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

def measure_with_memory(func, *args, **kwargs) -> Tuple[Any, float, float]:
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

def calculate_correct_latency(online_time: float, papers_to_score: pd.DataFrame,
                            reviewers_df: pd.DataFrame, task_type: str) -> float:
    """
    计算正确的在线延迟

    Args:
        online_time: 总在线时间(秒)
        papers_to_score: 需要评分的论文DataFrame
        reviewers_df: 审稿人DataFrame
        task_type: 任务类型 ('paper-centric' 或 'reviewer-centric')

    Returns:
        每查询延迟(毫秒)
    """
    if task_type == 'paper-centric':
        # 每个论文需要与所有审稿人匹配
        total_queries = len(papers_to_score) * len(reviewers_df)
    elif task_type == 'reviewer-centric':
        # 每个审稿人需要与所有论文匹配
        total_queries = len(reviewers_df) * len(papers_to_score)
    else:
        # 默认情况，假设是论文数量
        total_queries = len(papers_to_score)

    if total_queries == 0:
        return 0.0

    return (online_time * 1000) / total_queries

def load_config(config_file: str) -> Dict[str, Any]:
    """加载配置文件"""
    import yaml
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def _bm25_reviewer_centric_score(matcher, papers_to_score, reviewers_df, qrels_dict):
    """
    为reviewer-centric任务实现BM25评分
    
    Args:
        matcher: BM25Matcher实例
        papers_to_score: 需要评分的论文DataFrame
        reviewers_df: 审稿人DataFrame  
        qrels_dict: qrel数据字典
        
    Returns:
        (scores_df, detailed_results): 分数矩阵和详细结果
    """
    print("🔄 Implementing BM25 reviewer-centric scoring...")
    
    # 获取qrel中的审稿人列表
    first_qrel_type = list(qrels_dict.keys())[0]
    qrels_df = qrels_dict[first_qrel_type]
    target_reviewer_ids = qrels_df['reviewer_id'].unique().tolist()
    
    print(f"📊 Target reviewers from qrels: {len(target_reviewer_ids)}")
    print(f"📊 Available reviewers: {len(reviewers_df)}")
    print(f"📊 Papers to score: {len(papers_to_score)}")
    
    # 构建paper_reviewer_map（反向映射：每个论文对应哪些审稿人）
    paper_reviewer_map = {}
    for _, row in qrels_df.iterrows():
        paper_id = row['paper_id']
        reviewer_id = row['reviewer_id']
        if paper_id not in paper_reviewer_map:
            paper_reviewer_map[paper_id] = []
        paper_reviewer_map[paper_id].append(reviewer_id)
    
    print(f"📊 Papers with reviewers: {len(paper_reviewer_map)}")
    
    # 使用BM25Matcher的score方法，但传入paper_reviewer_map来优化
    scores_df, detailed_results = matcher.score(
        papers_to_score, reviewers_df, paper_reviewer_map, True
    )
    
    print(f"✅ BM25 reviewer-centric scoring completed")
    print(f"📊 Scores matrix shape: {scores_df.shape}")
    
    # 转换详细结果为reviewer-centric格式
    reviewer_centric_detailed_results = {}
    
    # 创建反向映射：从paper-centric转为reviewer-centric
    for reviewer_id in target_reviewer_ids:
        # 获取该审稿人标注的论文
        reviewer_papers = qrels_df[qrels_df['reviewer_id'] == reviewer_id]['paper_id'].tolist()
        
        # 收集该审稿人对应论文的分数
        candidates = []
        for paper_id in reviewer_papers:
            if paper_id in scores_df.index and reviewer_id in scores_df.columns:
                score = scores_df.loc[paper_id, reviewer_id]
                candidates.append({
                    "id": paper_id,
                    "score": float(score)
                })
        
        # 按分数降序排序
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        reviewer_centric_detailed_results[reviewer_id] = {
            "query_type": "reviewer",
            "candidates": candidates,
            "total_candidates": len(candidates)
        }
    
    print(f"📋 Generated reviewer-centric detailed results for {len(reviewer_centric_detailed_results)} reviewers")
    
    return scores_df, reviewer_centric_detailed_results

def get_strategy_class_for_task(strategy_class_name: str, task_type: str) -> str:
    """根据任务类型自动选择对应的策略类"""
    if task_type == 'reviewer-centric':
        # 自动映射到reviewer-centric版本
        if strategy_class_name == "ProfileAggregationStrategy":
            return "ReviewerCentricProfileAggregationStrategy"
        elif strategy_class_name == "ScoreAggregationStrategy":
            return "ReviewerCentricScoreAggregationStrategy"

    # paper-centric或未知任务类型，使用原策略
    return strategy_class_name

def generate_paper_centric_scores(strategy, embedding_map, papers_to_score, reviewers_df):
    """生成Paper-Centric的分数矩阵"""
    scores_list = []

    for _, paper_row in papers_to_score.iterrows():
        paper_id = paper_row['paper_id']
        query_embedding = embedding_map.get(paper_id)

        if query_embedding is None:
            continue

        for _, reviewer_row in reviewers_df.iterrows():
            reviewer_id = reviewer_row['reviewer_id']
            authored_ids = reviewer_row.get('authored_paper_ids', [])

            if authored_ids:
                reviewer_embs = [embedding_map.get(pid) for pid in authored_ids if pid in embedding_map]
                if reviewer_embs:
                    score = strategy.calculate_score(query_embedding, reviewer_embs)
                else:
                    score = 0.0
            else:
                score = 0.0

            scores_list.append({
                'paper_id': paper_id,
                'reviewer_id': reviewer_id,
                'score': score
            })

    scores_df_data = pd.DataFrame(scores_list)
    scores_df = scores_df_data.pivot(index='paper_id', columns='reviewer_id', values='score').fillna(0.0)
    return scores_df

def generate_reviewer_centric_scores(strategy, embedding_map, reviewers_df, papers_to_score, qrels_dict):
    """生成Reviewer-Centric的分数矩阵"""
    scores_list = []

    # 获取所有需要评分的论文ID
    all_paper_ids = set(papers_to_score['paper_id'].tolist())
    print(f"📊 Total papers to score: {len(all_paper_ids)}")

    # 检查qrels_dict的结构
    if isinstance(qrels_dict, dict) and 'raw' in qrels_dict:
        # 如果qrels_dict有'raw'键，使用raw数据
        raw_qrels = qrels_dict['raw']
        reviewer_qrels_map = {}
        for _, row in raw_qrels.iterrows():
            reviewer_id = row['reviewer_id']
            paper_id = row['paper_id']
            if reviewer_id not in reviewer_qrels_map:
                reviewer_qrels_map[reviewer_id] = {}
            reviewer_qrels_map[reviewer_id][paper_id] = row.get('relevance', 1)
    else:
        reviewer_qrels_map = qrels_dict

    print(f"📊 Reviewers with qrels: {len(reviewer_qrels_map)}")

    for _, reviewer_row in reviewers_df.iterrows():
        reviewer_id = reviewer_row['reviewer_id']
        authored_ids = reviewer_row.get('authored_paper_ids', [])

        # 获取审稿人的论文embeddings
        if authored_ids:
            reviewer_embs = [embedding_map.get(pid) for pid in authored_ids if pid in embedding_map]
        else:
            reviewer_embs = []

        # 获取该审稿人在qrel中标注的论文（只计算这些论文）
        reviewer_qrels = reviewer_qrels_map.get(reviewer_id, {})
        target_paper_ids = set(reviewer_qrels.keys()) & all_paper_ids

        if target_paper_ids:
            print(f"📝 Reviewer {reviewer_id}: {len(target_paper_ids)} target papers, {len(reviewer_embs)} authored papers")

        for paper_id in target_paper_ids:
            target_paper_embedding = embedding_map.get(paper_id)

            if target_paper_embedding is not None and reviewer_embs:
                # 使用reviewer-centric策略计算分数
                score = strategy.calculate_reviewer_paper_score(reviewer_embs, target_paper_embedding)
            else:
                score = 0.0

            scores_list.append({
                'reviewer_id': reviewer_id,
                'paper_id': paper_id,
                'score': score
            })

    print(f"📊 Generated {len(scores_list)} scores")

    if not scores_list:
        print("⚠️ No scores generated! Creating empty DataFrame...")
        # 对于reviewer-centric任务，创建正确的数据结构：行=审稿人，列=论文
        scores_df = pd.DataFrame(index=reviewers_df['reviewer_id'], columns=list(all_paper_ids)).fillna(0.0)
        return scores_df

    scores_df_data = pd.DataFrame(scores_list)
    # 对于reviewer-centric任务，创建正确的数据结构：行=审稿人，列=论文
    scores_df = scores_df_data.pivot(index='reviewer_id', columns='paper_id', values='score').fillna(0.0)
    return scores_df

def generate_rankings(config: Dict[str, Any], dataset_name: str, papers_df, reviewers_df, 
                     qrels_dict, all_reviewer_ids, metadata) -> tuple:
    """生成排序列表"""
    
    if 'matcher_class' in config:
        matcher_class_name = config['matcher_class']

        if matcher_class_name == "BM25Matcher":
            # BM25方法
            from matchers.keyword.bm25_matcher import BM25Matcher

            print("🔧 Initializing BM25 matcher...")
            matcher = BM25Matcher(config.get('matcher_config', {}))
            method_type = "BM25"

        elif matcher_class_name == "CoFMatcher":
            # CoF方法
            from matchers.cof import CoFMatcher

            print("🔧 Initializing CoF matcher...")
            matcher = CoFMatcher(**config.get('matcher_config', {}))
            method_type = "CoF"

        elif matcher_class_name == "CoFOriginalMatcher":
            # CoF原始实现
            from matchers.cof import CoFOriginalMatcher

            print("🔧 Initializing CoF original matcher...")
            matcher = CoFOriginalMatcher(**config.get('matcher_config', {}))
            method_type = "CoF_Original"

        elif matcher_class_name == "CoFExactMatcher":
            # CoF精确实现（使用预计算embedding）
            from matchers.cof import CoFExactMatcher

            print("🔧 Initializing CoF exact matcher...")
            matcher = CoFExactMatcher(**config.get('matcher_config', {}))
            method_type = "CoF_Exact"

        elif matcher_class_name == "GRUTwoStageMatcher":
            from matchers.gru.gru_matcher import GRUTwoStageMatcher

            print("🔧 Initializing GRU two-stage matcher...")
            matcher = GRUTwoStageMatcher(config.get('matcher_config', {}))
            method_type = "SelfContained"
        elif matcher_class_name == "TPMSTwoStageMatcher":
            from matchers.tpms.tpms_matcher import TPMSTwoStageMatcher

            print("🔧 Initializing TPMS two-stage matcher...")
            matcher = TPMSTwoStageMatcher(config.get('matcher_config', {}))
            method_type = "TPMS"
        else:
            raise ValueError(f"Unknown matcher class: {matcher_class_name}")

        print("📚 Training matcher...")
        # 使用内存监控的训练
        fit_result, offline_time, offline_memory = measure_with_memory(
            matcher.fit, reviewers_df, papers_df, metadata
        )

        print("🎯 Generating ranking lists...")
        # 使用第一个可用的qrel类型来确定需要评分的论文
        available_qrel_types = list(qrels_dict.keys())
        primary_qrel_type = available_qrel_types[0]
        print(f"Using {primary_qrel_type} qrel type to determine papers to score")
        papers_to_score = papers_df[papers_df['paper_id'].isin(qrels_dict[primary_qrel_type]['paper_id'])]

        # 使用内存监控的评分
        if matcher_class_name == "BM25Matcher":
            # 检查任务类型，决定如何处理BM25
            task_type = metadata.get('task_type', 'paper-centric')
            
            if task_type == 'reviewer-centric':
                # reviewer-centric: 需要特殊处理
                print("🔄 Processing BM25 for reviewer-centric task...")
                score_result, online_time, online_memory = measure_with_memory(
                    _bm25_reviewer_centric_score, matcher, papers_to_score, reviewers_df, qrels_dict
                )
                scores_df, detailed_results = score_result
            else:
                # paper-centric: 原有逻辑
                score_result, online_time, online_memory = measure_with_memory(
                    matcher.score, papers_to_score, reviewers_df, metadata, qrels_dict
                )
                scores_df, detailed_results = score_result
        else:
            # CoF和其他自包含匹配器
            task_type = metadata.get('task_type', 'paper-centric')

            if task_type == 'reviewer-centric' and hasattr(matcher, 'predict_reviewer_centric'):
                # 对于支持reviewer-centric的匹配器，使用专门的预测方法
                print("🔄 Processing self-contained matcher for reviewer-centric task...")
                predict_result, online_time, online_memory = measure_with_memory(
                    matcher.predict_reviewer_centric, papers_to_score, reviewers_df, qrels_dict
                )
                detailed_results = predict_result

                # 为了兼容性，也生成scores_df（虽然在reviewer-centric中不是主要输出）
                scores_df = pd.DataFrame(index=papers_to_score['paper_id'], columns=reviewers_df['reviewer_id']).fillna(0.0)

                # 将detailed_results转换为正确的格式
                formatted_detailed_results = {}
                for reviewer_id, candidates in detailed_results.items():
                    formatted_detailed_results[reviewer_id] = {
                        "query_type": "reviewer",
                        "candidates": candidates,
                        "total_candidates": len(candidates)
                    }
                detailed_results = formatted_detailed_results

            else:
                # paper-centric或不支持reviewer-centric的匹配器
                score_result, online_time, online_memory = measure_with_memory(
                    matcher.score, papers_to_score, reviewers_df, metadata, qrels_dict
                )
                scores_df = score_result

                # 生成详细结果
                print("📋 Generating detailed ranking lists...")
                detailed_handler = get_detailed_results_handler()
                detailed_results = detailed_handler.generate_detailed_results(scores_df, metadata)

        # 记录内存使用情况
        peak_memory = max(offline_memory, online_memory)

    elif 'encoder_class' in config:
        # Embedding方法
        encoder_class_name = config['encoder_class']
        strategy_class_name = config['strategy_class']

        # 根据任务类型自动映射策略类
        task_type = metadata.get('task_type', 'paper-centric')
        actual_strategy_class_name = get_strategy_class_for_task(strategy_class_name, task_type)

        print(f"📋 Task type: {task_type}")
        print(f"📋 Original strategy: {strategy_class_name}")
        print(f"📋 Actual strategy: {actual_strategy_class_name}")

        # 动态导入
        if encoder_class_name == "SentenceTransformerEncoder":
            from matchers.embedding.encoders import SentenceTransformerEncoder
            encoder_class = SentenceTransformerEncoder
        elif encoder_class_name == "SPECTER2Encoder":
            from matchers.embedding.encoders import SPECTER2Encoder
            encoder_class = SPECTER2Encoder
        elif encoder_class_name == "COCO_DREncoder":
            from matchers.embedding.encoders import COCO_DREncoder
            encoder_class = COCO_DREncoder
        elif encoder_class_name == "SciBERTEncoder":
            from matchers.embedding.encoders import SciBERTEncoder
            encoder_class = SciBERTEncoder
        else:
            raise ValueError(f"Unknown encoder class: {encoder_class_name}")

        if actual_strategy_class_name == "ProfileAggregationStrategy":
            from matchers.embedding.strategies import ProfileAggregationStrategy
            strategy_class = ProfileAggregationStrategy
        elif actual_strategy_class_name == "ScoreAggregationStrategy":
            from matchers.embedding.strategies import ScoreAggregationStrategy
            strategy_class = ScoreAggregationStrategy
        elif actual_strategy_class_name == "ReviewerCentricProfileAggregationStrategy":
            from matchers.embedding.strategies import ReviewerCentricProfileAggregationStrategy
            strategy_class = ReviewerCentricProfileAggregationStrategy
        elif actual_strategy_class_name == "ReviewerCentricScoreAggregationStrategy":
            from matchers.embedding.strategies import ReviewerCentricScoreAggregationStrategy
            strategy_class = ReviewerCentricScoreAggregationStrategy
        else:
            raise ValueError(f"Unknown strategy class: {actual_strategy_class_name}")
        
        # 获取需要评分的论文（使用第一个可用的qrel类型）
        first_qrel_type = list(qrels_dict.keys())[0]
        papers_to_score = papers_df[papers_df['paper_id'].isin(qrels_dict[first_qrel_type]['paper_id'])]
        print(f"📊 Using qrel type '{first_qrel_type}' to determine papers to score: {len(papers_to_score)} papers")
        
        print("📚 Offline phase: Checking cache first...")

        # 先检查缓存，避免不必要的模型加载
        from utils.embedding_cache import get_embedding_cache
        cache = get_embedding_cache()
        all_paper_ids = papers_df['paper_id'].tolist()
        
        # 从配置中获取模型名称，对SPECTER2需要特殊处理
        model_name = config['encoder_config'].get('model_name', 'unknown')

        # 对SPECTER2，需要根据adapter构建正确的缓存键
        if (config['encoder_class'] == 'SPECTER2Encoder' or
            (config['encoder_class'] == 'SentenceTransformerEncoder' and 'specter2' in model_name.lower())):

            if config['encoder_class'] == 'SPECTER2Encoder':
                # SPECTER2Encoder with adapter
                adapter_path = config['encoder_config'].get('adapter_name', '')
                if adapter_path:
                    # 从adapter路径中提取adapter名称 (e.g., "models/specter2/adapters/adhoc" -> "adhoc")
                    adapter_name = adapter_path.split('/')[-1]
                    # 构建与offline_preprocessing.py一致的模型名称
                    cache_model_name = f"specter2{adapter_name}"
                else:
                    cache_model_name = "specter2base"  # 默认base
            else:
                # SentenceTransformerEncoder for SPECTER2 Base
                cache_model_name = "specter2base"

            print(f"🔍 Checking cache for SPECTER2 model: {cache_model_name}")
        else:
            cache_model_name = model_name
            print(f"🔍 Checking cache for model: {cache_model_name}")

        cached_embeddings = cache.load_embeddings_by_ids(cache_model_name, all_paper_ids)

        if cached_embeddings is not None:
            print(f"✅ Cache hit! Using cached embeddings for {len(all_paper_ids)} papers")
            embedding_map = cached_embeddings
            offline_time = 0.0  # 缓存加载时间忽略不计
            offline_memory = get_memory_usage_mb()
            is_cached = True
            
            # 缓存命中时，只初始化strategy，不需要encoder
            print("🔧 Initializing strategy (encoder not needed due to cache hit)...")
            strategy = strategy_class(**config['strategy_config'])
            encoder = None  # 不需要encoder
        else:
            # 严格模式：缓存未命中则拒绝运行，要求先进行离线预计算
            print("❌ Cache miss! Embeddings must be precomputed before running experiments.")
            raise RuntimeError(
                f"❌ Embeddings not found in cache for model '{model_name}' with {len(all_paper_ids)} papers.\n"
                f"💡 Please precompute embeddings first using:\n"
                f"   python offline_preprocessing.py --dataset {dataset_name} --embedding_models specter\n"
                f"   或者指定正确的模型名称（与缓存保持完全一致）。"
            )


        
        print("🎯 Online phase: Computing matching scores...")

        # 计算匹配分数 - 根据任务类型选择不同逻辑
        task_type = metadata.get('task_type', 'paper-centric')

        def compute_scores():
            if task_type == 'reviewer-centric':
                return generate_reviewer_centric_scores(strategy, embedding_map, reviewers_df, papers_to_score, qrels_dict)
            else:
                return generate_paper_centric_scores(strategy, embedding_map, papers_to_score, reviewers_df)

        # 使用内存监控的分数计算
        scores_df, online_time, online_memory = measure_with_memory(compute_scores)
        
        # 生成详细结果
        print("📋 Generating detailed ranking lists...")
        detailed_handler = get_detailed_results_handler()
        detailed_results = detailed_handler.generate_detailed_results(scores_df, metadata)
        
        method_type = "Embedding"

        # 计算embedding方法的峰值内存
        peak_memory = max(offline_memory, online_memory)

    else:
        raise ValueError("Unknown method type in config")
    
    # 计算正确的效率指标
    task_type = metadata.get('task_type', 'paper-centric')
    correct_latency = calculate_correct_latency(online_time, papers_to_score, reviewers_df, task_type)

    # 计算峰值内存使用量
    if method_type == "BM25":
        peak_memory_mb = max(offline_memory, online_memory)
    else:  # Embedding
        peak_memory_mb = max(offline_memory, online_memory)

    efficiency_metrics = {
        "offline_time_seconds": offline_time,
        "online_latency_ms_per_query": correct_latency,
        "total_online_time_seconds": online_time,
        "total_experiment_time_seconds": offline_time + online_time,
        "memory_usage_mb": peak_memory_mb,
        "method_type": method_type,
        # 额外的调试信息
        "debug_info": {
            "papers_to_score": len(papers_to_score),
            "total_reviewers": len(reviewers_df),
            "task_type": task_type,
            "total_queries_computed": len(papers_to_score) * len(reviewers_df) if task_type == 'paper-centric' else len(reviewers_df) * len(papers_to_score),
            "is_cached": is_cached if method_type == "Embedding" else False,
            "offline_memory_mb": offline_memory,
            "online_memory_mb": online_memory if method_type == "BM25" else online_memory
        }
    }
    
    return detailed_results, efficiency_metrics, method_type

def evaluate_rankings(ranking_file: str, dataset_name: str) -> Dict[str, Dict]:
    """从排序列表评估所有qrel_type的指标"""
    
    # 加载排序列表
    with open(ranking_file, 'r', encoding='utf-8') as f:
        ranking_data = json.load(f)
    
    # 加载数据集
    loader = DataLoader()
    papers_df, reviewers_df, qrels_dict, all_reviewer_ids, metadata = loader.load_dataset(dataset_name)
    
    # 重构分数矩阵
    detailed_handler = get_detailed_results_handler()
    ranking_lists = ranking_data["ranking_lists"]
    
    task_type = metadata.get('task_type', 'paper-centric')
    if task_type == 'reviewer-centric':
        all_query_ids = list(ranking_lists.keys())
        all_candidate_ids = papers_df['paper_id'].tolist()
    else:
        all_query_ids = list(ranking_lists.keys())
        all_candidate_ids = all_reviewer_ids
    
    scores_df = detailed_handler.extract_scores_matrix(
        ranking_lists, all_query_ids, all_candidate_ids
    )
    
    # 评估所有qrel_type
    all_performance_metrics = {}
    for qrel_type, qrels_df in qrels_dict.items():
        evaluator = MetricsCalculator(qrels_df, all_reviewer_ids, metadata, qrel_type)
        metrics = evaluator.calculate_all(scores_df)
        all_performance_metrics[qrel_type] = metrics
    
    return all_performance_metrics

def get_output_path(mode: str, dataset_name: str, method_type: str, config_name: str) -> str:
    """根据results目录的文件组织规则生成输出路径"""
    dataset_lower = dataset_name.lower()

    # 特殊处理CoF原始实现，保存到cof目录下
    if config_name == "cof_original":
        if mode == "generate":
            return f"results/cof/{dataset_name}/{config_name}/{dataset_lower}_{config_name}_ranking.json"
        elif mode == "evaluate":
            return f"results/cof/{dataset_name}/{config_name}/{dataset_lower}_{config_name}_evaluation.json"
    else:
        if mode == "generate":
            return f"results/{method_type.lower()}/{dataset_name}/{config_name}/{dataset_lower}_{config_name}_ranking.json"
        elif mode == "evaluate":
            return f"results/{method_type.lower()}/{dataset_name}/{config_name}/{dataset_lower}_{config_name}_evaluation.json"

    if mode not in ["generate", "evaluate"]:
        raise ValueError(f"Unsupported mode: {mode}")

def infer_method_type(config: Dict[str, Any]) -> str:
    """从配置文件自动推断方法类型"""
    if 'matcher_class' in config:
        matcher_class = config['matcher_class']
        if matcher_class == "BM25Matcher":
            return "BM25"
        elif matcher_class == "CoFMatcher":
            return "CoF"
        elif matcher_class == "CoFOriginalMatcher":
            return "CoF_Original"
        elif matcher_class == "CoFExactMatcher":
            return "CoF_Exact"
        elif matcher_class == "TPMSTwoStageMatcher":
            return "TPMS"
        else:
            return "SelfContained"  # 其他自包含匹配器
    elif 'encoder_class' in config:
        return "Embedding"
    else:
        raise ValueError("Cannot infer method_type from config")

def display_smart_results(all_performance_metrics: Dict[str, Dict]):
    """智能显示所有可用的qrel类型结果"""
    print(f"\n🎯 Evaluation Results:")
    print("="*50)
    
    # 按优先级排序显示
    priority_order = ['raw', 'soft', 'hard']
    available_types = list(all_performance_metrics.keys())
    
    # 按优先级排序
    sorted_types = sorted(available_types, 
                         key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))
    
    for qrel_type in sorted_types:
        metrics = all_performance_metrics[qrel_type]
        print(f"\n📊 {qrel_type.upper()} Qrel Type:")
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
    
    # 显示可用类型信息
    print(f"\n💡 Available qrel types: {', '.join(sorted_types)}")


def save_results(output_path: str, mode: str, **kwargs):
    """保存结果"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if mode == "generate":
        # 保存排序列表
        result = {
            "ranking_format_version": "1.0",
            "experiment_config": kwargs["experiment_config"],
            "data_info": kwargs["data_info"],
            "efficiency_metrics": kwargs["efficiency_metrics"],
            "ranking_lists": kwargs["detailed_results"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    elif mode == "evaluate":
        # 保存评估结果
        result = {
            "evaluation_format_version": "1.0",
            "source_ranking_file": kwargs["source_file"],
            "experiment_config": kwargs["experiment_config"],
            "data_info": kwargs["data_info"],
            "efficiency_metrics": kwargs["efficiency_metrics"],
            "performance_metrics": kwargs["all_performance_metrics"].get("raw", {}),
            "all_performance_metrics": kwargs["all_performance_metrics"],
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generation_timestamp": kwargs.get("generation_timestamp", "unknown")
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="统一实验执行脚本")
    parser.add_argument("--dataset_name", required=True, help="数据集名称")
    parser.add_argument("--config_file", help="配置文件路径（generate模式需要）")
    parser.add_argument("--ranking_file", help="排序文件路径（evaluate模式需要）")
    parser.add_argument("--mode", choices=["generate", "evaluate"], required=True,
                       help="运行模式")
    parser.add_argument("--output_path", help="输出文件路径（可选）")
    # 移除qrel_type参数，系统自动检测和计算所有可用的qrel类型
    
    args = parser.parse_args()
    
    try:
        print(f"🚀 Running experiment: {args.dataset_name} ({args.mode} mode)")
        print("="*80)

        if args.mode == "generate":
            if not args.config_file:
                raise ValueError("generate模式需要--config_file参数")

            # 加载配置和数据集
            config = load_config(args.config_file)
            method_type = infer_method_type(config)
            config_name = Path(args.config_file).stem

            loader = DataLoader()
            papers_df, reviewers_df, qrels_dict, all_reviewer_ids, metadata = loader.load_dataset(args.dataset_name)

            # 生成排序列表
            detailed_results, efficiency_metrics, _ = generate_rankings(
                config, args.dataset_name, papers_df, reviewers_df, qrels_dict, all_reviewer_ids, metadata
            )

            # 构建配置信息
            experiment_config = {
                "config_file": args.config_file,
                "dataset": args.dataset_name,
                "experiment_name": config.get('experiment_name', 'Unnamed'),
                "method_type": method_type,
                "config": config
                # 移除qrel_type，系统自动检测
            }

            data_info = {
                "total_papers": len(papers_df),
                "total_reviewers": len(reviewers_df),
                "papers_scored": len(detailed_results),
                "task_type": metadata.get('task_type', 'paper-centric'),
                "qrel_format": metadata.get('qrel_format', 'closed')
            }

            # 确定输出路径
            if args.output_path:
                output_path = args.output_path
            else:
                output_path = get_output_path(args.mode, args.dataset_name, method_type, config_name)

            # 保存结果
            save_results(output_path, args.mode,
                        experiment_config=experiment_config,
                        data_info=data_info,
                        efficiency_metrics=efficiency_metrics,
                        detailed_results=detailed_results)

            # 显示效率信息
            print(f"\n⚡ Efficiency:")
            print(f"Offline time: {efficiency_metrics['offline_time_seconds']:.2f}s")
            print(f"Online latency: {efficiency_metrics['online_latency_ms_per_query']:.1f}ms/query")

        elif args.mode == "evaluate":
            if not args.ranking_file:
                raise ValueError("evaluate模式需要--ranking_file参数")

            # 从排序文件加载信息
            with open(args.ranking_file, 'r') as f:
                ranking_data = json.load(f)
            experiment_config = ranking_data["experiment_config"]
            data_info = ranking_data["data_info"]
            efficiency_metrics = ranking_data["efficiency_metrics"]
            method_type = experiment_config["method_type"]
            config_name = Path(experiment_config["config_file"]).stem

            # 评估排序列表
            all_performance_metrics = evaluate_rankings(args.ranking_file, args.dataset_name)

            # 确定输出路径
            if args.output_path:
                output_path = args.output_path
            else:
                output_path = get_output_path(args.mode, args.dataset_name, method_type, config_name)

            # 保存结果
            save_results(output_path, args.mode,
                        experiment_config=experiment_config,
                        data_info=data_info,
                        efficiency_metrics=efficiency_metrics,
                        all_performance_metrics=all_performance_metrics,
                        source_file=args.ranking_file,
                        generation_timestamp=ranking_data.get("timestamp", "unknown"))

            # 智能显示所有可用的qrel类型结果
            display_smart_results(all_performance_metrics)

        
        print(f"\n✅ Experiment completed successfully!")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
