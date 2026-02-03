#!/usr/bin/env python3
"""
utils/detailed_results.py - 详细结果处理模块

提供详细的匹配结果保存和加载功能，包括：
- 每个查询的完整排序列表
- 候选项的详细分数
- 支持后续的细粒度分析
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path


class DetailedResultsHandler:
    """详细结果处理器"""
    
    def __init__(self):
        pass
    
    def generate_detailed_results(self, scores_df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        从分数矩阵生成详细结果
        
        Args:
            scores_df: 分数矩阵 (行=查询, 列=候选)
            metadata: 数据集元数据
            
        Returns:
            详细结果字典
        """
        task_type = metadata.get('task_type', 'paper-centric')
        detailed_results = {}
        
        if task_type == 'reviewer-centric':
            # reviewer-centric: 查询是审稿人，候选是论文
            # scores_df结构：行=审稿人，列=论文，值=分数
            # 每个审稿人（行）对应其论文排序列表
            for reviewer_id in scores_df.index:
                reviewer_scores = scores_df.loc[reviewer_id]
                # 只保留非零分数的论文（有实际匹配的论文）
                non_zero_scores = reviewer_scores[reviewer_scores > 0]
                # 按分数降序排序
                sorted_papers = non_zero_scores.sort_values(ascending=False)

                candidates = []
                for paper_id, score in sorted_papers.items():
                    candidates.append({
                        "id": paper_id,
                        "score": float(score)
                    })

                # 如果没有非零分数，至少包含所有论文（分数为0）
                if len(candidates) == 0:
                    for paper_id in scores_df.columns:
                        candidates.append({
                            "id": paper_id,
                            "score": 0.0
                        })

                detailed_results[reviewer_id] = {
                    "query_type": "reviewer",
                    "candidates": candidates,
                    "total_candidates": len(candidates)
                }
        else:
            # paper-centric: 行是论文，列是审稿人，查询是论文
            for paper_id in scores_df.index:
                paper_scores = scores_df.loc[paper_id]
                # 按分数降序排序
                sorted_reviewers = paper_scores.sort_values(ascending=False)
                
                candidates = []
                for reviewer_id, score in sorted_reviewers.items():
                    candidates.append({
                        "id": reviewer_id,
                        "score": float(score)
                    })
                
                detailed_results[paper_id] = {
                    "query_type": "paper",
                    "candidates": candidates,
                    "total_candidates": len(candidates)
                }
        
        return detailed_results
    
    def save_detailed_results(self, output_path: str, experiment_config: Dict[str, Any],
                            detailed_results: Dict[str, Any], summary_metrics: Dict[str, Any],
                            efficiency_metrics: Dict[str, Any], data_info: Dict[str, Any]) -> None:
        """
        保存详细结果到文件
        
        Args:
            output_path: 输出文件路径
            experiment_config: 实验配置
            detailed_results: 详细结果
            summary_metrics: 汇总指标
            efficiency_metrics: 效率指标
            data_info: 数据信息
        """
        # 构建完整结果
        full_results = {
            "experiment_config": experiment_config,
            "detailed_results": detailed_results,
            "summary_metrics": summary_metrics,
            "efficiency_metrics": efficiency_metrics,
            "data_info": data_info,
            "result_format_version": "2.0"  # 标记新格式
        }
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to: {output_path}")
    
    def load_detailed_results(self, result_path: str) -> Dict[str, Any]:
        """
        从文件加载详细结果
        
        Args:
            result_path: 结果文件路径
            
        Returns:
            详细结果字典
        """
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 检查格式版本
        if results.get("result_format_version") != "2.0":
            raise ValueError(f"Unsupported result format. Expected version 2.0, got {results.get('result_format_version', '1.0')}")
        
        return results
    
    def extract_scores_matrix(self, detailed_results: Dict[str, Any], 
                            all_query_ids: List[str], all_candidate_ids: List[str]) -> pd.DataFrame:
        """
        从详细结果中提取分数矩阵
        
        Args:
            detailed_results: 详细结果
            all_query_ids: 所有查询ID列表
            all_candidate_ids: 所有候选ID列表
            
        Returns:
            分数矩阵DataFrame
        """
        # 初始化分数矩阵
        scores_matrix = np.zeros((len(all_query_ids), len(all_candidate_ids)))
        
        for i, query_id in enumerate(all_query_ids):
            if query_id in detailed_results:
                query_result = detailed_results[query_id]
                for candidate_info in query_result["candidates"]:
                    candidate_id = candidate_info["id"]
                    score = candidate_info["score"]
                    
                    if candidate_id in all_candidate_ids:
                        j = all_candidate_ids.index(candidate_id)
                        scores_matrix[i, j] = score
        
        # 构建DataFrame
        scores_df = pd.DataFrame(
            scores_matrix,
            index=all_query_ids,
            columns=all_candidate_ids
        )
        
        return scores_df
    
    def get_top_k_candidates(self, detailed_results: Dict[str, Any], 
                           query_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        获取指定查询的前K个候选项
        
        Args:
            detailed_results: 详细结果
            query_id: 查询ID
            k: 返回前K个
            
        Returns:
            前K个候选项列表
        """
        if query_id not in detailed_results:
            return []
        
        candidates = detailed_results[query_id]["candidates"]
        return candidates[:k]
    
    def analyze_score_distribution(self, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析分数分布统计
        
        Args:
            detailed_results: 详细结果
            
        Returns:
            分数分布统计
        """
        all_scores = []
        query_stats = {}
        
        for query_id, query_result in detailed_results.items():
            scores = [c["score"] for c in query_result["candidates"]]
            all_scores.extend(scores)
            
            query_stats[query_id] = {
                "min_score": min(scores),
                "max_score": max(scores),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "num_candidates": len(scores)
            }
        
        overall_stats = {
            "overall": {
                "min_score": min(all_scores),
                "max_score": max(all_scores),
                "mean_score": np.mean(all_scores),
                "std_score": np.std(all_scores),
                "total_scores": len(all_scores)
            },
            "per_query": query_stats
        }
        
        return overall_stats


# 全局实例
_detailed_handler = None

def get_detailed_results_handler() -> DetailedResultsHandler:
    """获取全局详细结果处理器实例"""
    global _detailed_handler
    if _detailed_handler is None:
        _detailed_handler = DetailedResultsHandler()
    return _detailed_handler
