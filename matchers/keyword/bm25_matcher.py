#!/usr/bin/env python3
"""
matchers/keyword/bm25_matcher.py - BM25算法的自包含匹配器实现
"""

import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Dict, List
from ..base import BaseMatcher # 从上级目录的base.py导入

class BM25Matcher(BaseMatcher):
    """
    使用BM25算法的自包含匹配器。
    它在fit阶段构建审稿人简介的索引，在score阶段进行查询。
    """

    def __init__(self, config: Dict):
        """
        初始化BM25Matcher。
        :param config: 配置字典，可以包含 'bm25_k1' 和 'bm25_b'。
        """
        super().__init__(config)
        self.bm25_k1 = self.config.get('bm25_k1', 1.5)
        self.bm25_b = self.config.get('bm25_b', 0.75)

        self.bm25_index = None
        self.reviewer_ids: List[str] = []

        # 支持预计算缓存
        self.use_precomputed = self.config.get('use_precomputed', False)
        self.cache_dir = self.config.get('cache_dir', 'cache')

    def _try_load_precomputed_index(self, reviewers_df: pd.DataFrame) -> bool:
        """尝试加载预计算的BM25索引"""
        if not self.use_precomputed:
            return False

        try:
            import pickle
            import hashlib
            from pathlib import Path

            # 生成缓存键
            profiles = reviewers_df['profile'].fillna('').tolist()
            text_hash = hashlib.md5(str(sorted(profiles)).encode()).hexdigest()
            cache_key = f"dataset_{text_hash}_k1{self.bm25_k1}_b{self.bm25_b}"
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()

            cache_file = Path(self.cache_dir) / "bm25" / f"{cache_key}.pkl"

            if cache_file.exists():
                print("Loading precomputed BM25 index...")
                with open(cache_file, 'rb') as f:
                    cached_result = pickle.load(f)

                self.bm25_index = cached_result['bm25_index']
                self.reviewer_ids = cached_result['reviewer_ids']
                self.is_fitted = True

                print(f"Loaded precomputed BM25 index for {len(self.reviewer_ids)} reviewers")
                return True
        except Exception as e:
            print(f"Failed to load precomputed index: {e}")

        return False

    def fit(self, reviewers_df: pd.DataFrame, papers_df: pd.DataFrame, metadata: Dict = None):
        """
        根据审稿人的profile文本构建BM25索引。

        :param reviewers_df: 审稿人信息的DataFrame，必须包含 'reviewer_id' 和 'profile' 列。
        :param papers_df: 在此模型中未使用，但为遵守接口而保留。
        :param metadata: 元数据信息（在此模型中未使用，但为遵守接口而保留）。
        """
        # 尝试加载预计算的索引
        if self._try_load_precomputed_index(reviewers_df):
            return

        print("Fitting BM25 index...")
        if 'profile' not in reviewers_df.columns:
            raise ValueError("reviewers_df must contain a 'profile' column.")

        # 1. 提取语料库并进行简单分词
        corpus = reviewers_df['profile'].fillna('').tolist()
        tokenized_corpus = []
        empty_profiles = 0

        for doc in corpus:
            tokens = doc.split()
            # 确保每个文档至少有一个token，避免BM25的除零错误
            if not tokens:
                tokens = ['empty']  # 为空文档添加占位符token
                empty_profiles += 1
            tokenized_corpus.append(tokens)



        # 2. 创建并存储BM25索引
        self.bm25_index = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)

        # 3. 严格按顺序存储reviewer_id，用于后续分数映射
        self.reviewer_ids = reviewers_df['reviewer_id'].tolist()
        self.is_fitted = True
        print(f"BM25 index fitted for {len(self.reviewer_ids)} reviewers.")

    def score(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame,
              paper_reviewer_map: Dict[str, List[str]] = None, return_detailed: bool = False) -> pd.DataFrame:
        """
        使用已构建的索引，为输入论文计算与审稿人的匹配分数。

        :param papers_df: 需要评分的论文DataFrame，必须包含 'paper_id', 'title' 和 'abstract' 列。
        :param reviewers_df: 候选审稿人的DataFrame（在此模型中主要用于获取完整的审稿人列表）。
        :param paper_reviewer_map: 可选的论文到审稿人ID列表的映射，用于只计算相关审稿人的分数
        :param return_detailed: 是否返回详细结果（用于两阶段评估）
        :return: 一个DataFrame，index为paper_id，columns为reviewer_id，值为BM25分数。
        """
        if self.bm25_index is None:
            raise RuntimeError("BM25Matcher has not been fitted. Please call .fit() first.")

        print(f"Scoring {len(papers_df)} papers with BM25...")

        # 创建审稿人ID到索引的映射
        reviewer_id_to_idx = {rid: idx for idx, rid in enumerate(self.reviewer_ids)}

        all_scores = {}
        detailed_results = {} if return_detailed else None

        for _, paper_row in papers_df.iterrows():
            # 使用标题+摘要作为查询文本
            title = paper_row.get('title', '')
            abstract = paper_row.get('abstract', '')

            # 确保都是字符串类型
            if not isinstance(title, str):
                title = ""
            if not isinstance(abstract, str):
                abstract = ""

            # 拼接标题和摘要
            query_parts = []
            if title.strip():
                query_parts.append(title.strip())
            if abstract.strip():
                query_parts.append(abstract.strip())

            query_text = ' '.join(query_parts)

            # 对查询进行同样的分词处理
            tokenized_query = query_text.split()
            if not tokenized_query:
                tokenized_query = ['empty']  # 为空查询添加占位符token
            
            paper_id = paper_row['paper_id']

            # 计算BM25分数
            doc_scores = self.bm25_index.get_scores(tokenized_query)

            # 根据是否有映射决定存储策略
            if paper_reviewer_map and paper_id in paper_reviewer_map:
                # 优化模式：只存储该论文标注的审稿人分数
                target_reviewers = paper_reviewer_map[paper_id]
                paper_scores = {}
                for reviewer_id in target_reviewers:
                    if reviewer_id in reviewer_id_to_idx:
                        idx = reviewer_id_to_idx[reviewer_id]
                        paper_scores[reviewer_id] = doc_scores[idx]
                    else:
                        paper_scores[reviewer_id] = 0.0
                all_scores[paper_id] = paper_scores

                # 收集详细结果（如果需要）
                if return_detailed:
                    # 为详细结果创建完整的候选列表
                    candidates = []
                    for reviewer_id in target_reviewers:
                        if reviewer_id in reviewer_id_to_idx:
                            idx = reviewer_id_to_idx[reviewer_id]
                            score = doc_scores[idx]
                        else:
                            score = 0.0
                        candidates.append({"id": reviewer_id, "score": float(score)})

                    # 按分数降序排序
                    candidates.sort(key=lambda x: x["score"], reverse=True)

                    detailed_results[paper_id] = {
                        "query_type": "paper",
                        "candidates": candidates,
                        "total_candidates": len(candidates)
                    }
            else:
                # 完整模式：计算与所有审稿人的分数
                all_scores[paper_id] = doc_scores

                # 收集详细结果（如果需要）
                if return_detailed:
                    candidates = []
                    for i, reviewer_id in enumerate(self.reviewer_ids):
                        candidates.append({"id": reviewer_id, "score": float(doc_scores[i])})

                    # 按分数降序排序
                    candidates.sort(key=lambda x: x["score"], reverse=True)

                    detailed_results[paper_id] = {
                        "query_type": "paper",
                        "candidates": candidates,
                        "total_candidates": len(candidates)
                    }

        # 将结果转换为DataFrame
        if paper_reviewer_map:
            # 稀疏模式：从字典构建DataFrame
            scores_df = pd.DataFrame.from_dict(all_scores, orient='index')
            # 确保包含所有需要的审稿人列
            all_target_reviewers = set()
            for paper_id in all_scores:
                if paper_id in paper_reviewer_map:
                    all_target_reviewers.update(paper_reviewer_map[paper_id])
            scores_df = scores_df.reindex(columns=sorted(all_target_reviewers), fill_value=0.0)
        else:
            # 完整模式：使用所有审稿人
            scores_df = pd.DataFrame.from_dict(all_scores, orient='index', columns=self.reviewer_ids)
            scores_df = scores_df.reindex(columns=reviewers_df['reviewer_id'], fill_value=0.0)

        # 返回结果
        if return_detailed:
            return scores_df, detailed_results
        else:
            return scores_df

    def score_streaming(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame,
                       evaluator, paper_reviewer_map: Dict[str, List[str]] = None) -> Dict[str, float]:
        """
        流式评分：逐个论文计算分数并立即评估，避免内存爆炸

        :param papers_df: 需要评分的论文DataFrame
        :param reviewers_df: 审稿人DataFrame
        :param evaluator: 流式评估器实例
        :param paper_reviewer_map: 论文到审稿人的映射（用于优化）
        :return: 评估结果字典
        """
        if self.bm25_index is None:
            raise RuntimeError("BM25Matcher has not been fitted. Please call .fit() first.")

        print(f"Streaming evaluation: {len(papers_df)} papers")
        evaluator.reset_accumulator()

        # 创建审稿人ID到索引的映射（必须与训练时的索引顺序一致）
        reviewer_id_to_idx = {rid: idx for idx, rid in enumerate(self.reviewer_ids)}

        processed_papers = 0
        for _, paper_row in papers_df.iterrows():
            paper_id = paper_row['paper_id']

            # 使用标题+摘要作为查询文本
            title = paper_row.get('title', '')
            abstract = paper_row.get('abstract', '')

            # 确保都是字符串类型
            if not isinstance(title, str):
                title = ""
            if not isinstance(abstract, str):
                abstract = ""

            # 拼接标题和摘要
            query_parts = []
            if title.strip():
                query_parts.append(title.strip())
            if abstract.strip():
                query_parts.append(abstract.strip())

            query_text = ' '.join(query_parts)

            # 分词处理
            tokenized_query = query_text.split()
            if not tokenized_query:
                tokenized_query = ['empty']

            # 确定需要计算分数的审稿人
            if paper_reviewer_map and paper_id in paper_reviewer_map:
                target_reviewers = paper_reviewer_map[paper_id]
            else:
                target_reviewers = reviewers_df['reviewer_id'].tolist()

            # 计算分数
            if paper_reviewer_map and paper_id in paper_reviewer_map:
                # 优化模式：只计算相关审稿人的分数
                predicted_scores = {}
                all_scores = self.bm25_index.get_scores(tokenized_query)
                for reviewer_id in target_reviewers:
                    if reviewer_id in reviewer_id_to_idx:
                        idx = reviewer_id_to_idx[reviewer_id]
                        predicted_scores[reviewer_id] = all_scores[idx]
            else:
                # 完整模式：计算所有审稿人的分数
                all_scores = self.bm25_index.get_scores(tokenized_query)
                predicted_scores = {
                    reviewer_id: all_scores[idx]
                    for reviewer_id, idx in reviewer_id_to_idx.items()
                }

            # 立即评估这篇论文
            evaluator.evaluate_single_query(paper_id, predicted_scores)

            processed_papers += 1
            if processed_papers % 100 == 0:
                progress = evaluator.get_progress_info()
                print(f"  Processed {processed_papers}/{len(papers_df)} papers "
                      f"({progress['progress_percentage']:.1f}%)")

        return evaluator.get_final_results()

# --- 使用示例与单元测试 ---
if __name__ == '__main__':
    # 1. 模拟DataLoader的输出
    papers_data = {
        'paper_id': ['p1', 'p2'],
        'title': ['Paper A', 'Paper B'],
        'abstract': ['deep learning for nlp', 'graph neural networks']
    }
    reviewers_data = {
        'reviewer_id': ['r1', 'r2', 'r3'],
        'authored_paper_ids': [['...'], ['...'], ['...']],
        'profile': [
            'graph attention networks are powerful', 
            'nlp and deep learning have synergy',
            'graph theory is a field of mathematics'
        ]
    }
    papers_df = pd.DataFrame(papers_data)
    reviewers_df = pd.DataFrame(reviewers_data)
    
    # 2. 初始化并使用BM25Matcher
    bm25_config = {'bm25_k1': 1.2, 'bm25_b': 0.75}
    bm25_matcher = BM25Matcher(config=bm25_config)
    
    # 3. 训练/准备模型
    bm25_matcher.fit(reviewers_df, papers_df)
    
    # 4. 评分
    # 只对p1进行评分
    scores = bm25_matcher.score(papers_df.head(1), reviewers_df)
    
    # 5. 打印结果
    print("\n--- BM25 Score Output ---")
    print(scores)
    # 预期输出一个 1x3 的DataFrame，index为'p1'，columns为'r1', 'r2', 'r3'
    # 'p1'的查询是'deep learning for nlp'，应该与r2的profile 'nlp and deep learning'得分最高