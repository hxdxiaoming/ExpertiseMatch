#!/usr/bin/env python3
"""
matchers/keyword/reviewer_centric_bm25_matcher.py - 支持reviewer-centric任务的BM25匹配器

这个匹配器专门为reviewer-centric任务设计，与传统的BM25Matcher相反：
- fit阶段：为论文构建BM25索引（基于论文title+abstract拼接）
- score阶段：用审稿人profile作为查询，在论文索引中搜索

用于stelmakh等reviewer-centric数据集。
"""

import pandas as pd
from rank_bm25 import BM25Okapi
from typing import Dict, List
from ..base import BaseMatcher


class ReviewerCentricBM25Matcher(BaseMatcher):
    """
    支持reviewer-centric任务的BM25匹配器。
    与传统BM25Matcher相反：为论文建索引（基于title+abstract），用审稿人查询。
    """

    def __init__(self, config: Dict):
        """
        初始化ReviewerCentricBM25Matcher。
        :param config: 配置字典，可以包含 'bm25_k1' 和 'bm25_b'。
        """
        super().__init__(config)
        self.bm25_k1 = self.config.get('bm25_k1', 1.5)
        self.bm25_b = self.config.get('bm25_b', 0.75)
        
        self.bm25_index = None
        self.paper_ids: List[str] = []

    def fit(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame):
        """
        根据论文的title+abstract拼接构建BM25索引。

        :param papers_df: 论文信息的DataFrame，必须包含 'paper_id', 'title' 和 'abstract' 列。
        :param reviewers_df: 在此模型中未使用，但为遵守接口而保留。
        """
        print("Fitting BM25 index for papers using title+abstract (reviewer-centric mode)...")
        required_columns = ['paper_id', 'title', 'abstract']
        missing_columns = [col for col in required_columns if col not in papers_df.columns]
        if missing_columns:
            raise ValueError(f"papers_df must contain columns: {missing_columns}")

        # 1. 构建title+abstract拼接的语料库
        corpus = []
        for _, row in papers_df.iterrows():
            title = row['title'] if pd.notna(row['title']) else ''
            abstract = row['abstract'] if pd.notna(row['abstract']) else ''

            # 拼接title和abstract，提供更丰富的信息
            text_parts = []
            if title.strip():
                text_parts.append(title.strip())
            if abstract.strip():
                text_parts.append(abstract.strip())

            combined_text = ' '.join(text_parts)
            corpus.append(combined_text)

        # 2. 分词处理
        tokenized_corpus = []
        for doc in corpus:
            tokens = doc.split()
            # 确保每个文档至少有一个token，避免BM25的除零错误
            if not tokens:
                tokens = ['empty']  # 为空文档添加占位符token
            tokenized_corpus.append(tokens)

        # 2. 创建并存储BM25索引
        self.bm25_index = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
        
        # 3. 严格按顺序存储paper_id，用于后续分数映射
        self.paper_ids = papers_df['paper_id'].tolist()
        self.is_fitted = True
        print(f"BM25 index fitted for {len(self.paper_ids)} papers.")

    def score(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame,
              reviewer_paper_map: Dict[str, List[str]] = None) -> pd.DataFrame:
        """
        使用已构建的论文索引，为每个审稿人计算与指定论文的匹配分数。

        :param papers_df: 需要评分的论文DataFrame（用于确定输出矩阵的行）
        :param reviewers_df: 审稿人DataFrame，必须包含 'reviewer_id' 和 'profile' 列
        :param reviewer_paper_map: 可选的审稿人到论文ID列表的映射，用于只计算相关论文的分数
        :return: 一个DataFrame，index为paper_id，columns为reviewer_id，值为BM25分数。
        """
        if self.bm25_index is None:
            raise RuntimeError("ReviewerCentricBM25Matcher has not been fitted. Please call .fit() first.")

        if reviewer_paper_map:
            total_computations = sum(len(papers) for papers in reviewer_paper_map.values())
            print(f"Optimized scoring: {len(reviewers_df)} reviewers, {total_computations} targeted computations")
        else:
            print(f"Scoring {len(papers_df)} papers against {len(reviewers_df)} reviewer queries...")

        # 预计算审稿人的查询分数（优化版：只计算相关论文）
        print("Pre-computing reviewer query scores...")
        reviewer_scores = {}

        for _, reviewer_row in reviewers_df.iterrows():
            reviewer_id = reviewer_row['reviewer_id']
            query_text = reviewer_row['profile']
            if not isinstance(query_text, str):
                query_text = ""

            # 对查询进行分词处理
            tokenized_query = query_text.split()
            if not tokenized_query:
                tokenized_query = ['empty']

            # 总是只计算该审稿人标注的论文分数（优化计算）
            if reviewer_paper_map and reviewer_id in reviewer_paper_map:
                target_papers = reviewer_paper_map[reviewer_id]
                doc_scores = self._compute_scores_for_papers(tokenized_query, target_papers)
            else:
                # 如果没有提供映射，跳过该审稿人（因为没有标注数据）
                print(f"Warning: No paper mapping found for reviewer {reviewer_id}, skipping")
                continue

            reviewer_scores[reviewer_id] = doc_scores

        print("Building scores matrix...")

        # 构建分数矩阵
        scores_matrix = []
        for _, paper_row in papers_df.iterrows():
            paper_id = paper_row['paper_id']

            # 从预计算的分数中提取这篇论文的分数
            row_scores = {}
            for reviewer_id in reviewers_df['reviewer_id']:
                if reviewer_id in reviewer_scores:
                    if reviewer_paper_map and reviewer_id in reviewer_paper_map:
                        # 优化模式：从稀疏字典中获取分数
                        row_scores[reviewer_id] = reviewer_scores[reviewer_id].get(paper_id, 0.0)
                    else:
                        # 原始模式：从完整数组中获取分数
                        paper_idx = self.paper_ids.index(paper_id) if paper_id in self.paper_ids else None
                        if paper_idx is not None:
                            row_scores[reviewer_id] = reviewer_scores[reviewer_id][paper_idx]
                        else:
                            row_scores[reviewer_id] = 0.0
                else:
                    row_scores[reviewer_id] = 0.0

            row_scores['paper_id'] = paper_id
            scores_matrix.append(row_scores)

        # 创建DataFrame并设置paper_id为索引
        scores_df = pd.DataFrame(scores_matrix)
        scores_df = scores_df.set_index('paper_id')

        # 确保返回的DataFrame的列是所有候选审稿人
        return scores_df.reindex(columns=reviewers_df['reviewer_id'], fill_value=0.0)

    def _compute_scores_for_papers(self, tokenized_query: List[str], target_paper_ids: List[str]) -> Dict[str, float]:
        """
        为指定的论文列表计算BM25分数（优化版本）

        :param tokenized_query: 分词后的查询
        :param target_paper_ids: 目标论文ID列表
        :return: 论文ID到分数的映射字典
        """
        # 创建稀疏的分数字典，只包含目标论文
        scores_dict = {}

        # 获取目标论文在索引中的位置
        target_indices = []
        for paper_id in target_paper_ids:
            if paper_id in self.paper_ids:
                idx = self.paper_ids.index(paper_id)
                target_indices.append((paper_id, idx))

        if not target_indices:
            return scores_dict

        # 计算所有论文的BM25分数（这是BM25库的限制，无法只计算部分）
        all_scores = self.bm25_index.get_scores(tokenized_query)

        # 只提取目标论文的分数
        for paper_id, idx in target_indices:
            scores_dict[paper_id] = all_scores[idx]

        return scores_dict


# --- 使用示例与单元测试 ---
if __name__ == '__main__':
    # 1. 模拟DataLoader的输出
    papers_data = {
        'paper_id': ['p1', 'p2', 'p3'],
        'title': ['Paper A', 'Paper B', 'Paper C'],
        'abstract': ['deep learning for nlp', 'graph neural networks', 'computer vision']
    }
    papers_df = pd.DataFrame(papers_data)

    reviewers_data = {
        'reviewer_id': ['r1', 'r2'],
        'profile': ['natural language processing deep learning', 'graph theory neural networks']
    }
    reviewers_df = pd.DataFrame(reviewers_data)

    # 2. 创建并训练模型
    config = {'bm25_k1': 1.2, 'bm25_b': 0.75}
    matcher = ReviewerCentricBM25Matcher(config)
    matcher.fit(papers_df, reviewers_df)

    # 3. 评分
    papers_to_score = papers_df[papers_df['paper_id'].isin(['p1', 'p2'])]  # 模拟测试集
    scores_df = matcher.score(papers_to_score, reviewers_df)

    print("Scores DataFrame:")
    print(scores_df)
    print(f"Shape: {scores_df.shape}")
    
    # 验证结果
    assert scores_df.shape == (2, 2), f"Expected shape (2, 2), got {scores_df.shape}"
    assert list(scores_df.index) == ['p1', 'p2'], f"Unexpected index: {list(scores_df.index)}"
    assert list(scores_df.columns) == ['r1', 'r2'], f"Unexpected columns: {list(scores_df.columns)}"
    
    print("✅ All tests passed!")
