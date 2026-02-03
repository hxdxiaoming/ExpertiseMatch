#!/usr/bin/env python3
"""
GRUTwoStageMatcher - 论文仓库的两阶段匹配实现：
1) 主题分类筛选（取top-3类别，基于作者历史类别过滤候选审稿人）
2) 专业评分排序（提交论文与审稿人论文向量的相似度，取top-3平均）

输入格式遵循框架：
- papers_df: ['paper_id','title','abstract']
- reviewers_df: ['reviewer_id','authored_paper_ids']，可附加 'categories'（每位审稿人的类别集合，可选）

注意：两阶段都复用同一GRU模型；embedding通过隐藏态获得。
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .gru_encoder import GRUArxivEncoder
from ..base import BaseMatcher, validate_dataframe_format
from utils.embedding_cache import get_embedding_cache


class GRUTwoStageMatcher(BaseMatcher):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        cfg = config or {}
        # 第一阶段参数
        self.top_k_categories = int(cfg.get('top_k_categories', 3))
        self.min_papers_per_category = int(cfg.get('min_papers_per_category', 1))
        self.max_reviewer_papers_for_ranking = int(cfg.get('max_reviewer_papers_for_ranking', 10))
        # 第二阶段参数
        self.top_k_similar_papers = int(cfg.get('top_k_similar_papers', 3))

        # 编码器（加载GRU模型）
        self.encoder = GRUArxivEncoder(**cfg.get('encoder_config', {}))

        # 类别映射（来自autonomous仓库的labels.json）
        self.labels = None  # 仅在需要做类别预测时用

    def fit(self, reviewers_df: pd.DataFrame, papers_df: pd.DataFrame, metadata: Optional[Dict] = None) -> None:
        # 校验输入
        validate_dataframe_format(papers_df, ['paper_id', 'title', 'abstract'], 'papers_df')
        validate_dataframe_format(reviewers_df, ['reviewer_id', 'authored_paper_ids'], 'reviewers_df')

        # 准备审稿人画像文本（使用 DataLoader 已构建的 'profile' 更优；不强制依赖）
        if 'profile' not in reviewers_df.columns:
            # 基于作者论文拼接标题和摘要，简单画像（fallback）
            # 这里不做重编码，第一阶段只需要类别统计；第二阶段按论文级处理
            reviewers_df = reviewers_df.copy()
            reviewers_df['profile'] = reviewers_df['authored_paper_ids'].apply(lambda x: '')

        self.reviewers_df = reviewers_df
        self.papers_df = papers_df
        self.is_fitted = True

    def _predict_topk_categories(self, texts: List[str]) -> List[List[str]]:
        # 使用编码器提供的top-k预测
        try:
            return self.encoder.predict_topk_categories(texts, k=self.top_k_categories)
        except Exception:
            return [[] for _ in texts]

    def _stage1_filter_reviewers(self, paper_text: str) -> List[str]:
        # 尝试类别预测；如果不可用，则不过滤
        topk_categories = self._predict_topk_categories([paper_text])[0]
        if not topk_categories or 'categories' not in self.reviewers_df.columns:
            return self.reviewers_df['reviewer_id'].tolist()

        # 基于审稿人历史类别进行过滤（审稿人需要提供其历史类别集合）
        def reviewer_passes(row) -> bool:
            cats = row.get('categories', [])
            if isinstance(cats, list):
                return any(c in cats for c in topk_categories)
            return False

        filtered = self.reviewers_df[self.reviewers_df.apply(reviewer_passes, axis=1)]
        if filtered.empty:
            return self.reviewers_df['reviewer_id'].tolist()
        return filtered['reviewer_id'].tolist()

    def _paper_embedding(self, title: str, abstract: str) -> np.ndarray:
        text_parts = []
        if isinstance(title, str) and title.strip():
            text_parts.append(title.strip())
        if isinstance(abstract, str) and abstract.strip():
            text_parts.append(abstract.strip())
        text = ' '.join(text_parts) if text_parts else 'empty document'
        return self.encoder.encode([text])[0]

    def _reviewer_paper_embeddings(self, reviewer_id: str) -> List[np.ndarray]:
        row = self.reviewers_df[self.reviewers_df['reviewer_id'] == reviewer_id]
        if row.empty:
            return []
        authored_ids = row.iloc[0]['authored_paper_ids'] or []
        # 只取最近/前N篇
        authored_ids = authored_ids[-self.max_reviewer_papers_for_ranking:]

        id_set = set(authored_ids)
        papers = self.papers_df[self.papers_df['paper_id'].isin(id_set)]
        texts = []
        for _, r in papers.iterrows():
            title = r.get('title', '')
            abstract = r.get('abstract', '')
            parts = []
            if isinstance(title, str) and title.strip():
                parts.append(title.strip())
            if isinstance(abstract, str) and abstract.strip():
                parts.append(abstract.strip())
            texts.append(' '.join(parts) if parts else 'empty document')
        if not texts:
            return []
        embs = self.encoder.encode(texts)
        return [embs[i] for i in range(len(texts))]

    def _expertise_score(self, query_emb: np.ndarray, reviewer_embs: List[np.ndarray]) -> float:
        if not reviewer_embs:
            return 0.0
        sims = cosine_similarity(query_emb.reshape(1, -1), np.vstack(reviewer_embs))[0]
        topk = sorted(sims, reverse=True)[: self.top_k_similar_papers]
        if not topk:
            return 0.0
        return float(np.mean(topk))

    def score(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame, metadata: Optional[Dict] = None, qrels_dict: Optional[Dict] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Call fit() before score().")

        # 检查任务类型
        task_type = metadata.get('task_type', 'paper-centric') if metadata else 'paper-centric'

        if task_type == 'reviewer-centric':
            return self._score_reviewer_centric(papers_df, reviewers_df, qrels_dict)
        else:
            return self._score_paper_centric(papers_df, reviewers_df)

    def predict_reviewer_centric(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame, qrels_dict: Optional[Dict] = None) -> Dict:
        """
        为reviewer-centric任务生成预测结果

        Returns:
            Dict: 格式为 {reviewer_id: [{"id": paper_id, "score": score}, ...]}
        """
        print("🔄 Starting GRU reviewer-centric prediction...")

        # 优先尝试使用离线预处理生成的缓存embedding
        cache = get_embedding_cache()
        model_cache_name = self.config.get('encoder_config', {}).get('model_name', 'gru_arxiv')

        # 需要的paper_id集合：目标论文 + 审稿人论文
        target_paper_ids = set(papers_df['paper_id'].tolist())
        reviewer_paper_ids = set()
        for ids in reviewers_df['authored_paper_ids']:
            reviewer_paper_ids.update(ids)
        all_needed_ids = list(target_paper_ids | reviewer_paper_ids)

        embedding_map = cache.load_embeddings_by_ids(model_cache_name, all_needed_ids)

        results = {}

        for _, reviewer_row in reviewers_df.iterrows():
            reviewer_id = reviewer_row['reviewer_id']

            # 第一步：确定审稿人的专业领域并筛选论文
            candidate_papers = self._get_candidate_papers_for_reviewer(reviewer_id, papers_df, qrels_dict)

            # 获取审稿人的论文嵌入向量
            if embedding_map:
                # 从缓存构造审稿人论文embeddings
                authored_ids = reviewer_row['authored_paper_ids'] or []
                authored_ids = authored_ids[-self.max_reviewer_papers_for_ranking:]  # 取最近的N篇
                reviewer_embs = [embedding_map[aid] for aid in authored_ids if aid in embedding_map]
            else:
                reviewer_embs = self._reviewer_paper_embeddings(reviewer_id)

            if not reviewer_embs:
                # 如果审稿人没有论文，给所有候选论文打0分
                results[reviewer_id] = [{"id": paper_id, "score": 0.0} for paper_id in candidate_papers]
                continue

            # 第二步：计算专业匹配度并对论文进行排名
            paper_scores = []
            for paper_id in candidate_papers:
                paper_row = papers_df[papers_df['paper_id'] == paper_id]
                if paper_row.empty:
                    continue

                paper_row = paper_row.iloc[0]

                # 获取论文嵌入向量
                if embedding_map and paper_id in embedding_map:
                    paper_emb = embedding_map[paper_id]
                else:
                    paper_emb = self._paper_embedding(paper_row.get('title', ''), paper_row.get('abstract', ''))

                # 计算匹配分数：论文与审稿人所有论文的相似度，取top-3平均
                score = self._expertise_score(paper_emb, reviewer_embs)
                paper_scores.append({"id": paper_id, "score": score})

            # 按分数降序排序
            paper_scores.sort(key=lambda x: x["score"], reverse=True)
            results[reviewer_id] = paper_scores

        print(f"✅ GRU reviewer-centric prediction completed")
        print(f"📊 Generated predictions for {len(results)} reviewers")

        return results

    def _score_paper_centric(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame) -> pd.DataFrame:
        """原有的paper-centric评分逻辑"""
        # 优先尝试使用离线预处理生成的缓存embedding（如已运行 offline_preprocessing.py --embedding_models gru_arxiv）
        cache = get_embedding_cache()
        model_cache_name = self.config.get('encoder_config', {}).get('model_name', 'gru_arxiv')

        # 需要的paper_id集合：目标论文 + 审稿人论文
        target_paper_ids = set(papers_df['paper_id'].tolist())
        reviewer_paper_ids = set()
        for ids in reviewers_df['authored_paper_ids']:
            reviewer_paper_ids.update(ids)
        all_needed_ids = list(target_paper_ids | reviewer_paper_ids)

        embedding_map = cache.load_embeddings_by_ids(model_cache_name, all_needed_ids)

        scores = []
        for _, p in papers_df.iterrows():
            pid = p['paper_id']
            if embedding_map and pid in embedding_map:
                query_emb = embedding_map[pid]
            else:
                query_emb = self._paper_embedding(p.get('title', ''), p.get('abstract', ''))

            # 阶段一：筛选候选审稿人（当前默认不过滤以避免缺少类别元数据）
            text_for_filter = (p.get('title', '') or '') + ' ' + (p.get('abstract', '') or '')
            candidate_reviewers = self._stage1_filter_reviewers(text_for_filter)

            for rid in candidate_reviewers:
                if embedding_map:
                    # 从缓存构造审稿人论文embeddings
                    row = self.reviewers_df[self.reviewers_df['reviewer_id'] == rid]
                    authored_ids = row.iloc[0]['authored_paper_ids'] if not row.empty else []
                    authored_ids = authored_ids[-self.max_reviewer_papers_for_ranking:]
                    r_embs = [embedding_map[aid] for aid in authored_ids if aid in embedding_map]
                else:
                    r_embs = self._reviewer_paper_embeddings(rid)
                s = self._expertise_score(query_emb, r_embs)
                scores.append({'paper_id': pid, 'reviewer_id': rid, 'score': s})

        df = pd.DataFrame(scores)
        if df.empty:
            return pd.DataFrame(index=papers_df['paper_id'], columns=reviewers_df['reviewer_id']).fillna(0.0)
        scores_df = df.pivot(index='paper_id', columns='reviewer_id', values='score').fillna(0.0)
        return scores_df

    def _score_reviewer_centric(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame, qrels_dict: Optional[Dict] = None) -> pd.DataFrame:
        """
        Reviewer-centric评分逻辑：
        第一步：确定审稿人的专业领域并筛选论文 (Filtering)
        第二步：计算专业匹配度并对论文进行排名 (Ranking)
        """
        print("🔄 Starting GRU reviewer-centric scoring...")

        # 优先尝试使用离线预处理生成的缓存embedding
        cache = get_embedding_cache()
        model_cache_name = self.config.get('encoder_config', {}).get('model_name', 'gru_arxiv')

        # 需要的paper_id集合：目标论文 + 审稿人论文
        target_paper_ids = set(papers_df['paper_id'].tolist())
        reviewer_paper_ids = set()
        for ids in reviewers_df['authored_paper_ids']:
            reviewer_paper_ids.update(ids)
        all_needed_ids = list(target_paper_ids | reviewer_paper_ids)

        embedding_map = cache.load_embeddings_by_ids(model_cache_name, all_needed_ids)

        scores = []

        for _, reviewer_row in reviewers_df.iterrows():
            reviewer_id = reviewer_row['reviewer_id']

            # 第一步：确定审稿人的专业领域并筛选论文
            candidate_papers = self._get_candidate_papers_for_reviewer(reviewer_id, papers_df, qrels_dict)

            # 获取审稿人的论文嵌入向量
            if embedding_map:
                # 从缓存构造审稿人论文embeddings
                authored_ids = reviewer_row['authored_paper_ids'] or []
                authored_ids = authored_ids[-self.max_reviewer_papers_for_ranking:]  # 取最近的N篇
                reviewer_embs = [embedding_map[aid] for aid in authored_ids if aid in embedding_map]
            else:
                reviewer_embs = self._reviewer_paper_embeddings(reviewer_id)

            if not reviewer_embs:
                # 如果审稿人没有论文，给所有候选论文打0分
                for paper_id in candidate_papers:
                    scores.append({'paper_id': paper_id, 'reviewer_id': reviewer_id, 'score': 0.0})
                continue

            # 第二步：计算专业匹配度并对论文进行排名
            for paper_id in candidate_papers:
                paper_row = papers_df[papers_df['paper_id'] == paper_id]
                if paper_row.empty:
                    continue

                paper_row = paper_row.iloc[0]

                # 获取论文嵌入向量
                if embedding_map and paper_id in embedding_map:
                    paper_emb = embedding_map[paper_id]
                else:
                    paper_emb = self._paper_embedding(paper_row.get('title', ''), paper_row.get('abstract', ''))

                # 计算匹配分数：论文与审稿人所有论文的相似度，取top-3平均
                score = self._expertise_score(paper_emb, reviewer_embs)
                scores.append({'paper_id': paper_id, 'reviewer_id': reviewer_id, 'score': score})

        df = pd.DataFrame(scores)
        if df.empty:
            return pd.DataFrame(index=papers_df['paper_id'], columns=reviewers_df['reviewer_id']).fillna(0.0)

        # 对于reviewer-centric任务，返回的矩阵仍然是：行=论文，列=审稿人
        scores_df = df.pivot(index='paper_id', columns='reviewer_id', values='score').fillna(0.0)

        print(f"✅ GRU reviewer-centric scoring completed")
        print(f"📊 Scores matrix shape: {scores_df.shape}")

        return scores_df

    def _get_candidate_papers_for_reviewer(self, reviewer_id: str, papers_df: pd.DataFrame, qrels_dict: Optional[Dict] = None) -> List[str]:
        """
        第一步：确定审稿人的专业领域并筛选论文

        Args:
            reviewer_id: 审稿人ID
            papers_df: 论文DataFrame
            qrels_dict: 标注数据字典

        Returns:
            候选论文ID列表
        """
        # 首先从qrels中获取该审稿人有标注的论文
        qrel_candidate_papers = []
        if qrels_dict:
            # qrels_dict是一个字典，key为qrel类型(如'raw', 'soft', 'hard')，value为DataFrame
            # 使用'raw'类型的qrel数据（如果存在）
            qrel_types = ['raw', 'soft', 'hard']  # 按优先级排序
            qrels_df = None

            for qrel_type in qrel_types:
                if qrel_type in qrels_dict:
                    qrels_df = qrels_dict[qrel_type]
                    break

            if qrels_df is not None and hasattr(qrels_df, 'iterrows'):
                # DataFrame格式：包含reviewer_id, paper_id, relevance_score列
                reviewer_qrels = qrels_df[qrels_df['reviewer_id'] == reviewer_id]
                qrel_candidate_papers = reviewer_qrels['paper_id'].tolist()

        # 如果没有qrel数据，使用所有论文作为候选
        if not qrel_candidate_papers:
            print(f"⚠️ Warning: No qrel data for reviewer {reviewer_id}, using all papers")
            qrel_candidate_papers = papers_df['paper_id'].tolist()

        # print(f"📋 Reviewer {reviewer_id}: {len(qrel_candidate_papers)} papers in qrel")

        # 获取审稿人的过往论文（最近10篇）用于主题分析
        reviewer_row = self.reviewers_df[self.reviewers_df['reviewer_id'] == reviewer_id]
        if reviewer_row.empty:
            return qrel_candidate_papers

        authored_ids = reviewer_row.iloc[0]['authored_paper_ids'] or []
        recent_authored_ids = authored_ids[-10:]  # 取最近10篇论文

        if not recent_authored_ids:
            return qrel_candidate_papers

        # 获取审稿人过往论文的文本
        reviewer_papers = self.papers_df[self.papers_df['paper_id'].isin(recent_authored_ids)]
        reviewer_texts = []
        for _, paper in reviewer_papers.iterrows():
            title = paper.get('title', '') or ''
            abstract = paper.get('abstract', '') or ''
            text = f"{title} {abstract}".strip()
            if text:
                reviewer_texts.append(text)

        if not reviewer_texts:
            return qrel_candidate_papers

        # 利用主题分类模型，为审稿人的每篇过往论文确定主题类别
        reviewer_categories = []
        for text in reviewer_texts:
            categories = self._predict_topk_categories([text])[0]
            reviewer_categories.extend(categories)

        # 汇总主题，形成审稿人专业知识领域的"主题集合"
        reviewer_topic_set = set(reviewer_categories)

        if not reviewer_topic_set:
            return qrel_candidate_papers

        # 在qrel候选论文中进行主题筛选
        qrel_papers_df = papers_df[papers_df['paper_id'].isin(qrel_candidate_papers)]
        topic_filtered_papers = []

        for _, paper in qrel_papers_df.iterrows():
            title = paper.get('title', '') or ''
            abstract = paper.get('abstract', '') or ''
            paper_text = f"{title} {abstract}".strip()

            if paper_text:
                paper_categories = self._predict_topk_categories([paper_text])[0]
                # 检查是否有主题重叠
                if any(cat in reviewer_topic_set for cat in paper_categories):
                    topic_filtered_papers.append(paper['paper_id'])

        # 如果主题筛选后没有匹配的论文，返回所有qrel论文（避免空结果）
        if not topic_filtered_papers:
            print(f"⚠️ Warning: No topic-matched papers for reviewer {reviewer_id}, using all qrel papers")
            return qrel_candidate_papers

        # print(f"📋 Reviewer {reviewer_id}: {len(reviewer_topic_set)} topics, {len(topic_filtered_papers)}/{len(qrel_candidate_papers)} candidate papers")
        return topic_filtered_papers


