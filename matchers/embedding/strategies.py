#!/usr/bin/env python3
"""
matchers/embedding/strategies.py - 实现多种使用嵌入向量进行匹配的策略

核心设计思路：组合与配置
为了实现各种策略，同时避免代码冗余，我们将策略的各个"组件"拆分出来，
然后通过配置来"组装"成一个完整的策略。

可配置的"组件"包括：
- 相似度函数 (Similarity Function): 余弦相似度 vs 点积
- 聚合方式 (Aggregation Method): 平均 vs 按位最大值 vs 加权平均
- Top-K值 (k): 选择多少篇论文进行处理

策略家族：
1. Profile-level Aggregation: 先聚合，后比较
2. Score-level Aggregation: 先比较，后聚合
"""

import numpy as np
from typing import List, Literal
from ..base import BaseMatchingStrategy

# 可选依赖导入
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# --- 辅助函数，用于解耦相似度计算 ---
def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray, func: Literal['cosine', 'dot']) -> np.ndarray:
    """
    计算两个向量/矩阵之间的相似度

    Args:
        vec1: 第一个向量或矩阵
        vec2: 第二个向量或矩阵
        func: 相似度函数类型 ('cosine' 或 'dot')

    Returns:
        相似度矩阵
    """
    if func == 'cosine':
        if SKLEARN_AVAILABLE:
            return cosine_similarity(vec1, vec2)
        else:
            # 手动实现余弦相似度
            vec1_norm = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-8)
            vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)
            return np.dot(vec1_norm, vec2_norm.T)
    elif func == 'dot':
        return np.dot(vec1, vec2.T)
    else:
        raise ValueError(f"Unknown similarity function: {func}")


def _cosine_similarity_single(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个单一向量的余弦相似度（向后兼容）"""
    result = calculate_similarity(a.reshape(1, -1), b.reshape(1, -1), 'cosine')
    return float(result[0, 0])


# ----------------------------------------------------------------------------------
# 策略家族 1: 先聚合，后比较 (Profile-level Aggregation)
# ----------------------------------------------------------------------------------

class ProfileAggregationStrategy(BaseMatchingStrategy):
    """
    先将审稿人的多篇论文嵌入聚合成一个单一的画像向量，然后再与查询论文进行比较。

    这是"先聚合，后比较"的策略家族，支持多种配置组合：
    - k: 选择前k篇论文
    - aggregation: 聚合方式 ('mean', 'max')
    - similarity: 相似度函数 ('cosine', 'dot')
    """

    def __init__(self,
                 k: int = 10,
                 aggregation: Literal['mean', 'weighted_mean', 'max'] = 'mean',
                 similarity: Literal['cosine', 'dot'] = 'cosine',
                 config: dict = None):
        """
        初始化Profile聚合策略

        Args:
            k: 选择前k篇论文进行聚合
            aggregation: 聚合方式 ('mean'平均, 'weighted_mean'加权平均, 'max'按位最大值)
            similarity: 相似度函数 ('cosine'余弦相似度, 'dot'点积)
            config: 额外配置参数
        """
        super().__init__(config)
        self.k = k
        self.aggregation = aggregation
        self.similarity = similarity
        self.strategy_name = f"ProfileAgg-{aggregation}-{similarity}-k{k}"

        # 为加权平均创建权重（线性衰减）
        if aggregation == 'weighted_mean':
            self.weights = np.linspace(1.0, 0.5, k)
            self.weights = self.weights / np.sum(self.weights)  # 归一化

        print(f"Initialized ProfileAggregationStrategy (k={k}, agg='{aggregation}', sim='{similarity}')")

    def calculate_score(self, query_embedding: np.ndarray, reviewer_paper_embeddings: List[np.ndarray]) -> float:
        """
        计算匹配分数

        Args:
            query_embedding: 查询论文的嵌入向量
            reviewer_paper_embeddings: 审稿人的论文嵌入向量列表

        Returns:
            匹配分数
        """
        if not reviewer_paper_embeddings:
            return 0.0

        try:
            # 1. 根据聚合方式选择不同的处理逻辑
            if self.aggregation in ['weighted_mean', 'weighted']:  # 支持 weighted 别名
                # 查询感知的位置加权：先排序，再加权
                return self._calculate_query_aware_weighted_score(query_embedding, reviewer_paper_embeddings)
            else:
                # 传统方式：直接选择前k篇论文
                top_k_embeddings = np.vstack(reviewer_paper_embeddings[:self.k])

                if self.aggregation == 'mean':
                    profile_vector = np.mean(top_k_embeddings, axis=0)
                elif self.aggregation == 'max':
                    profile_vector = np.max(top_k_embeddings, axis=0)
                else:
                    raise ValueError(f"Unknown aggregation method: {self.aggregation}")

                # 计算最终分数
                score_matrix = calculate_similarity(
                    query_embedding.reshape(1, -1),
                    profile_vector.reshape(1, -1),
                    func=self.similarity
                )
                return float(score_matrix[0, 0])

        except Exception as e:
            print(f"Warning: Error in ProfileAggregationStrategy.calculate_score: {e}")
            return 0.0

    def _calculate_query_aware_weighted_score(self, query_embedding: np.ndarray, reviewer_paper_embeddings: List[np.ndarray]) -> float:
        """
        查询感知的加权平均计算
        先根据与查询论文的相似度排序，再按位置加权聚合
        🚀 优化版本：使用批量矩阵运算提升性能
        """
        # 1. 🚀 批量计算所有相似度 - 与ScoreAggregationStrategy相同的高性能方法
        all_scores = calculate_similarity(
            query_embedding.reshape(1, -1),
            np.vstack(reviewer_paper_embeddings),  # 所有embedding堆叠成矩阵
            func=self.similarity
        )[0]  # 得到一个1D的分数数组

        # 2. 按相似度排序（降序）
        sorted_indices = np.argsort(all_scores)[::-1]

        # 3. 选择前k篇最相关的论文
        k_to_use = min(self.k, len(reviewer_paper_embeddings))
        top_k_indices = sorted_indices[:k_to_use]
        top_k_embeddings = [reviewer_paper_embeddings[i] for i in top_k_indices]

        # 4. 按位置分配权重（最相关的权重最高）
        position_weights = np.linspace(1.0, 0.5, len(top_k_embeddings))
        position_weights = position_weights / np.sum(position_weights)

        # 5. 加权聚合得到审稿人profile
        top_k_embeddings = np.vstack(top_k_embeddings)
        reviewer_profile = np.average(top_k_embeddings, axis=0, weights=position_weights)

        # 6. 计算最终相似度
        final_score_matrix = calculate_similarity(
            query_embedding.reshape(1, -1),
            reviewer_profile.reshape(1, -1),
            func=self.similarity
        )

        return float(final_score_matrix[0, 0])


# ----------------------------------------------------------------------------------
# 策略家族 2: 先比较，后聚合 (Score-level Aggregation)
# ----------------------------------------------------------------------------------

class ScoreAggregationStrategy(BaseMatchingStrategy):
    """
    先计算查询论文与审稿人每一篇论文的相似度分数，然后对这些分数进行聚合。

    这是"先比较，后聚合"的策略家族，支持多种配置组合：
    - k: 选择前k个最高分数
    - aggregation: 聚合方式 ('mean', 'weighted_mean', 'max')
    - similarity: 相似度函数 ('cosine', 'dot')
    """

    def __init__(self,
                 k: int = 10,
                 aggregation: Literal['mean', 'weighted_mean', 'max'] = 'mean',
                 similarity: Literal['cosine', 'dot'] = 'cosine',
                 config: dict = None):
        """
        初始化Score聚合策略

        Args:
            k: 选择前k个最高分数进行聚合
            aggregation: 聚合方式 ('mean'平均, 'weighted_mean'加权平均, 'max'最大值)
            similarity: 相似度函数 ('cosine'余弦相似度, 'dot'点积)
            config: 额外配置参数
        """
        super().__init__(config)
        self.k = k
        self.aggregation = aggregation
        self.similarity = similarity
        self.strategy_name = f"ScoreAgg-{aggregation}-{similarity}-k{k}"

        # 为加权平均创建权重（线性衰减）
        self.weights = np.linspace(1.0, 0.5, k)
        self.weights = self.weights / np.sum(self.weights)  # 归一化

        print(f"Initialized ScoreAggregationStrategy (k={k}, agg='{aggregation}', sim='{similarity}')")

    def calculate_score(self, query_embedding: np.ndarray, reviewer_paper_embeddings: List[np.ndarray]) -> float:
        """
        计算匹配分数

        Args:
            query_embedding: 查询论文的嵌入向量
            reviewer_paper_embeddings: 审稿人的论文嵌入向量列表

        Returns:
            匹配分数
        """
        if not reviewer_paper_embeddings:
            return 0.0

        try:
            # 1. 计算所有论文的相似度分数
            all_scores = calculate_similarity(
                query_embedding.reshape(1, -1),
                np.vstack(reviewer_paper_embeddings),
                func=self.similarity
            )[0]  # 得到一个1D的分数数组

            # 2. 对分数进行排序并选取前k个
            top_k_scores = np.sort(all_scores)[::-1][:self.k]

            # 3. 根据指定方式聚合分数
            if self.aggregation == 'mean':
                final_score = np.mean(top_k_scores)
            elif self.aggregation in ['weighted_mean', 'weighted']:  # 支持 weighted 别名
                weights_to_use = self.weights[:len(top_k_scores)]
                final_score = np.sum(top_k_scores * weights_to_use)
            elif self.aggregation == 'max':
                final_score = np.max(top_k_scores)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")

            return float(final_score)

        except Exception as e:
            print(f"Warning: Error in ScoreAggregationStrategy.calculate_score: {e}")
            return 0.0


# ----------------------------------------------------------------------------------
# 向后兼容的策略别名
# ----------------------------------------------------------------------------------

class AverageProfileStrategy(ProfileAggregationStrategy):
    """平均画像策略 - 向后兼容别名"""
    def __init__(self, config: dict = None):
        super().__init__(k=10, aggregation='mean', similarity='cosine', config=config)
        self.strategy_name = "AverageProfile"


class MaxSimilarityStrategy(ScoreAggregationStrategy):
    """最大相似度策略 - 向后兼容别名"""
    def __init__(self, config: dict = None):
        super().__init__(k=1, aggregation='max', similarity='cosine', config=config)
        self.strategy_name = "MaxSimilarity"


class WeightedAverageStrategy(ScoreAggregationStrategy):
    """加权平均策略 - 向后兼容别名"""
    def __init__(self, config: dict = None):
        k = 10
        if config and 'k' in config:
            k = config['k']
        super().__init__(k=k, aggregation='weighted_mean', similarity='cosine', config=config)
        self.strategy_name = "WeightedAverage"


# --- 使用示例与单元测试 ---
if __name__ == '__main__':
    print("=" * 60)
    print("测试组合式匹配策略")
    print("=" * 60)

    # 模拟数据
    np.random.seed(42)  # 确保结果可重现
    query_emb = np.random.rand(128)
    reviewer_papers_embs = [np.random.rand(128) for _ in range(20)]

    print("--- 测试不同的策略组合 ---\n")

    # 场景1: 取前5篇论文嵌入做平均，然后计算余弦相似度
    strategy1 = ProfileAggregationStrategy(k=5, aggregation='mean', similarity='cosine')
    score1 = strategy1.calculate_score(query_emb, reviewer_papers_embs)
    print(f"Profile-Mean-Cosine (k=5): {score1:.4f}")

    # 场景2: 取前5篇论文嵌入做按位最大值，然后计算点积
    strategy2 = ProfileAggregationStrategy(k=5, aggregation='max', similarity='dot')
    score2 = strategy2.calculate_score(query_emb, reviewer_papers_embs)
    print(f"Profile-Max-Dot (k=5): {score2:.4f}")

    # 场景3: 计算所有相似度后，取前10个最高分做加权平均
    strategy3 = ScoreAggregationStrategy(k=10, aggregation='weighted_mean', similarity='cosine')
    score3 = strategy3.calculate_score(query_emb, reviewer_papers_embs)
    print(f"Score-WeightedMean-Cosine (k=10): {score3:.4f}")

    # 场景4: 取最高相似度分数
    strategy4 = ScoreAggregationStrategy(k=1, aggregation='max', similarity='cosine')
    score4 = strategy4.calculate_score(query_emb, reviewer_papers_embs)
    print(f"Score-Max-Cosine (k=1): {score4:.4f}")

    print("\n--- 测试向后兼容的策略别名 ---\n")

    # 测试向后兼容的策略
    avg_strategy = AverageProfileStrategy()
    max_strategy = MaxSimilarityStrategy()
    weighted_strategy = WeightedAverageStrategy({'k': 5})

    avg_score = avg_strategy.calculate_score(query_emb, reviewer_papers_embs)
    max_score = max_strategy.calculate_score(query_emb, reviewer_papers_embs)
    weighted_score = weighted_strategy.calculate_score(query_emb, reviewer_papers_embs)

    print(f"AverageProfileStrategy: {avg_score:.4f}")
    print(f"MaxSimilarityStrategy: {max_score:.4f}")
    print(f"WeightedAverageStrategy: {weighted_score:.4f}")

    print("\n✅ 组合式策略测试完成！")
    print("\n💡 优势:")
    print("  - 通过配置组合实现多种策略")
    print("  - 避免代码重复")
    print("  - 保持向后兼容性")
    print("  - 易于扩展新的组合")


# ----------------------------------------------------------------------------------
# Reviewer-Centric策略家族
# ----------------------------------------------------------------------------------

class ReviewerCentricProfileAggregationStrategy(BaseMatchingStrategy):
    """
    Reviewer-Centric Profile聚合策略
    给定审稿人，构建其profile embedding，然后与候选论文计算相似度

    方法①：先聚合审稿人的论文embedding，再与候选论文比较
    - 支持mean, weighted_mean, max聚合方式
    - weighted_mean使用查询感知的位置加权
    """

    def __init__(self,
                 k: int = 25,
                 aggregation: Literal['mean', 'weighted_mean', 'max'] = 'mean',
                 similarity: Literal['cosine', 'dot'] = 'cosine',
                 config: dict = None):
        """
        初始化Reviewer-Centric Profile聚合策略

        Args:
            k: 选择审稿人前k篇论文进行聚合
            aggregation: 聚合方式
            similarity: 相似度函数
            config: 额外配置参数
        """
        super().__init__(config)
        self.k = k
        self.aggregation = aggregation
        self.similarity = similarity
        self.strategy_name = f"ReviewerCentricProfileAgg-{aggregation}-{similarity}-k{k}"

        print(f"Initialized ReviewerCentricProfileAggregationStrategy (k={k}, agg='{aggregation}', sim='{similarity}')")

    def calculate_reviewer_paper_score(self, reviewer_paper_embeddings: List[np.ndarray],
                                     target_paper_embedding: np.ndarray) -> float:
        """
        计算审稿人与目标论文的匹配分数

        Args:
            reviewer_paper_embeddings: 审稿人的论文embedding列表
            target_paper_embedding: 目标论文的embedding

        Returns:
            匹配分数
        """
        if not reviewer_paper_embeddings:
            return 0.0

        try:
            # 1. 根据聚合方式选择不同的处理逻辑
            if self.aggregation in ['weighted_mean', 'weighted']:  # 支持 weighted 别名
                # 查询感知的位置加权：先排序，再加权
                return self._calculate_query_aware_weighted_score(target_paper_embedding, reviewer_paper_embeddings)
            else:
                # 传统方式：直接选择前k篇论文
                k_to_use = min(self.k, len(reviewer_paper_embeddings))
                top_k_embeddings = np.vstack(reviewer_paper_embeddings[:k_to_use])

                if self.aggregation == 'mean':
                    reviewer_profile = np.mean(top_k_embeddings, axis=0)
                elif self.aggregation == 'max':
                    reviewer_profile = np.max(top_k_embeddings, axis=0)
                else:
                    raise ValueError(f"Unknown aggregation method: {self.aggregation}")

                # 计算审稿人profile与目标论文的相似度
                score_matrix = calculate_similarity(
                    reviewer_profile.reshape(1, -1),
                    target_paper_embedding.reshape(1, -1),
                    func=self.similarity
                )
                return float(score_matrix[0, 0])

        except Exception as e:
            print(f"Warning: Error in ReviewerCentricProfileAggregationStrategy: {e}")
            return 0.0

    def _calculate_query_aware_weighted_score(self, target_paper_embedding: np.ndarray,
                                            reviewer_paper_embeddings: List[np.ndarray]) -> float:
        """
        查询感知的加权平均计算（Reviewer-Centric版本）
        先根据与目标论文的相似度排序审稿人论文，再按位置加权聚合
        🚀 优化版本：使用批量矩阵运算提升性能
        """
        # 1. 🚀 批量计算所有相似度 - 与ScoreAggregationStrategy相同的高性能方法
        all_scores = calculate_similarity(
            target_paper_embedding.reshape(1, -1),
            np.vstack(reviewer_paper_embeddings),  # 所有embedding堆叠成矩阵
            func=self.similarity
        )[0]  # 得到一个1D的分数数组

        # 2. 按相似度排序（降序）
        sorted_indices = np.argsort(all_scores)[::-1]

        # 3. 选择前k篇最相关的论文
        k_to_use = min(self.k, len(reviewer_paper_embeddings))
        top_k_indices = sorted_indices[:k_to_use]
        top_k_embeddings = [reviewer_paper_embeddings[i] for i in top_k_indices]

        # 4. 按位置分配权重（最相关的权重最高）
        position_weights = np.linspace(1.0, 0.5, len(top_k_embeddings))
        position_weights = position_weights / np.sum(position_weights)

        # 5. 加权聚合得到审稿人profile
        top_k_embeddings = np.vstack(top_k_embeddings)
        reviewer_profile = np.average(top_k_embeddings, axis=0, weights=position_weights)

        # 6. 计算最终相似度
        final_score_matrix = calculate_similarity(
            reviewer_profile.reshape(1, -1),
            target_paper_embedding.reshape(1, -1),
            func=self.similarity
        )

        return float(final_score_matrix[0, 0])

    def calculate_score(self, query_embedding: np.ndarray, reviewer_paper_embeddings: List[np.ndarray]) -> float:
        """
        实现抽象方法 - 为了兼容性，但在reviewer-centric模式下不会被调用
        实际使用calculate_reviewer_paper_score方法
        """
        # 这个方法在reviewer-centric模式下不会被调用
        # 但需要实现以满足抽象基类要求
        return 0.0


class ReviewerCentricScoreAggregationStrategy(BaseMatchingStrategy):
    """
    Reviewer-Centric Score聚合策略
    给定审稿人，先计算其每篇论文与候选论文的相似度，再聚合分数

    方法②：先计算候选论文与审稿人每篇论文的相似度，再聚合分数
    - 支持max, mean, weighted_mean聚合方式
    """

    def __init__(self,
                 k: int = 10,
                 aggregation: Literal['max', 'mean', 'weighted_mean'] = 'mean',
                 similarity: Literal['cosine', 'dot'] = 'cosine',
                 config: dict = None):
        """
        初始化Reviewer-Centric Score聚合策略

        Args:
            k: 聚合前k个最高分数
            aggregation: 聚合方式
            similarity: 相似度函数
            config: 额外配置参数
        """
        super().__init__(config)
        self.k = k
        self.aggregation = aggregation
        self.similarity = similarity
        self.strategy_name = f"ReviewerCentricScoreAgg-{aggregation}-{similarity}-k{k}"

        print(f"Initialized ReviewerCentricScoreAggregationStrategy (k={k}, agg='{aggregation}', sim='{similarity}')")

    def calculate_reviewer_paper_score(self, reviewer_paper_embeddings: List[np.ndarray],
                                     target_paper_embedding: np.ndarray) -> float:
        """
        计算审稿人与目标论文的匹配分数

        Args:
            reviewer_paper_embeddings: 审稿人的论文embedding列表
            target_paper_embedding: 目标论文的embedding

        Returns:
            匹配分数
        """
        if not reviewer_paper_embeddings:
            return 0.0

        try:
            # 1. 计算目标论文与审稿人每篇论文的相似度
            similarities = []
            for paper_emb in reviewer_paper_embeddings:
                sim_matrix = calculate_similarity(
                    target_paper_embedding.reshape(1, -1),
                    paper_emb.reshape(1, -1),
                    func=self.similarity
                )
                similarities.append(float(sim_matrix[0, 0]))

            # 2. 根据聚合方式处理分数
            if self.aggregation == 'max':
                # 取最大相似度
                return max(similarities)

            elif self.aggregation == 'mean':
                # 取前k个最高分数的平均
                k_to_use = min(self.k, len(similarities))
                top_k_scores = sorted(similarities, reverse=True)[:k_to_use]
                return np.mean(top_k_scores)

            elif self.aggregation in ['weighted_mean', 'weighted']:  # 支持 weighted 别名
                # 取前k个最高分数的加权平均
                k_to_use = min(self.k, len(similarities))
                top_k_scores = sorted(similarities, reverse=True)[:k_to_use]

                # 按位置分配权重（最高分权重最大）
                position_weights = np.linspace(1.0, 0.5, len(top_k_scores))
                position_weights = position_weights / np.sum(position_weights)

                return np.average(top_k_scores, weights=position_weights)

            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        except Exception as e:
            print(f"Warning: Error in ReviewerCentricScoreAggregationStrategy: {e}")
            return 0.0

    def calculate_score(self, query_embedding: np.ndarray, reviewer_paper_embeddings: List[np.ndarray]) -> float:
        """
        实现抽象方法 - 为了兼容性，但在reviewer-centric模式下不会被调用
        实际使用calculate_reviewer_paper_score方法
        """
        # 这个方法在reviewer-centric模式下不会被调用
        # 但需要实现以满足抽象基类要求
        return 0.0
