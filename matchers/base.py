#!/usr/bin/env python3
"""
matchers/base.py - 定义所有模型组件的标准接口

这个文件是整个匹配器包的"设计蓝图"或"契约"。
所有具体的编码器、策略或自包含匹配器都必须遵循这里定义的接口。
这确保了整个框架的模块化和可扩展性。

设计哲学：
1. BaseEncoder: 专注于文本到向量的转换
2. BaseMatchingStrategy: 专注于向量间的匹配计算
3. BaseMatcher: 为端到端模型提供完整的匹配流程
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np


class BaseEncoder(ABC):
    """
    编码器(Encoder)的抽象基类。
    
    职责：将文本列表转换为一个嵌入向量矩阵。
    这个接口专注于单一职责：文本编码。
    
    设计原则：
    - 输入标准化：统一接收字符串列表
    - 输出标准化：统一返回NumPy数组
    - 批处理优化：支持批量编码以提高效率
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化编码器
        
        :param config: 编码器的配置参数字典
        """
        self.config = config or {}
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        对一个文本列表进行编码。

        :param texts: 一个包含待编码文本的字符串列表
        :return: 一个NumPy数组，形状为 (len(texts), embedding_dimension)
        
        注意事项：
        - 实现时应考虑空文本的处理
        - 建议支持批处理以提高效率
        - 返回的向量应该是归一化的（如果适用）
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        返回编码器输出的嵌入向量维度
        
        :return: 嵌入向量的维度
        """
        pass
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        编码单个文本的便利方法
        
        :param text: 待编码的单个文本
        :return: 形状为 (embedding_dimension,) 的1D数组
        """
        result = self.encode([text])
        return result[0]


class BaseMatchingStrategy(ABC):
    """
    匹配策略(Matching Strategy)的抽象基类。
    
    职责：接收嵌入向量，并根据特定策略计算最终匹配分数。
    这个接口专注于向量间的相似度计算和聚合策略。
    
    设计原则：
    - 策略多样性：支持不同的相似度计算方法
    - 聚合灵活性：支持不同的多向量聚合策略
    - 可配置性：通过配置参数调整策略行为
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化匹配策略
        
        :param config: 策略的配置参数字典
        """
        self.config = config or {}

    @abstractmethod
    def calculate_score(self, query_embedding: np.ndarray, reviewer_paper_embeddings: List[np.ndarray]) -> float:
        """
        计算一篇查询论文和一个审稿人之间的匹配分数。

        :param query_embedding: 查询论文的嵌入向量 (1D array)
        :param reviewer_paper_embeddings: 代表审稿人的一组论文的嵌入向量列表 (list of 1D arrays)
        :return: 一个浮点数，代表最终的匹配分数
        
        注意事项：
        - 应处理空的reviewer_paper_embeddings列表
        - 分数应该是可比较的（建议归一化到[0,1]或[-1,1]）
        - 实现时考虑向量维度一致性检查
        """
        pass
    
    def calculate_batch_scores(self, query_embedding: np.ndarray, 
                             all_reviewer_embeddings: List[List[np.ndarray]]) -> List[float]:
        """
        批量计算多个审稿人的匹配分数的便利方法
        
        :param query_embedding: 查询论文的嵌入向量
        :param all_reviewer_embeddings: 所有审稿人的论文嵌入向量列表
        :return: 对应每个审稿人的匹配分数列表
        """
        scores = []
        for reviewer_embeddings in all_reviewer_embeddings:
            score = self.calculate_score(query_embedding, reviewer_embeddings)
            scores.append(score)
        return scores


class BaseMatcher(ABC):
    """
    自包含匹配器(Self-contained Matcher)的抽象基类。
    
    职责：封装一个从数据到分数的完整匹配流程。
    适用于那些无法或不适合拆分为"编码器+策略"的模型。
    
    设计原则：
    - 端到端处理：从原始数据到最终分数
    - 两阶段设计：fit()准备阶段 + score()计算阶段
    - 标准化输出：统一的DataFrame格式
    
    典型使用场景：
    - BM25: 需要构建倒排索引
    - APT: 需要训练主题模型
    - CoF: 零样本模型，fit()可以为空
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化匹配器
        
        :param config: 匹配器的配置参数字典
        """
        self.config = config or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, reviewers_df: pd.DataFrame, papers_df: pd.DataFrame) -> None:
        """
        "训练"或"准备"模型。
        
        例如：
        - BM25: 构建倒排索引
        - APT: 训练主题模型
        - 神经网络: 训练模型参数
        - CoF: 可以直接pass（零样本）
        
        :param reviewers_df: 审稿人信息的DataFrame，包含reviewer_id和profile列
        :param papers_df: 论文信息的DataFrame，包含paper_id、title、abstract列
        
        注意事项：
        - 实现后应设置self.is_fitted = True
        - 应验证输入数据的格式和完整性
        - 可以在此阶段进行数据预处理和索引构建
        """
        pass

    @abstractmethod
    def score(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame) -> pd.DataFrame:
        """
        为所有待匹配论文和所有候选审稿人计算匹配分数。

        :param papers_df: 需要计算分数的论文DataFrame
        :param reviewers_df: 候选审稿人的DataFrame
        :return: 一个DataFrame，index为paper_id，columns为reviewer_id，值为匹配分数
        
        返回格式示例：
                    reviewer_1  reviewer_2  reviewer_3
        paper_1        0.85        0.23        0.67
        paper_2        0.12        0.91        0.45
        
        注意事项：
        - 调用前应确保已经调用过fit()
        - 分数应该是可比较的数值
        - 处理缺失数据时应返回合理的默认值（如0）
        """
        pass
    
    def predict(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame, 
                top_k: Optional[int] = None) -> pd.DataFrame:
        """
        预测并返回每篇论文的top-k审稿人推荐
        
        :param papers_df: 需要预测的论文DataFrame
        :param reviewers_df: 候选审稿人DataFrame
        :param top_k: 返回每篇论文的前k个审稿人，None表示返回所有
        :return: DataFrame，包含paper_id, reviewer_id, score列，按分数降序排列
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        scores_df = self.score(papers_df, reviewers_df)
        
        # 转换为长格式
        results = []
        for paper_id in scores_df.index:
            paper_scores = scores_df.loc[paper_id].sort_values(ascending=False)
            
            if top_k is not None:
                paper_scores = paper_scores.head(top_k)
            
            for reviewer_id, score in paper_scores.items():
                results.append({
                    'paper_id': paper_id,
                    'reviewer_id': reviewer_id,
                    'score': score
                })
        
        return pd.DataFrame(results)

    def predict_reviewer_centric(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame,
                                qrels_dict: Optional[Dict] = None) -> Dict:
        """
        为reviewer-centric任务生成预测结果

        :param papers_df: 候选论文DataFrame
        :param reviewers_df: 需要预测的审稿人DataFrame
        :param qrels_dict: 标注数据字典（可选）
        :return: Dict，格式为 {reviewer_id: [{"id": paper_id, "score": score}, ...]}
        """
        # 默认实现：基于score方法的结果转换
        scores_df = self.score(papers_df, reviewers_df)

        results = {}
        for reviewer_id in scores_df.columns:
            reviewer_scores = scores_df[reviewer_id].sort_values(ascending=False)
            results[reviewer_id] = [
                {"id": paper_id, "score": score}
                for paper_id, score in reviewer_scores.items()
                if score > 0  # 过滤掉0分
            ]

        return results

    def get_config(self) -> Dict[str, Any]:
        """
        获取当前配置
        
        :return: 配置参数字典
        """
        return self.config.copy()
    
    def set_config(self, **kwargs) -> None:
        """
        更新配置参数
        
        :param kwargs: 要更新的配置参数
        """
        self.config.update(kwargs)


# 工具函数和常量
class MatcherType:
    """匹配器类型常量"""
    ENCODER_STRATEGY = "encoder_strategy"  # 编码器+策略组合
    SELF_CONTAINED = "self_contained"      # 自包含匹配器


def validate_dataframe_format(df: pd.DataFrame, required_columns: List[str], df_name: str) -> None:
    """
    验证DataFrame格式的工具函数
    
    :param df: 要验证的DataFrame
    :param required_columns: 必需的列名列表
    :param df_name: DataFrame的名称（用于错误信息）
    :raises ValueError: 如果格式不正确
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{df_name} missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError(f"{df_name} cannot be empty")


def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    分数归一化工具函数
    
    :param scores: 要归一化的分数数组
    :param method: 归一化方法 ("minmax", "zscore", "sigmoid")
    :return: 归一化后的分数数组
    """
    if method == "minmax":
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)
    
    elif method == "zscore":
        mean_score, std_score = scores.mean(), scores.std()
        if std_score == 0:
            return np.zeros_like(scores)
        return (scores - mean_score) / std_score
    
    elif method == "sigmoid":
        return 1 / (1 + np.exp(-scores))
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
