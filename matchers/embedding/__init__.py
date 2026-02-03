#!/usr/bin/env python3
"""
matchers/embedding - 嵌入向量相关模块

包含各种文本编码器和匹配策略的实现：
- 编码器：将文本转换为高质量的嵌入向量
- 策略：使用嵌入向量计算匹配分数
"""

from .encoders import SentenceTransformerEncoder, BaseTransformerEncoder, SPECTER2Encoder, COCO_DREncoder, SciBERTEncoder
from .strategies import (
    ProfileAggregationStrategy,
    ScoreAggregationStrategy,
    AverageProfileStrategy,
    MaxSimilarityStrategy,
    WeightedAverageStrategy
)

__all__ = [
    # 编码器
    "SentenceTransformerEncoder",
    "BaseTransformerEncoder",
    "SPECTER2Encoder",
    "COCO_DREncoder",
    "SciBERTEncoder",
    # 组合式匹配策略
    "ProfileAggregationStrategy",
    "ScoreAggregationStrategy",
    # 向后兼容的策略别名
    "AverageProfileStrategy",
    "MaxSimilarityStrategy",
    "WeightedAverageStrategy"
]
