#!/usr/bin/env python3
"""
matchers - 论文审稿人匹配器包

这个包提供了一个统一的框架来实现和使用各种论文-审稿人匹配算法。

核心组件：
- BaseEncoder: 文本编码器的抽象基类
- BaseMatchingStrategy: 匹配策略的抽象基类  
- BaseMatcher: 自包含匹配器的抽象基类

设计哲学：
1. 模块化：将编码和匹配逻辑分离
2. 可扩展：通过继承基类轻松添加新算法
3. 标准化：统一的接口和数据格式
"""

from .base import (
    BaseEncoder,
    BaseMatchingStrategy, 
    BaseMatcher,
    MatcherType,
    validate_dataframe_format,
    normalize_scores
)

# Optional GRU-based encoder and matcher
try:
    from .gru.gru_encoder import GRUArxivEncoder  # noqa: F401
    from .gru.gru_matcher import GRUTwoStageMatcher  # noqa: F401
except Exception:
    # Allow package import even if GRU dependencies are missing
    GRUArxivEncoder = None
    GRUTwoStageMatcher = None

# TPMS matcher
try:
    from .tpms.tpms_matcher import TPMSTwoStageMatcher  # noqa: F401
except Exception:
    # Allow package import even if TPMS dependencies are missing
    TPMSTwoStageMatcher = None

__version__ = "1.0.0"
__author__ = "Paper Reviewer Matching System"

__all__ = [
    "BaseEncoder",
    "BaseMatchingStrategy", 
    "BaseMatcher",
    "MatcherType",
    "validate_dataframe_format",
    "normalize_scores",
    "GRUArxivEncoder",
    "GRUTwoStageMatcher",
    "TPMSTwoStageMatcher",
]
