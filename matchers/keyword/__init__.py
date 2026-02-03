#!/usr/bin/env python3
"""
matchers/keyword - 基于关键词的匹配器模块

包含基于传统信息检索方法的匹配器实现，如BM25等。
"""

from .bm25_matcher import BM25Matcher

__all__ = ["BM25Matcher"]
