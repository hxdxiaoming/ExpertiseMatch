#!/usr/bin/env python3
"""
utils/embedding_cache.py - 统一的Embedding缓存系统

提供通用的embedding缓存功能，支持：
- 自动缓存和加载embeddings
- 多种模型的embeddings管理
- 高效的批量操作
- 跨组件共享缓存
"""

import hashlib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time


class EmbeddingCache:
    """统一的Embedding缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache/embeddings"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}  # 内存缓存
        
    def _generate_cache_key(self, model_name: str, texts: List[str]) -> str:
        """
        生成缓存键
        
        Args:
            model_name: 模型名称
            texts: 文本列表
            
        Returns:
            缓存键字符串
        """
        # 使用模型名称和文本内容的hash生成唯一键
        content = f"{model_name}_{sorted(texts)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def has_embeddings(self, model_name: str, texts: List[str]) -> bool:
        """
        检查是否存在指定的embeddings
        
        Args:
            model_name: 模型名称
            texts: 文本列表
            
        Returns:
            是否存在缓存
        """
        cache_key = self._generate_cache_key(model_name, texts)
        
        # 检查内存缓存
        if cache_key in self._memory_cache:
            return True
            
        # 检查磁盘缓存
        cache_file = self._get_cache_file(cache_key)
        return cache_file.exists()
    
    def load_embeddings_by_ids(self, model_name: str, paper_ids: List[str]) -> Optional[Dict[str, np.ndarray]]:
        """
        根据paper_ids加载embeddings（兼容多种缓存格式）

        Args:
            model_name: 模型名称
            paper_ids: 论文ID列表

        Returns:
            {paper_id: embedding}字典，如果不存在则返回None
        """
        # 检查所有缓存文件，寻找匹配的模型
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # 检查模型名称是否匹配
                if cached_data.get('model_name') != model_name:
                    continue

                if 'embeddings' not in cached_data:
                    continue

                embeddings = cached_data['embeddings']

                # 格式1: 字典格式 {paper_id: embedding}
                if isinstance(embeddings, dict):
                    # 检查是否包含所需的paper_ids
                    missing_ids = [pid for pid in paper_ids if pid not in embeddings]
                    if not missing_ids:
                        print(f"✅ Loaded embeddings for {len(paper_ids)} papers from cache (dict format)")
                        return {pid: embeddings[pid] for pid in paper_ids}
                    else:
                        print(f"⚠️ Cache found but missing {len(missing_ids)} papers (dict format)")
                        continue

                # 格式2: 数组格式 + texts列表
                elif hasattr(embeddings, 'shape') and 'texts' in cached_data:
                    texts = cached_data['texts']
                    if len(texts) != len(embeddings):
                        print(f"⚠️ Mismatch between texts ({len(texts)}) and embeddings ({len(embeddings)})")
                        continue

                    # 需要从DataLoader获取paper_id到text的映射来匹配
                    # 这种格式需要额外的信息才能使用，暂时跳过
                    print(f"⚠️ Found array format cache but cannot map to paper_ids without additional info")
                    continue

                # 格式3: offline_preprocessing格式 (dataset_name + embeddings dict)
                elif 'dataset_name' in cached_data and isinstance(embeddings, dict):
                    # 检查是否包含所需的paper_ids
                    missing_ids = [pid for pid in paper_ids if pid not in embeddings]
                    if not missing_ids:
                        print(f"✅ Loaded embeddings for {len(paper_ids)} papers from cache (offline format)")
                        return {pid: embeddings[pid] for pid in paper_ids}
                    else:
                        print(f"⚠️ Cache found but missing {len(missing_ids)} papers (offline format)")
                        continue

            except Exception as e:
                print(f"⚠️ Error reading cache file {cache_file.name}: {e}")
                continue

        print(f"❌ No compatible cache found for model '{model_name}' with {len(paper_ids)} paper IDs")
        return None

    def load_embeddings(self, model_name: str, texts: List[str]) -> Optional[np.ndarray]:
        """
        加载embeddings（新格式）

        Args:
            model_name: 模型名称
            texts: 文本列表

        Returns:
            embeddings数组，如果不存在则返回None
        """
        cache_key = self._generate_cache_key(model_name, texts)

        # 先检查内存缓存
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]['embeddings']

        # 检查磁盘缓存
        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # 验证缓存数据
                if (cached_data.get('model_name') == model_name and
                    'embeddings' in cached_data and
                    'texts' in cached_data):

                    embeddings = cached_data['embeddings']
                    # 加载到内存缓存
                    self._memory_cache[cache_key] = cached_data

                    print(f"Loaded {len(embeddings)} embeddings from cache")
                    return embeddings

            except Exception as e:
                print(f"Error loading cache {cache_file}: {e}")

        return None
    
    def save_embeddings(self, model_name: str, texts: List[str], 
                       embeddings: np.ndarray) -> None:
        """
        保存embeddings到缓存
        
        Args:
            model_name: 模型名称
            texts: 文本列表
            embeddings: embeddings数组
        """
        cache_key = self._generate_cache_key(model_name, texts)
        
        cached_data = {
            'model_name': model_name,
            'texts': texts,
            'embeddings': embeddings,
            'timestamp': time.time(),
            'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'num_texts': len(texts)
        }
        
        # 保存到内存缓存
        self._memory_cache[cache_key] = cached_data
        
        # 保存到磁盘缓存
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"Saved {len(embeddings)} embeddings to cache")
        except Exception as e:
            print(f"Error saving cache {cache_file}: {e}")
    
    def load_partial_embeddings(self, model_name: str, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        加载部分embeddings，返回已缓存的部分和缺失的文本
        
        Args:
            model_name: 模型名称
            texts: 文本列表
            
        Returns:
            (已有的embeddings, 缺失的文本列表)
        """
        # 这个功能需要更复杂的缓存策略，暂时返回空
        return np.array([]), texts
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'num_cache_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'memory_cache_entries': len(self._memory_cache)
        }
    
    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        清理缓存
        
        Args:
            model_name: 如果指定，只清理该模型的缓存；否则清理所有缓存
        """
        if model_name is None:
            # 清理所有缓存
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self._memory_cache.clear()
            print("Cleared all embedding cache")
        else:
            # 清理特定模型的缓存
            to_remove = []
            for cache_key, cached_data in self._memory_cache.items():
                if cached_data.get('model_name') == model_name:
                    to_remove.append(cache_key)
                    cache_file = self._get_cache_file(cache_key)
                    if cache_file.exists():
                        cache_file.unlink()
            
            for key in to_remove:
                del self._memory_cache[key]
            
            print(f"Cleared cache for model: {model_name}")


# 全局缓存实例
_global_cache = None

def get_embedding_cache() -> EmbeddingCache:
    """获取全局embedding缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache()
    return _global_cache
