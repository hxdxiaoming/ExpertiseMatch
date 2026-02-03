#!/usr/bin/env python3
"""
matchers/embedding/encoders.py - 文本编码器实现

提供多种将文本转换为嵌入向量的编码器：
1. SentenceTransformerEncoder - 通用sentence-transformers编码器
2. SPECTER2Encoder - 专用SPECTER2编码器（支持Adapter机制）
"""

import numpy as np
from typing import List, Optional, Dict, Any
import sys
import os

# 导入基类
from ..base import BaseEncoder

# 添加项目路径以导入utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.embedding_cache import get_embedding_cache

# 可选依赖导入
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 单独尝试导入adapters，避免影响transformers
try:
    from adapters import AutoAdapterModel
    ADAPTERS_AVAILABLE = True
except (ImportError, RuntimeError):
    ADAPTERS_AVAILABLE = False

# CoF现在作为独立模块导入，不需要特殊处理


class SentenceTransformerEncoder(BaseEncoder):
    """
    基于sentence-transformers库的通用编码器
    
    适用于：
    - SPECTER v1 (allenai/specter)
    - SciNCL (malteos/scincl)
    - 其他sentence-transformers兼容模型
    """
    
    def __init__(self, model_name: str = "allenai/specter", 
                 device: Optional[str] = None, 
                 batch_size: int = 32,
                 max_length: int = 512,
                 **kwargs):
        """
        初始化SentenceTransformer编码器
        
        Args:
            model_name: 模型名称或路径
            device: 设备 ('cuda', 'cpu', 或 None 自动选择)
            batch_size: 批处理大小
            max_length: 最大序列长度
            **kwargs: 传递给SentenceTransformer的额外参数
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for SentenceTransformerEncoder. "
                "Install with: pip install torch"
            )

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEncoder. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 设备选择
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载模型
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device, **kwargs)
        
        # 获取嵌入维度
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ SentenceTransformer loaded: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Embedding dimension: {self._embedding_dim}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        编码文本列表为嵌入向量矩阵

        Args:
            texts: 文本列表

        Returns:
            嵌入向量矩阵，形状为 (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self._embedding_dim))

        # 优先尝试从缓存中直接返回（如果存在文本级缓存实现）
        try:
            cache = get_embedding_cache()
            cached_embeddings = cache.load_embeddings(self.model_name, texts)
            if cached_embeddings is not None:
                print(f"Using cached embeddings for {len(texts)} texts")
                return cached_embeddings
        except Exception:
            # 缓存不可用或未实现文本级缓存，直接计算
            pass

        # 直接使用 SentenceTransformer 进行编码
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings


    
    @property
    def embedding_dimension(self) -> int:
        """返回嵌入向量维度"""
        return self._embedding_dim


class BaseTransformerEncoder(BaseEncoder):
    """
    通用的基于transformers库的编码器基类

    封装了手动进行"分词 -> 前馈 -> 池化"的通用逻辑，
    避免在不同编码器中重复相同的代码。
    """

    def __init__(self,
                 model_name: str,
                 device: Optional[str] = None,
                 batch_size: int = 16,
                 max_length: int = 512,
                 pooling_strategy: str = "cls"):
        """
        初始化基础Transformer编码器

        Args:
            model_name: 模型名称或路径
            device: 设备
            batch_size: 批处理大小
            max_length: 最大序列长度
            pooling_strategy: 池化策略 ('cls', 'mean', 'max')
        """
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for BaseTransformerEncoder. "
                "Install with: pip install torch"
            )

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for BaseTransformerEncoder. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy

        # 设备选择
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 加载模型和分词器
        self._load_model()

        print(f"✅ BaseTransformer loaded: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Embedding dimension: {self._embedding_dim}")
        print(f"   Pooling strategy: {pooling_strategy}")

    def _load_model(self):
        """加载模型和分词器（子类可以重写此方法）"""
        print(f"Loading Transformer model: {self.model_name}")

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        # 移动到指定设备
        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

        # 获取嵌入维度
        self._embedding_dim = self.model.config.hidden_size

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        通用的编码流程：分词 -> 前馈 -> 池化

        Args:
            texts: 文本列表

        Returns:
            嵌入向量矩阵，形状为 (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self._embedding_dim))

        print(f"Encoding {len(texts)} texts with BaseTransformerEncoder logic...")
        all_embeddings = []

        # 批处理编码
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)

        # 合并所有批次的结果
        return np.vstack(all_embeddings)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """编码一个批次的文本"""
        # 分词
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前馈推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # 池化
        embeddings = self._pool_embeddings(hidden_states, inputs['attention_mask'])

        # 转换为numpy数组
        return embeddings.cpu().numpy()

    def _pool_embeddings(self, hidden_states, attention_mask):
        """
        对隐藏状态进行池化

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        Returns:
            pooled_embeddings: (batch_size, hidden_size)
        """
        if self.pooling_strategy == "cls":
            # 使用[CLS] token的表示
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == "mean":
            # 平均池化（忽略padding tokens）
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling_strategy == "max":
            # 最大池化
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9  # 将padding位置设为很小的值
            return torch.max(hidden_states, 1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    @property
    def embedding_dimension(self) -> int:
        """返回嵌入向量维度"""
        return self._embedding_dim


class SPECTER2Encoder(BaseTransformerEncoder):
    """
    专用SPECTER2编码器，继承通用逻辑，只添加Adapter处理

    SPECTER2的特殊性：
    1. 继承BaseTransformerEncoder的通用逻辑
    2. 添加Adapter机制支持
    """

    def __init__(self,
                 model_name: str = "allenai/specter2_base",
                 adapter_name: str = "allenai/specter2",
                 device: Optional[str] = None,
                 batch_size: int = 16,
                 max_length: int = 512,
                 pooling_strategy: str = "cls"):
        """
        初始化SPECTER2编码器

        Args:
            model_name: 基础模型名称或路径
            adapter_name: Adapter名称或路径
            device: 设备
            batch_size: 批处理大小
            max_length: 最大序列长度
            pooling_strategy: 池化策略
        """
        self.adapter_name = adapter_name

        # 调用父类初始化
        super().__init__(model_name, device, batch_size, max_length, pooling_strategy)

        print(f"   Adapter: {adapter_name}")

    def _load_model(self):
        """重写模型加载方法，添加Adapter支持"""
        print(f"Loading SPECTER2 model: {self.model_name}")

        # 根据是否有adapters库选择加载方式
        if ADAPTERS_AVAILABLE and self.adapter_name:
            print(f"Loading model with adapter support...")
            
            # 使用本地的SPECTER2 base模型路径
            base_model_path = "models/specter2/base"
            print(f"Loading base model from: {base_model_path}")
            
            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            self.model = AutoAdapterModel.from_pretrained(base_model_path)

            # 根据adapter类型确定本地adapter路径和标识符
            if 'proximity' in self.adapter_name.lower():
                adapter_path = "models/specter2/adapters/proximity"
                load_as = "specter2_proximity"
                print(f"Loading Proximity adapter from: {adapter_path}")
            elif 'classification' in self.adapter_name.lower():
                adapter_path = "models/specter2/adapters/classification"
                load_as = "specter2_classification"
                print(f"Loading Classification adapter from: {adapter_path}")
            elif 'adhoc' in self.adapter_name.lower():
                adapter_path = "models/specter2/adapters/adhoc"
                load_as = "specter2_adhoc"
                print(f"Loading Adhoc adapter from: {adapter_path}")
            else:
                # 对于base模型，不加载adapter
                adapter_path = None
                load_as = None
                print(f"Using base model without adapter")

            # 如果有adapter路径，则加载并激活adapter
            if adapter_path:
                # 使用load_as参数指定adapter的标识符
                self.model.load_adapter(adapter_path, load_as=load_as, set_active=False)
                print(f"✅ Adapter loaded from '{adapter_path}' as '{load_as}'")
                
                # 明确激活adapter
                self.model.set_active_adapters(load_as)
                print(f"✅ Adapter '{load_as}' activated")
                
                # 验证adapter是否真正激活
                active_adapters = self.model.active_adapters
                print(f"🔍 Active adapters: {active_adapters}")
            
        else:
            if not ADAPTERS_AVAILABLE:
                print("⚠️  adapters library not available, loading base model only")
                print("   Install with: pip install adapters")
                base_model_path = self.model_name
            else:
                print("Loading SPECTER2 base model without adapter")
                base_model_path = "models/specter2/base"
            
            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            self.model = AutoModel.from_pretrained(base_model_path)

        # 移动到指定设备
        self.model = self.model.to(self.device)
        self.model.eval()

        # 获取嵌入维度
        self._embedding_dim = self.model.config.hidden_size


class COCO_DREncoder(BaseTransformerEncoder):
    """
    COCO-DR编码器，完全遵循BaseTransformerEncoder的逻辑

    COCO-DR是一个基于BERT的密集检索模型，不需要特殊处理，
    只需要在初始化时传入正确的模型名称即可。
    """

    def __init__(self,
                 model_name: str = "OpenMatch/cocodr-base-msmarco",
                 device: Optional[str] = None,
                 batch_size: int = 16,
                 max_length: int = 512,
                 pooling_strategy: str = "cls"):
        """
        初始化COCO-DR编码器

        Args:
            model_name: COCO-DR模型名称或路径
            device: 设备
            batch_size: 批处理大小
            max_length: 最大序列长度
            pooling_strategy: 池化策略
        """
        # 直接调用父类初始化，无需任何额外操作
        super().__init__(model_name, device, batch_size, max_length, pooling_strategy)


class SciBERTEncoder(BaseTransformerEncoder):
    """
    SciBERT编码器，继承自BaseTransformerEncoder

    SciBERT是专门为科学文本预训练的BERT模型，使用科学词汇表。
    它完全复用父类的编码和池化逻辑，无需任何特殊处理。
    """

    def __init__(self,
                 model_name: str = "allenai/scibert_scivocab_uncased",
                 device: Optional[str] = None,
                 batch_size: int = 16,
                 max_length: int = 512,
                 pooling_strategy: str = "cls"):
        """
        初始化SciBERT编码器

        Args:
            model_name: SciBERT模型名称或路径
                       常用模型:
                       - allenai/scibert_scivocab_uncased (默认)
                       - allenai/scibert_scivocab_cased
            device: 设备
            batch_size: 批处理大小
            max_length: 最大序列长度
            pooling_strategy: 池化策略
        """
        # 直接调用父类初始化，传入SciBERT的模型名称即可
        super().__init__(model_name, device, batch_size, max_length, pooling_strategy)


# CoF编码器现在在独立的cof模块中


# 导出所有编码器类
__all__ = [
    'BaseEncoder',
    'SentenceTransformerEncoder',
    'BaseTransformerEncoder',
    'SPECTER2Encoder',
    'COCO_DREncoder',
    'SciBERTEncoder'
]
