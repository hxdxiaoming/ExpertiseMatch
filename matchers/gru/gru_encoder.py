#!/usr/bin/env python3
"""
GRUArxivEncoder - 使用论文仓库训练的双向GRU模型，对文本生成隐藏向量作为embedding。

依赖文件（默认路径，可通过环境变量覆盖）：
- 词表: autonomous-peer-review-platform-main/model/data/tokenizer/custom-wp-vocab-50k-vocab.txt
- 标签: autonomous-peer-review-platform-main/model/data/labels.json（可选，仅分类用）
- 检查点: autonomous-peer-review-platform-main/model/gru_model/checkpoints/checkpoint_epoch_*.pth

环境变量（可选）：
- GRU_ARXIV_MODEL_DIR: 指向autonomous-peer-review-platform-main/model 目录
- GRU_ARXIV_CHECKPOINT: 指定具体checkpoint路径
"""

import os
import glob
import numpy as np
from typing import List, Optional

import torch
import torch.nn as nn

from tokenizers import BertWordPieceTokenizer

from ..base import BaseEncoder


def _default_model_root() -> str:
    # 默认指向当前项目内的 models/gru 目录
    return os.environ.get(
        "GRU_ARXIV_MODEL_DIR",
        os.path.join("models", "gru")
    )


def _resolve_checkpoint(model_root: str) -> str:
    ck_env = os.environ.get("GRU_ARXIV_CHECKPOINT")
    if ck_env and os.path.exists(ck_env):
        return ck_env
    # 先优先 models/gru/model_checkpoint.pth
    default_ckpt = os.path.join(model_root, "model_checkpoint.pth")
    if os.path.exists(default_ckpt):
        return default_ckpt
    # 兼容老目录结构（若用户仍放在autonomous仓库下）
    ck_dir_legacy = os.path.join(model_root, "gru_model", "checkpoints")
    candidates = sorted(glob.glob(os.path.join(ck_dir_legacy, "checkpoint_epoch_*.pth")))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No checkpoint found. Expected {default_ckpt} or legacy {ck_dir_legacy}/checkpoint_epoch_*.pth"
    )


class _GRUModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers=2, bidirectional=True, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        _, hidden = self.gru(embedded)
        # concat last layer's two directions
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        logits = self.fc(hidden)
        return logits, hidden


class GRUArxivEncoder(BaseEncoder):
    """
    使用训练好的GRU模型输出隐藏态作为embedding。

    config 可选参数：
    - model_root: str 指向 autonomous-peer-review-platform-main/model
    - max_length: int 默认256
    - batch_size: int 默认64
    - device: 'cuda' 或 'cpu'，默认自动选择
    """

    def __init__(self, model_name: str = "gru_arxiv", **config):
        super().__init__(config)
        self.model_name = model_name
        self.model_root = self.config.get("model_root", _default_model_root())
        self.max_length = int(self.config.get("max_length", 256))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.device = self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        # 路径
        # 词表默认放在 models/gru/tokenizer 下
        self.vocab_file = os.path.join(self.model_root, "tokenizer", "custom-wp-vocab-50k-vocab.txt")
        if not os.path.exists(self.vocab_file):
            raise FileNotFoundError(f"Tokenizer vocab not found: {self.vocab_file}. Run tokenizer training first.")

        self.checkpoint_path = _resolve_checkpoint(self.model_root)

        # 初始化分词器
        self.tokenizer = BertWordPieceTokenizer.from_file(self.vocab_file)
        self.vocab_size = self.tokenizer.get_vocab_size()

        # 标签数量（输出维度）
        # 与训练脚本一致: labels.json 包含158个类别
        # 标签文件默认放在 models/gru/labels.json 下
        labels_file = os.path.join(self.model_root, "labels.json")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        import json
        with open(labels_file, 'r') as f:
            label_set = json.load(f)
        # 构造 index->label 映射
        self.num_classes = len(label_set)
        self.index_to_label = [None] * self.num_classes
        for label, idx in label_set.items():
            if 0 <= idx < self.num_classes:
                self.index_to_label[idx] = label

        # 模型
        self.model = _GRUModel(
            input_dim=self.vocab_size,
            embedding_dim=256,
            hidden_dim=256,
            output_dim=self.num_classes,
            n_layers=2,
            bidirectional=True,
            dropout=0.5,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 隐藏态维度
        self._embedding_dim = 512  # 256*2 for bidirectional

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        # 分词
        enc_list = self.tokenizer.encode_batch(texts)
        ids_list = []
        for enc in enc_list:
            ids = enc.ids[: self.max_length]
            if len(ids) < self.max_length:
                ids = ids + [0] * (self.max_length - len(ids))
            ids_list.append(ids)
        input_ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)

        with torch.no_grad():
            _, hidden = self.model(input_ids)
        return hidden.detach().cpu().numpy()

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dimension))
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embs = self._encode_batch(batch)
            all_embs.append(embs)
        return np.vstack(all_embs)

    def predict_topk_categories(self, texts: List[str], k: int = 3) -> List[List[str]]:
        """使用分类头输出top-k类别标签名称。"""
        if not texts:
            return []
        # 分词
        enc_list = self.tokenizer.encode_batch(texts)
        ids_list = []
        for enc in enc_list:
            ids = enc.ids[: self.max_length]
            if len(ids) < self.max_length:
                ids = ids + [0] * (self.max_length - len(ids))
            ids_list.append(ids)
        input_ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, _ = self.model(input_ids)
            logits = logits.detach().cpu().numpy()

        results: List[List[str]] = []
        for row in logits:
            topk_idx = np.argsort(row)[-k:][::-1]
            labels = [self.index_to_label[idx] for idx in topk_idx if 0 <= idx < len(self.index_to_label)]
            results.append(labels)
        return results

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dim


