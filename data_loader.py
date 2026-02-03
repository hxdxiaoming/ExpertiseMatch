#!/usr/bin/env python3
"""
DataLoader 模块 - 统一的数据集加载和预处理接口

核心设计哲学：
- 抽象文件系统，隐藏所有文件路径和格式细节
- 标准化数据表示，返回一致的内存对象
- 数据扩充与预处理，动态构建审稿人文本简介
- 元数据供给，提供数据集的完整"身份证"
- 保障效率，通过内存缓存避免重复I/O
"""

import pandas as pd
import json
import os
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path


class DataLoader:
    """
    统一的数据集加载器，作为文件系统和模型算法之间的桥梁
    """
    
    def __init__(self, base_path: str = 'data/', use_cache: bool = True,
                 max_papers_per_reviewer: int = None,
                 paper_selection_method: str = "all",
                 embedding_model_name: str = None):
        """
        初始化DataLoader

        :param base_path: 数据集根目录路径
        :param use_cache: 是否启用内存缓存
        :param max_papers_per_reviewer: 每个审稿人最多使用多少篇论文构建画像
        :param paper_selection_method: 论文选择方法 ("all", "recent", "random", "embedding_based")
        :param embedding_model_name: 用于论文选择的embedding模型名称
        """
        self.base_path = Path(base_path)
        self.use_cache = use_cache
        self._cache: Dict[str, Tuple] = {}

        # 新增的画像控制参数
        self.max_papers_per_reviewer = max_papers_per_reviewer
        self.paper_selection_method = paper_selection_method
        self.embedding_model_name = embedding_model_name

        # 如果使用embedding方法，初始化embedding模型
        self.embedding_model = None
        self.embedding_cache = {}  # 缓存已计算的embeddings
        self.use_precomputed_embeddings = False  # 是否使用预计算的embeddings

        if paper_selection_method == "embedding_based" and embedding_model_name:
            # 首先尝试加载预计算的embeddings
            if self._try_load_precomputed_embeddings(embedding_model_name):
                self.use_precomputed_embeddings = True
                print(f"Using precomputed embeddings for {embedding_model_name}")
            else:
                self._initialize_embedding_model()

        # 验证基础路径存在
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base data path does not exist: {self.base_path}")
    
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], List[str], Dict]:
        """
        加载并处理一个完整的数据集。这是该类的唯一公共接口。

        :param dataset_name: 数据集名称 (例如 'NIPS', 'exHarmony', 'stelmakh')
        :return: 一个包含五个元素的元组:
                 1. papers_df: 包含所有论文信息的DataFrame
                    - Columns: ['paper_id', 'title', 'abstract']
                 2. reviewers_df: 包含所有审稿人信息（及构建好的profile）的DataFrame
                    - Columns: ['reviewer_id', 'authored_paper_ids', 'profile']
                 3. qrels_dict: 一个字典，key为评估类型(如 'raw', 'easy', 'hard'), 
                               value为对应的qrels DataFrame
                 4. all_reviewer_ids: 数据集中所有审稿人的ID列表
                 5. metadata: 从 meta.json 加载的完整元信息字典
        """
        
        # 第1步：检查缓存
        if self.use_cache and dataset_name in self._cache:
            print(f"Loading {dataset_name} from cache...")
            return self._cache[dataset_name]
        
        print(f"Loading dataset: {dataset_name}")
        
        # 第2步：读取元数据 (meta.json)
        metadata = self._load_metadata(dataset_name)
        
        # 第3步：加载核心数据文件
        papers_df = self._load_papers(dataset_name)
        reviewers_df = self._load_reviewers(dataset_name)
        
        # 第4步：根据元数据加载所有评估文件 (Qrels)
        qrels_dict = self._load_qrels(dataset_name, metadata)
        
        # 第5步：执行核心的数据扩充逻辑
        # 如果使用embedding_based方法，需要获取查询论文
        query_papers = None
        if self.paper_selection_method == "embedding_based" and qrels_dict:
            # 从qrels中提取所有需要评分的论文作为查询论文
            all_query_papers = set()
            for qrels_df in qrels_dict.values():
                all_query_papers.update(qrels_df['paper_id'].unique())
            query_papers = list(all_query_papers)
            print(f"  Using {len(query_papers)} query papers for embedding-based selection")

        reviewers_df, embedding_time = self._build_reviewer_profiles(reviewers_df, papers_df, query_papers)

        # 第6步：组装、缓存并返回
        all_reviewer_ids = reviewers_df['reviewer_id'].tolist()

        # 将embedding时间添加到metadata中，供后续使用
        metadata_with_timing = metadata.copy()
        metadata_with_timing['embedding_offline_time'] = embedding_time

        result = (papers_df, reviewers_df, qrels_dict, all_reviewer_ids, metadata_with_timing)
        
        if self.use_cache:
            self._cache[dataset_name] = result
        
        print(f"Successfully loaded {dataset_name}: "
              f"{len(papers_df)} papers, {len(reviewers_df)} reviewers, "
              f"{len(qrels_dict)} qrel types")
        
        return result
    
    def _load_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """加载数据集的元数据"""
        meta_path = self.base_path / dataset_name / f"{dataset_name}_meta.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return metadata
    
    def _find_data_file(self, dataset_name: str, file_pattern: str) -> Optional[Path]:
        """
        根据模式查找数据文件，例如 'papers' 或 'reviewers'
        """
        # {pattern} 会被替换成 'papers' 或 'reviewers'
        possible_filenames = [
            f"{dataset_name}_{file_pattern}.json",
            f"{dataset_name}_{file_pattern}_test.json"
        ]

        for filename in possible_filenames:
            file_path = self.base_path / dataset_name / filename
            if file_path.exists():
                return file_path
        return None

    def _load_papers(self, dataset_name: str) -> pd.DataFrame:
        """加载论文数据"""
        papers_file = self._find_data_file(dataset_name, "papers")
        if papers_file is None:
            raise FileNotFoundError(f"Papers file not found for dataset: {dataset_name}")

        papers_data = self._load_jsonl(papers_file)

        # 标准化列名
        papers_df = pd.DataFrame(papers_data)
        if 'paper' in papers_df.columns:
            papers_df = papers_df.rename(columns={'paper': 'paper_id'})

        # 确保必要的列存在
        required_columns = ['paper_id', 'title', 'abstract']
        for col in required_columns:
            if col not in papers_df.columns:
                raise ValueError(f"Missing required column '{col}' in papers data")

        return papers_df[required_columns]
    
    def _load_reviewers(self, dataset_name: str) -> pd.DataFrame:
        """加载审稿人数据"""
        reviewers_file = self._find_data_file(dataset_name, "reviewers")
        if reviewers_file is None:
            raise FileNotFoundError(f"Reviewers file not found for dataset: {dataset_name}")

        reviewers_data = self._load_jsonl(reviewers_file)

        # 标准化列名
        reviewers_df = pd.DataFrame(reviewers_data)
        if 'reviewer' in reviewers_df.columns:
            reviewers_df = reviewers_df.rename(columns={'reviewer': 'reviewer_id'})
        if 'papers' in reviewers_df.columns:
            reviewers_df = reviewers_df.rename(columns={'papers': 'authored_paper_ids'})

        # 确保必要的列存在
        required_columns = ['reviewer_id', 'authored_paper_ids']
        for col in required_columns:
            if col not in reviewers_df.columns:
                raise ValueError(f"Missing required column '{col}' in reviewers data")

        return reviewers_df
    
    def _load_qrels(self, dataset_name: str, metadata: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """根据元数据加载所有qrels文件，并将嵌套的score字典展开"""
        qrels_dict = {}
        qrels_profiles = metadata.get('qrels_profiles', {})

        # 检查任务类型，确定数据格式
        task_type = metadata.get('task_type', 'paper-centric')  # 默认为paper-centric

        for qrel_type, qrel_info in qrels_profiles.items():
            filename = qrel_info.get('file')
            if not filename:
                print(f"Warning: No 'file' key in qrels_profiles for type '{qrel_type}'")
                continue

            file_path = self.base_path / dataset_name / filename
            if not file_path.exists():
                print(f"Warning: Qrels file not found: {file_path}")
                continue

            qrels_data = self._load_jsonl(file_path)

            # 【核心优化】将JSONL的嵌套结构展开为扁平化的记录
            qrels_records = []
            for item in qrels_data:
                query_id = item.get('query_id')
                scores = item.get('score', {})
                if not query_id or not isinstance(scores, dict):
                    continue

                if task_type == 'reviewer-centric':
                    # reviewer-centric: query_id是reviewer_id，scores的key是paper_id
                    for paper_id, score in scores.items():
                        qrels_records.append({
                            'paper_id': paper_id,
                            'reviewer_id': query_id,
                            'relevance_score': score
                        })
                else:
                    # paper-centric: query_id是paper_id，scores的key是reviewer_id
                    for reviewer_id, score in scores.items():
                        qrels_records.append({
                            'paper_id': query_id,
                            'reviewer_id': reviewer_id,
                            'relevance_score': score
                        })

            qrels_dict[qrel_type] = pd.DataFrame(qrels_records)

        return qrels_dict

    def _initialize_embedding_model(self):
        """初始化embedding模型用于论文选择"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Initializing embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("Embedding model initialized successfully")
        except ImportError:
            print("Warning: sentence-transformers not available, falling back to 'all' method")
            self.paper_selection_method = "all"
            self.embedding_model = None
        except Exception as e:
            print(f"Warning: Failed to initialize embedding model {self.embedding_model_name}: {e}")
            print("Falling back to 'all' method")
            self.paper_selection_method = "all"
            self.embedding_model = None

    def _try_load_precomputed_embeddings(self, model_name: str) -> bool:
        """尝试加载预计算的embeddings"""
        try:
            import pickle
            from pathlib import Path

            cache_dir = Path("cache/embeddings")
            if not cache_dir.exists():
                return False

            # 查找匹配的缓存文件
            for cache_file in cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_result = pickle.load(f)

                    if (cached_result.get('model_name') == model_name and
                        'embeddings' in cached_result):
                        self.embedding_cache = cached_result['embeddings']
                        print(f"Loaded {len(self.embedding_cache)} precomputed embeddings")
                        return True
                except:
                    continue

            return False
        except Exception as e:
            print(f"Failed to load precomputed embeddings: {e}")
            return False

    def _select_papers_for_reviewer(self, authored_papers: List[str], papers_df: pd.DataFrame,
                                   query_papers: List[str] = None) -> List[str]:
        """
        为审稿人选择用于构建画像的论文

        :param authored_papers: 审稿人发表的所有论文ID列表
        :param papers_df: 论文DataFrame
        :param query_papers: 查询论文列表（用于embedding_based方法）
        :return: 选择的论文ID列表
        """
        if not self.max_papers_per_reviewer or len(authored_papers) <= self.max_papers_per_reviewer:
            return authored_papers

        if self.paper_selection_method == "all":
            return authored_papers[:self.max_papers_per_reviewer]

        elif self.paper_selection_method == "recent":
            # 假设论文ID包含时间信息，或者按ID排序选择最近的
            return authored_papers[-self.max_papers_per_reviewer:]

        elif self.paper_selection_method == "random":
            import random
            return random.sample(authored_papers, min(self.max_papers_per_reviewer, len(authored_papers)))

        elif self.paper_selection_method == "embedding_based" and self.embedding_model and query_papers:
            return self._select_papers_by_embedding(authored_papers, papers_df, query_papers)

        else:
            # 默认返回前N篇
            return authored_papers[:self.max_papers_per_reviewer]

    def _precompute_embeddings(self, papers_df: pd.DataFrame, query_papers: List[str],
                              all_reviewer_papers: set) -> Dict[str, Any]:
        """
        预计算所有需要的论文embeddings

        :param papers_df: 论文DataFrame
        :param query_papers: 查询论文ID列表
        :param all_reviewer_papers: 所有审稿人论文ID集合
        :return: 包含embeddings和计算时间的字典
        """
        import time
        import numpy as np

        start_time = time.time()

        # 检查是否所有需要的embeddings都已经在缓存中
        all_papers_to_encode = set(query_papers) | all_reviewer_papers
        missing_papers = [pid for pid in all_papers_to_encode if pid not in self.embedding_cache]

        if not missing_papers:
            # 所有embeddings都在缓存中，无需重新计算
            print("  All required embeddings found in cache, skipping computation...")
            embedding_time = 0.0  # 使用缓存，计算时间为0
            total_papers_encoded = len(all_papers_to_encode)
        else:
            # 需要计算部分或全部embeddings
            print("  Precomputing embeddings for embedding-based selection...")

            # 获取abstracts
            paper_abstracts = {}
            valid_papers = []

            for paper_id in all_papers_to_encode:
                paper_row = papers_df[papers_df['paper_id'] == paper_id]
                if not paper_row.empty:
                    abstract = paper_row.iloc[0]['abstract']
                    if isinstance(abstract, str) and abstract.strip():
                        paper_abstracts[paper_id] = abstract.strip()
                        valid_papers.append(paper_id)

            print(f"    Computing embeddings for {len(valid_papers)} papers...")

            # 批量计算embeddings
            if valid_papers:
                abstracts_list = [paper_abstracts[paper_id] for paper_id in valid_papers]
                embeddings_array = self.embedding_model.encode(abstracts_list, show_progress_bar=True)

                # 构建embedding字典
                for i, paper_id in enumerate(valid_papers):
                    self.embedding_cache[paper_id] = embeddings_array[i]

            embedding_time = time.time() - start_time
            total_papers_encoded = len(valid_papers)

        # 分离查询论文和审稿人论文的embeddings
        query_embeddings = []
        query_paper_ids = []

        for paper_id in query_papers:
            if paper_id in self.embedding_cache:
                query_embeddings.append(self.embedding_cache[paper_id])
                query_paper_ids.append(paper_id)

        if embedding_time > 0:
            print(f"    Embedding computation completed in {embedding_time:.2f} seconds")
        else:
            print(f"    Used cached embeddings for {total_papers_encoded} papers")

        return {
            'query_embeddings': np.array(query_embeddings) if query_embeddings else None,
            'query_paper_ids': query_paper_ids,
            'embedding_time': embedding_time,
            'total_papers_encoded': total_papers_encoded
        }

    def _select_papers_by_embedding_optimized(self, authored_papers: List[str],
                                            query_embeddings: np.ndarray) -> List[str]:
        """
        使用预计算的embeddings选择与查询论文最相关的审稿人论文

        :param authored_papers: 审稿人发表的论文ID列表
        :param query_embeddings: 预计算的查询论文embeddings
        :return: 选择的论文ID列表
        """
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            # 获取审稿人论文的embeddings
            reviewer_embeddings = []
            valid_papers = []

            for paper_id in authored_papers:
                if paper_id in self.embedding_cache:
                    reviewer_embeddings.append(self.embedding_cache[paper_id])
                    valid_papers.append(paper_id)

            if len(valid_papers) <= self.max_papers_per_reviewer:
                return valid_papers

            if not reviewer_embeddings or query_embeddings is None:
                return authored_papers[:self.max_papers_per_reviewer]

            # 计算相似度矩阵
            reviewer_embeddings = np.array(reviewer_embeddings)
            similarities = cosine_similarity(reviewer_embeddings, query_embeddings)

            # 每篇审稿人论文与所有查询论文的最大相似度
            max_similarities = np.max(similarities, axis=1)

            # 选择相似度最高的N篇论文
            top_indices = np.argsort(max_similarities)[-self.max_papers_per_reviewer:]
            selected_papers = [valid_papers[i] for i in top_indices]

            return selected_papers

        except Exception as e:
            print(f"Warning: Optimized embedding selection failed: {e}, falling back to first N papers")
            return authored_papers[:self.max_papers_per_reviewer]

    def _build_reviewer_profiles(self, reviewers_df: pd.DataFrame, papers_df: pd.DataFrame,
                                query_papers: List[str] = None) -> Tuple[pd.DataFrame, float]:
        """构建审稿人的文本简介 (支持论文选择策略)"""
        import time
        import numpy as np

        profile_start_time = time.time()
        print("Building reviewer profiles...")

        if self.max_papers_per_reviewer:
            print(f"  Using paper selection: method={self.paper_selection_method}, max_papers={self.max_papers_per_reviewer}")
            if self.embedding_model_name:
                print(f"  Embedding model: {self.embedding_model_name}")

        # 预计算embeddings（如果使用embedding方法）
        embedding_time = 0.0
        query_embeddings = None

        if (self.paper_selection_method == "embedding_based" and
            self.embedding_model and query_papers):

            # 收集所有审稿人论文ID
            all_reviewer_papers = set()
            for authored_papers in reviewers_df['authored_paper_ids']:
                all_reviewer_papers.update(authored_papers)

            # 预计算embeddings
            embedding_result = self._precompute_embeddings(papers_df, query_papers, all_reviewer_papers)
            query_embeddings = embedding_result['query_embeddings']
            embedding_time = embedding_result['embedding_time']

            print(f"  Precomputed embeddings for {embedding_result['total_papers_encoded']} papers")

        # 创建paper_id到文本内容的快速查找字典
        # 使用title和abstract拼接，提供更丰富的信息
        paper_texts = {}
        for _, row in papers_df.iterrows():
            paper_id = row['paper_id']
            abstract = row['abstract']
            title = row['title']

            # 收集有效的文本片段
            text_parts = []

            # 添加title（如果存在）
            if title and isinstance(title, str) and title.strip():
                text_parts.append(title.strip())

            # 添加abstract（如果存在）
            if abstract and isinstance(abstract, str) and abstract.strip():
                text_parts.append(abstract.strip())

            # 拼接所有文本片段
            paper_texts[paper_id] = " ".join(text_parts)



        def get_profile(authored_papers: List[str]) -> str:
            """构建审稿人画像的辅助函数"""
            # 应用论文选择策略
            if self.max_papers_per_reviewer and len(authored_papers) > self.max_papers_per_reviewer:
                if (self.paper_selection_method == "embedding_based" and
                    query_embeddings is not None):
                    # 使用优化的embedding选择
                    selected_papers = self._select_papers_by_embedding_optimized(authored_papers, query_embeddings)
                else:
                    # 使用其他选择方法
                    selected_papers = self._select_papers_for_reviewer(authored_papers, papers_df, query_papers)
            else:
                selected_papers = authored_papers

            # 构建画像文本
            texts = []
            for paper_id in selected_papers:
                text = paper_texts.get(paper_id)
                if text and isinstance(text, str):
                    texts.append(text.strip())

            return " ".join(texts)

        # 使用.apply()代替iterrows()
        reviewers_df = reviewers_df.copy()  # 避免SettingWithCopyWarning
        reviewers_df['profile'] = reviewers_df['authored_paper_ids'].apply(get_profile)

        profile_total_time = time.time() - profile_start_time

        # 统计信息
        if self.max_papers_per_reviewer:
            avg_papers_used = reviewers_df['authored_paper_ids'].apply(
                lambda papers: min(len(papers), self.max_papers_per_reviewer)
            ).mean()
            print(f"  Built profiles for {len(reviewers_df)} reviewers (avg {avg_papers_used:.1f} papers per reviewer)")
        else:
            print(f"  Built profiles for {len(reviewers_df)} reviewers (using all authored papers)")

        print(f"  Profile building completed in {profile_total_time:.2f} seconds")
        if embedding_time > 0:
            print(f"    - Embedding computation: {embedding_time:.2f} seconds")
            print(f"    - Profile construction: {profile_total_time - embedding_time:.2f} seconds")

        return reviewers_df, embedding_time
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载JSONL格式文件"""
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
                    continue
        
        return data
    
    def get_available_datasets(self) -> List[str]:
        """获取所有可用的数据集名称"""
        datasets = []

        for item in self.base_path.iterdir():
            if item.is_dir() and (item / f"{item.name}_meta.json").exists():
                datasets.append(item.name)

        return sorted(datasets)
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        print("Cache cleared")


# 使用示例
if __name__ == "__main__":
    # 创建DataLoader实例
    loader = DataLoader()
    
    # 查看可用数据集
    print("Available datasets:", loader.get_available_datasets())
    
    # 加载NIPS数据集
    papers_df, reviewers_df, qrels_dict, all_reviewer_ids, metadata = loader.load_dataset("NIPS")
    
    print(f"\nNIPS Dataset Summary:")
    print(f"Papers: {len(papers_df)}")
    print(f"Reviewers: {len(reviewers_df)}")
    print(f"Qrels types: {list(qrels_dict.keys())}")
    print(f"Metadata: {metadata['name']}, format: {metadata['qrel_format']}")
