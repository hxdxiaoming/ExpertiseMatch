#!/usr/bin/env python3
"""
offline_preprocessing.py - 离线预处理系统

负责为数据集预计算embeddings和BM25索引，避免在线实验时的重复计算。
支持多种embedding模型和缓存机制。
"""

import argparse
import json
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from data_loader import DataLoader
from matchers.embedding.encoders import SentenceTransformerEncoder, COCO_DREncoder, SPECTER2Encoder, SciBERTEncoder
from matchers.gru.gru_encoder import GRUArxivEncoder
# from matchers.cof import CoFEncoder  # CoF模块暂时不可用


class EmbeddingPreprocessor:
    """负责论文embedding的预计算和缓存"""

    def __init__(self, cache_dir: str = "cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = None
        
    def _get_cache_key(self, dataset_name: str, model_name: str, paper_ids: List[str]) -> str:
        """生成缓存键"""
        # 使用数据集名称、模型名称和论文ID列表的hash作为缓存键
        content = f"{dataset_name}_{model_name}_{sorted(paper_ids)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _detect_encoder_class(self, model_name: str):
        """检测应该使用哪个encoder类"""
        if 'cof' in model_name.lower():
            # CoF模块暂时不可用，使用SentenceTransformerEncoder作为替代
            print("⚠️  CoF模块暂时不可用，使用SentenceTransformerEncoder作为替代")
            return SentenceTransformerEncoder
        elif 'gru' in model_name.lower():
            # 支持 'gru' 或 'gru_arxiv' 等标识
            return GRUArxivEncoder
        elif 'coco-dr' in model_name.lower():
            return COCO_DREncoder
        elif 'specter2' in model_name.lower():
            return SPECTER2Encoder
        elif 'scibert' in model_name.lower():
            return SciBERTEncoder
        else:
            return SentenceTransformerEncoder

    def _load_encoder(self, model_name: str):
        """使用encoder类加载模型"""
        # 修复：每次都重新加载encoder，避免不同模型共享同一个encoder
        encoder_class = self._detect_encoder_class(model_name)
        print(f"Loading embedding model: {model_name} (using {encoder_class.__name__})")

        try:
            if encoder_class == SPECTER2Encoder:
                # SPECTER2需要特殊处理adapter
                # 使用本地的base模型和adapter文件
                base_model_path = "models/specter2/base"

                if 'proximity' in model_name.lower():
                    adapter_path = "models/specter2/adapters/proximity"
                    print(f"Using SPECTER2 with Proximity adapter: {adapter_path}")
                elif 'adhoc' in model_name.lower():
                    adapter_path = "models/specter2/adapters/adhoc"
                    print(f"Using SPECTER2 with Adhoc adapter: {adapter_path}")
                elif 'classification' in model_name.lower():
                    adapter_path = "models/specter2/adapters/classification"
                    print(f"Using SPECTER2 with Classification adapter: {adapter_path}")
                else:
                    # 对于base模型，不使用adapter
                    adapter_path = None
                    print(f"Using SPECTER2 base model without adapter")

                self.encoder = encoder_class(
                    model_name=base_model_path,
                    adapter_name=adapter_path,
                    batch_size=256  # 🚀 进一步优化：从64增加到256
                )
            # CoF模块暂时不可用，跳过处理
            # elif encoder_class == CoFEncoder:
            #     # CoF需要特殊处理，需要指定因子类型
            #     ...
            elif encoder_class == GRUArxivEncoder:
                # GRU编码器使用已训练的GRU模型隐藏态，batch_size在其内部控制
                self.encoder = encoder_class()
            else:
                # 其他encoder使用默认参数
                # Qwen3模型需要更小的batch size避免CUDA OOM
                if 'qwen3' in model_name.lower():
                    batch_size = 1  # 🚀 减小到1避免CUDA OOM
                    self.encoder = encoder_class(
                        model_name=model_name,
                        batch_size=batch_size,
                        trust_remote_code=True  # Qwen3需要这个参数
                    )
                elif encoder_class == SentenceTransformerEncoder:
                    batch_size = 512  # 🚀 进一步优化：从128增加到512
                    self.encoder = encoder_class(
                        model_name=model_name,
                        batch_size=batch_size
                    )
                else:
                    batch_size = 256  # 🚀 进一步优化：从64增加到256
                    self.encoder = encoder_class(
                        model_name=model_name,
                        batch_size=batch_size
                    )

            print("Encoder loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load encoder for model {model_name}: {e}")
    
    def precompute_embeddings(self, dataset_name: str, papers_df: pd.DataFrame, 
                            model_name: str, force_recompute: bool = False) -> Dict[str, Any]:
        """
        预计算数据集中所有论文的embeddings
        
        :param dataset_name: 数据集名称
        :param papers_df: 论文DataFrame
        :param model_name: embedding模型名称
        :param force_recompute: 是否强制重新计算
        :return: 包含embeddings和统计信息的字典
        """
        print(f"=== Preprocessing embeddings for {dataset_name} ===")
        print(f"Model: {model_name}")
        print(f"Papers: {len(papers_df)}")
        
        # 检查缓存
        paper_ids = papers_df['paper_id'].tolist()
        cache_key = self._get_cache_key(dataset_name, model_name, paper_ids)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists() and not force_recompute:
            print("Loading embeddings from cache...")
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
            print(f"Loaded {len(cached_result['embeddings'])} cached embeddings")
            return cached_result
        
        # 加载encoder
        self._load_encoder(model_name)
        
        # 准备文本数据（标题+摘要）
        print("Preparing title+abstract texts...")
        valid_papers = []
        texts = []

        for _, row in papers_df.iterrows():
            paper_id = row['paper_id']
            title = row.get('title', '')
            abstract = row.get('abstract', '')

            # 确保都是字符串类型
            if not isinstance(title, str):
                title = ""
            if not isinstance(abstract, str):
                abstract = ""

            # 拼接标题和摘要
            text_parts = []
            if title.strip():
                text_parts.append(title.strip())
            if abstract.strip():
                text_parts.append(abstract.strip())

            # 只要有标题或摘要之一就包含
            if text_parts:
                combined_text = ' '.join(text_parts)
                valid_papers.append(paper_id)
                texts.append(combined_text)
            else:
                # 如果标题和摘要都为空，使用占位符
                valid_papers.append(paper_id)
                texts.append("empty document")

        print(f"Valid papers with title/abstract: {len(valid_papers)}")
        
        # 计算embeddings
        print("Computing embeddings...")
        start_time = time.time()

        embeddings_array = self.encoder.encode(texts)
        embedding_time = time.time() - start_time

        # 构建结果
        embeddings_dict = {}
        for i, paper_id in enumerate(valid_papers):
            embeddings_dict[paper_id] = embeddings_array[i]
        
        result = {
            'dataset_name': dataset_name,
            'model_name': model_name,
            'embeddings': embeddings_dict,
            'embedding_time': embedding_time,
            'total_papers': len(papers_df),
            'valid_papers': len(valid_papers),
            'embedding_dim': embeddings_array.shape[1] if len(embeddings_array) > 0 else 0,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存到缓存
        print(f"Saving embeddings to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"Embedding computation completed in {embedding_time:.2f} seconds")
        print(f"Average time per paper: {embedding_time/len(valid_papers)*1000:.2f} ms")
        
        return result




class BM25Preprocessor:
    """负责BM25索引的预计算和缓存"""
    
    def __init__(self, cache_dir: str = "cache/bm25"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, dataset_name: str, text_data: List[str], 
                      k1: float, b: float) -> str:
        """生成BM25缓存键"""
        # 使用文本数据的hash和BM25参数作为缓存键
        text_hash = hashlib.md5(str(sorted(text_data)).encode()).hexdigest()
        content = f"{dataset_name}_{text_hash}_k1{k1}_b{b}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def precompute_bm25_index(self, dataset_name: str, reviewers_df: pd.DataFrame,
                            k1: float = 1.2, b: float = 0.75, 
                            force_recompute: bool = False) -> Dict[str, Any]:
        """
        预计算审稿人的BM25索引
        
        :param dataset_name: 数据集名称
        :param reviewers_df: 审稿人DataFrame
        :param k1: BM25参数k1
        :param b: BM25参数b
        :param force_recompute: 是否强制重新计算
        :return: 包含BM25索引和统计信息的字典
        """
        print(f"=== Preprocessing BM25 index for {dataset_name} ===")
        print(f"Reviewers: {len(reviewers_df)}")
        print(f"BM25 parameters: k1={k1}, b={b}")
        
        # 准备文本数据
        profiles = reviewers_df['profile'].tolist()
        reviewer_ids = reviewers_df['reviewer_id'].tolist()
        
        # 检查缓存
        cache_key = self._get_cache_key(dataset_name, profiles, k1, b)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists() and not force_recompute:
            print("Loading BM25 index from cache...")
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
            print("BM25 index loaded from cache")
            return cached_result
        
        # 构建BM25索引
        print("Building BM25 index...")
        start_time = time.time()
        
        from rank_bm25 import BM25Okapi
        
        # 分词处理
        tokenized_profiles = []
        for profile in profiles:
            tokens = profile.split() if isinstance(profile, str) else []
            # 确保每个文档至少有一个token，避免BM25的除零错误
            if not tokens:
                tokens = ['empty']  # 为空文档添加占位符token
            tokenized_profiles.append(tokens)

        # 构建BM25索引
        bm25_index = BM25Okapi(tokenized_profiles, k1=k1, b=b)
        
        index_time = time.time() - start_time
        
        # 构建结果
        result = {
            'dataset_name': dataset_name,
            'bm25_index': bm25_index,
            'reviewer_ids': reviewer_ids,
            'k1': k1,
            'b': b,
            'index_time': index_time,
            'total_reviewers': len(reviewers_df),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存到缓存
        print(f"Saving BM25 index to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"BM25 index building completed in {index_time:.2f} seconds")
        
        return result


class PreprocessingManager:
    """统一管理离线预处理流程"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.embedding_preprocessor = EmbeddingPreprocessor(cache_dir + "/embeddings")
        self.bm25_preprocessor = BM25Preprocessor(cache_dir + "/bm25")
    
    def _get_model_path(self, model_name: str) -> str:
        """将简短的模型名称映射到实际的本地路径"""
        model_mapping = {
            'specter': 'models/sentence-transformers/specter',
            'scincl': 'models/sentence-transformers/scincl',
            'cocodr': 'models/coco-dr/base-msmarco',
            'coco-dr': 'models/coco-dr/base-msmarco',  # 修复：添加带连字符的版本
            'specter2base': 'specter2base',  # 特殊标识，在_load_encoder中处理
            'specter2proximity': 'specter2proximity',  # 特殊标识，在_load_encoder中处理
            'specter2adhoc': 'specter2adhoc',  # 特殊标识，在_load_encoder中处理
            'specter2classification': 'specter2classification',  # 特殊标识，在_load_encoder中处理
            'scibert': 'models/scibert',
            'qwen3': 'models/sentence-transformers/qwen3-0.6B',  # 修复：添加简短版本
            'qwen3-0.6b': 'models/sentence-transformers/qwen3-0.6B',
        }

        if model_name in model_mapping:
            return model_mapping[model_name]
        else:
            # 如果不在映射中，假设是完整路径
            return model_name

    def preprocess_dataset(self, dataset_name: str,
                          embedding_models: List[str] = None,
                          bm25_params: Dict[str, float] = None,
                          force_recompute: bool = False) -> Dict[str, Any]:
        """
        对数据集进行完整的离线预处理

        :param dataset_name: 数据集名称
        :param embedding_models: 要预计算的embedding模型列表
        :param bm25_params: BM25参数字典
        :param force_recompute: 是否强制重新计算
        :return: 预处理结果统计
        """
        print("=" * 80)
        print(f"OFFLINE PREPROCESSING: {dataset_name}")
        print("=" * 80)

        total_start_time = time.time()
        results = {
            'dataset_name': dataset_name,
            'embeddings': {},
            'bm25': {},
            'total_time': 0,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 加载数据集
        print("Loading dataset...")
        loader = DataLoader()
        papers_df, reviewers_df, qrels_dict, all_reviewer_ids, metadata = loader.load_dataset(dataset_name)

        # 预处理embeddings
        if embedding_models:
            for model_name in embedding_models:
                try:
                    # 特殊处理CoF模型 - 需要为三个因子分别生成embeddings
                    if 'cof' in model_name.lower():
                        print(f"Processing CoF model: {model_name}")
                        cof_factors = ['semantic', 'topic', 'citation']

                        for factor in cof_factors:
                            factor_model_name = f"cof_{factor}"
                            print(f"Processing CoF factor: {factor}")

                            embedding_result = self.embedding_preprocessor.precompute_embeddings(
                                dataset_name, papers_df, factor_model_name, force_recompute
                            )
                            results['embeddings'][factor_model_name] = {
                                'embedding_time': embedding_result['embedding_time'],
                                'total_papers': embedding_result['total_papers'],
                                'valid_papers': embedding_result['valid_papers'],
                                'embedding_dim': embedding_result['embedding_dim']
                            }
                    else:
                        # 将模型名称映射到实际路径
                        actual_model_path = self._get_model_path(model_name)
                        print(f"Processing model: {model_name} -> {actual_model_path}")

                        embedding_result = self.embedding_preprocessor.precompute_embeddings(
                            dataset_name, papers_df, actual_model_path, force_recompute
                        )
                        results['embeddings'][model_name] = {
                            'embedding_time': embedding_result['embedding_time'],
                            'total_papers': embedding_result['total_papers'],
                            'valid_papers': embedding_result['valid_papers'],
                            'embedding_dim': embedding_result['embedding_dim']
                        }
                except Exception as e:
                    print(f"Error preprocessing embeddings with {model_name}: {e}")
                    results['embeddings'][model_name] = {'error': str(e)}
        
        # 预处理BM25索引
        if bm25_params:
            try:
                bm25_result = self.bm25_preprocessor.precompute_bm25_index(
                    dataset_name, reviewers_df, 
                    bm25_params.get('k1', 1.2), 
                    bm25_params.get('b', 0.75),
                    force_recompute
                )
                results['bm25'] = {
                    'index_time': bm25_result['index_time'],
                    'total_reviewers': bm25_result['total_reviewers'],
                    'k1': bm25_result['k1'],
                    'b': bm25_result['b']
                }
            except Exception as e:
                print(f"Error preprocessing BM25 index: {e}")
                results['bm25'] = {'error': str(e)}
        
        results['total_time'] = time.time() - total_start_time
        
        print("=" * 80)
        print("PREPROCESSING COMPLETED")
        print("=" * 80)
        print(f"Total time: {results['total_time']:.2f} seconds")
        
        return results


def save_structured_results(results: Dict[str, Any], dataset_name: str,
                          output_base_dir: str = "preprocessing_results"):
    """
    将预处理结果保存到结构化的目录中

    :param results: 预处理结果字典
    :param dataset_name: 数据集名称
    :param output_base_dir: 输出基础目录
    """
    base_path = Path(output_base_dir)
    saved_files = []

    # 保存BM25结果
    if results.get('bm25'):
        bm25_dir = base_path / "bm25"
        bm25_dir.mkdir(parents=True, exist_ok=True)

        bm25_result = {
            'dataset_name': dataset_name,
            'method': 'BM25',
            'results': results['bm25'],
            'timestamp': results['timestamp']
        }

        bm25_file = bm25_dir / f"{dataset_name}_bm25.json"
        with open(bm25_file, 'w') as f:
            json.dump(bm25_result, f, indent=2, default=str)
        saved_files.append(str(bm25_file))
        print(f"BM25 results saved to: {bm25_file}")

    # 保存Embedding结果
    if results.get('embeddings'):
        embeddings_base_dir = base_path / "embeddings"

        for model_name, embedding_result in results['embeddings'].items():
            # 提取模型名称的最后部分作为目录名
            model_dir_name = model_name.split('/')[-1] if '/' in model_name else model_name
            model_dir = embeddings_base_dir / model_dir_name
            model_dir.mkdir(parents=True, exist_ok=True)

            embedding_result_full = {
                'dataset_name': dataset_name,
                'method': 'Embedding',
                'model_name': model_name,
                'results': embedding_result,
                'timestamp': results['timestamp']
            }

            embedding_file = model_dir / f"{dataset_name}_{model_dir_name}.json"
            with open(embedding_file, 'w') as f:
                json.dump(embedding_result_full, f, indent=2, default=str)
            saved_files.append(str(embedding_file))
            print(f"Embedding results saved to: {embedding_file}")

    # 保存完整结果到summary目录
    summary_dir = base_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    full_result_file = summary_dir / f"{dataset_name}_full.json"
    with open(full_result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    saved_files.append(str(full_result_file))
    print(f"Full results saved to: {full_result_file}")

    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Offline preprocessing for paper-reviewer matching")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name to preprocess")
    parser.add_argument("--embedding_models", type=str, nargs='+',
                       help="List of embedding models to precompute")
    parser.add_argument("--bm25_k1", type=float, default=1.2,
                       help="BM25 k1 parameter")
    parser.add_argument("--bm25_b", type=float, default=0.75,
                       help="BM25 b parameter")
    parser.add_argument("--force_recompute", action="store_true",
                       help="Force recomputation even if cache exists")
    parser.add_argument("--output_dir", type=str, default="preprocessing_results",
                       help="Output base directory for structured results")
    parser.add_argument("--legacy_output", type=str,
                       help="Legacy single file output (optional)")

    args = parser.parse_args()

    # 设置预处理参数
    bm25_params = {'k1': args.bm25_k1, 'b': args.bm25_b} if not args.embedding_models else None

    # 执行预处理
    manager = PreprocessingManager()
    results = manager.preprocess_dataset(
        args.dataset,
        embedding_models=args.embedding_models,
        bm25_params=bm25_params,
        force_recompute=args.force_recompute
    )

    # 保存结构化结果
    saved_files = save_structured_results(results, args.dataset, args.output_dir)

    # 如果指定了legacy输出，也保存到单个文件
    if args.legacy_output:
        with open(args.legacy_output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        saved_files.append(args.legacy_output)
        print(f"Legacy results saved to: {args.legacy_output}")

    print("=" * 60)
    print("STRUCTURED PREPROCESSING COMPLETED")
    print("=" * 60)
    print(f"Total files saved: {len(saved_files)}")
    for file_path in saved_files:
        print(f"  - {file_path}")
    print("=" * 60)


class EmbeddingTimeTracker:
    """专门记录各种模型在各个数据集上的embedding时间"""

    def __init__(self, output_dir: str = "preprocessing_results/embedding_times"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_embedding_time(self, dataset_name: str, model_name: str,
                            embedding_time: float, total_papers: int,
                            valid_papers: int, embedding_dim: int):
        """记录embedding时间到专门的文件"""

        # 创建记录
        record = {
            'dataset_name': dataset_name,
            'model_name': model_name,
            'embedding_time_seconds': embedding_time,
            'total_papers': total_papers,
            'valid_papers': valid_papers,
            'embedding_dim': embedding_dim,
            'avg_time_per_paper_ms': (embedding_time / valid_papers * 1000) if valid_papers > 0 else 0,
            'papers_per_second': valid_papers / embedding_time if embedding_time > 0 else 0,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'date': time.strftime("%Y-%m-%d")
        }

        # 保存到模型专用文件
        model_safe_name = model_name.replace('/', '_').replace('-', '_')
        model_file = self.output_dir / f"{model_safe_name}_times.json"

        # 读取现有记录
        existing_records = []
        if model_file.exists():
            try:
                with open(model_file, 'r') as f:
                    existing_records = json.load(f)
            except:
                existing_records = []

        # 添加新记录
        existing_records.append(record)

        # 保存更新后的记录
        with open(model_file, 'w') as f:
            json.dump(existing_records, f, indent=2)

        # 也保存到总的时间记录文件
        all_times_file = self.output_dir / "all_embedding_times.json"
        all_records = []
        if all_times_file.exists():
            try:
                with open(all_times_file, 'r') as f:
                    all_records = json.load(f)
            except:
                all_records = []

        all_records.append(record)
        with open(all_times_file, 'w') as f:
            json.dump(all_records, f, indent=2)

        print(f"📊 Embedding time recorded:")
        print(f"   Model: {model_name}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Time: {embedding_time:.2f}s ({embedding_time/60:.1f}min)")
        print(f"   Papers: {valid_papers}")
        print(f"   Speed: {valid_papers/embedding_time:.1f} papers/sec")
        print(f"   Saved to: {model_file}")


def preprocess_qwen3_for_wiz1000():
    """专门为wiz1000数据集预计算Qwen3 embedding"""
    print("🚀 Starting Qwen3 embedding preprocessing for wiz1000 dataset")
    print("="*80)

    # 初始化
    manager = PreprocessingManager()
    time_tracker = EmbeddingTimeTracker()

    # 预处理wiz1000的Qwen3 embedding
    try:
        results = manager.preprocess_dataset(
            dataset_name="wiz1000",
            embedding_models=["qwen3-0.6b"],
            force_recompute=False  # 如果已有缓存就使用缓存
        )

        # 记录时间
        if 'qwen3-0.6b' in results['embeddings']:
            qwen3_result = results['embeddings']['qwen3-0.6b']
            if 'embedding_time' in qwen3_result:
                time_tracker.record_embedding_time(
                    dataset_name="wiz1000",
                    model_name="models/sentence-transformers/qwen3-0.6B",
                    embedding_time=qwen3_result['embedding_time'],
                    total_papers=qwen3_result['total_papers'],
                    valid_papers=qwen3_result['valid_papers'],
                    embedding_dim=qwen3_result['embedding_dim']
                )

        # 保存结构化结果
        save_structured_results(results, "wiz1000")

        print("✅ Qwen3 embedding preprocessing for wiz1000 completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during Qwen3 preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    # 检查是否是专门的wiz1000预处理调用
    if len(sys.argv) > 1 and sys.argv[1] == "--qwen3-wiz1000":
        success = preprocess_qwen3_for_wiz1000()
        sys.exit(0 if success else 1)
    else:
        main()
