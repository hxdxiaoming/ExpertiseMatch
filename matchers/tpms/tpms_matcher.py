#!/usr/bin/env python3
"""
TPMS Matcher - 基于文本的论文匹配系统

TPMS (Text-based Paper Matching System) 是一个基于TF-IDF的文本相似度计算系统，
主要用于计算审稿人与论文之间的相似度。

基于Xu等人2019年的IJCAI论文实现。
"""

import os
import json
import math
import re
import unicodedata
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer

from ..base import BaseMatcher, MatcherType


class TPMSTokenizer:
    """TPMS文本预处理和分词器"""
    
    def __init__(self):
        # 停用词列表
        self.stopwords = {
            'a', 'about', 'above', 'accordingly', 'across', 'after', 'afterwards', 'again',
            'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although',
            'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and',
            'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are',
            'around', 'as', 'aside', 'at', 'away', 'back', 'be', 'became', 'because',
            'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind',
            'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both',
            'bottom', 'briefly', 'but', 'by', 'call', 'came', 'can', 'cannot', 'cant',
            'certain', 'certainly', 'co', 'computer', 'con', 'could', 'couldnt', 'cry',
            'de', 'describe', 'detail', 'do', 'does', 'done', 'down', 'due', 'during',
            'each', 'edit', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty',
            'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything',
            'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire',
            'first', 'five', 'following', 'for', 'former', 'formerly', 'forty', 'found',
            'four', 'from', 'front', 'full', 'further', 'gave', 'get', 'gets', 'give',
            'given', 'giving', 'go', 'gone', 'got', 'had', 'hardly', 'has', 'hasnt', 'have',
            'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
            'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however',
            'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is',
            'it', 'its', 'itself', 'just', 'keep', 'kept', 'kg', 'knowledge', 'largely',
            'last', 'latter', 'latterly', 'least', 'less', 'like', 'ltd', 'made', 'mainly',
            'make', 'many', 'may', 'me', 'meanwhile', 'mg', 'might', 'mill', 'mine', 'ml',
            'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself',
            'name', 'namely', 'nearly', 'necessarily', 'neither', 'never', 'nevertheless',
            'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'normally', 'not',
            'noted', 'nothing', 'now', 'nowhere', 'obtain', 'obtained', 'of', 'off',
            'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others',
            'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'owing', 'own', 'part',
            'particularly', 'past', 'per', 'perhaps', 'please', 'poorly', 'possible',
            'possibly', 'potentially', 'predominantly', 'present', 'previously',
            'primarily', 'probably', 'prompt', 'promptly', 'put', 'quickly', 'quite',
            'rather', 're', 'readily', 'really', 'recently', 'refs', 'regarding',
            'regardless', 'relatively', 'respectively', 'resulted', 'resulting', 'results', 'rst',
            'said', 'same', 'second', 'see', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'serious',
            'several', 'shall', 'she', 'should', 'show', 'showed', 'shown', 'shows', 'side',
            'significantly', 'similar', 'similarly', 'since', 'sincere', 'six', 'sixty',
            'slightly', 'so', 'some', 'somehow', 'someone', 'something', 'sometime',
            'sometimes', 'somewhat', 'somewhere', 'soon', 'specifically', 'state', 'states',
            'still', 'strongly', 'substantially', 'successfully', 'such', 'sufficiently',
            'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them',
            'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore',
            'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this',
            'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to',
            'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un',
            'under', 'unless', 'until', 'up', 'upon', 'us', 'use', 'used', 'usefully',
            'usefulness', 'using', 'usually', 'various', 'very', 'via', 'was', 'we', 'well',
            'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
            'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which',
            'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'widely',
            'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours',
            'yourself', 'yourselves'
        }
        
        self.stemmer = PorterStemmer()
    
    def is_uninformative_word(self, word: str) -> bool:
        """判断词汇是否无信息"""
        if len(word) <= 1:
            return True
        return word.lower() in self.stopwords
    
    def sanitize(self, text: str) -> str:
        """清理文本，去除重音符号和特殊字符"""
        # 特殊字符映射
        char_map = {
            'æ': 'ae', 'ø': 'o', '¨': 'o', 'ß': 'ss', 'Ø': 'o',
            '\xef\xac\x80': 'ff', '\xef\xac\x81': 'fi', '\xef\xac\x82': 'fl'
        }
        
        # 替换特殊字符
        for char, replace_char in char_map.items():
            text = text.replace(char, replace_char)
        
        # 去除重音符号
        text = ''.join((c for c in unicodedata.normalize('NFD', text) 
                       if unicodedata.category(c) != 'Mn'))
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """分词处理"""
        if not text:
            return []
        
        # 清理文本
        text = self.sanitize(text)
        
        # 分割文本，只保留字母字符
        words = re.split(r'[^a-zA-Z]', text)
        words = [w for w in words if len(w) > 0]
        
        return words
    
    def text_to_bow(self, text: str) -> Counter:
        """将文本转换为词袋模型"""
        if not text:
            return Counter()
        
        words = [w.lower() for w in self.tokenize(text)]
        # 过滤无信息词汇
        words = [w for w in words if not self.is_uninformative_word(w)]
        # 词干提取
        words = [self.stemmer.stem(w) for w in words]
        
        return Counter(words)


class TPMSSimilarityCalculator:
    """TPMS相似度计算器"""
    
    def __init__(self):
        self.idf_cache = {}
    
    def compute_idf(self, profiles: List[Dict[str, Counter]]) -> Dict[str, float]:
        """计算逆文档频率"""
        if not profiles:
            return {}
        
        # 计算总文档数（每个Counter代表一个文档）
        total_docs = len(profiles)
        
        # 计算每个词在多少文档中出现（文档频次）
        word_doc_count = defaultdict(int)
        for profile in profiles:
            # Counter 的键集合即该文档出现过的唯一词
            for word in profile.keys():
                word_doc_count[word] += 1
        
        # 计算IDF
        idf = {}
        for word, doc_count in word_doc_count.items():
            idf[word] = math.log(total_docs / doc_count)
        
        return idf
    
    def compute_similarity(self, profile1: Counter, profile2: Counter, 
                          idf_dict: Dict[str, float]) -> float:
        """计算两个档案之间的相似度"""
        if not profile1 or not profile2:
            return 0.0
        
        # 获取最大词频用于归一化
        max_freq1 = max(profile1.values()) if profile1 else 1
        max_freq2 = max(profile2.values()) if profile2 else 1
        
        similarity = 0.0
        norm1, norm2 = 0.0, 0.0
        
        # 计算profile1的贡献
        for word, freq in profile1.items():
            tf1 = 0.5 + 0.5 * freq / max_freq1
            idf = idf_dict.get(word, 0.0)
            norm1 += (tf1 * idf) ** 2
            
            if word in profile2:
                tf2 = 0.5 + 0.5 * profile2[word] / max_freq2
                similarity += tf1 * tf2 * (idf ** 2)
        
        # 计算profile2的贡献
        for word, freq in profile2.items():
            tf2 = 0.5 + 0.5 * freq / max_freq2
            idf = idf_dict.get(word, 0.0)
            norm2 += (tf2 * idf) ** 2
        
        # 归一化
        norm1, norm2 = math.sqrt(norm1), math.sqrt(norm2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return similarity / (norm1 * norm2)


class TPMSPaperProfileBuilder:
    """论文档案构建器"""
    
    def __init__(self, tokenizer: TPMSTokenizer):
        self.tokenizer = tokenizer
    
    def build_paper_profile(self, paper_data: Dict[str, Any], 
                           regime: str = 'title+abstract') -> Counter:
        """构建论文档案"""
        texts = []
        text_counter = Counter()
        
        # 添加标题（确保是字符串）
        if 'title' in paper_data and paper_data['title'] is not None:
            title = str(paper_data['title']).strip()
            if title and title.lower() != 'nan':
                texts.append(title)
        
        # 根据模式添加摘要（确保是字符串）
        if regime in ['title+abstract', 'full'] and 'abstract' in paper_data and paper_data['abstract'] is not None:
            abstract = str(paper_data['abstract']).strip()
            if abstract and abstract.lower() != 'nan':
                texts.append(abstract)
        
        # 根据模式添加全文
        if regime == 'full' and 'text' in paper_data and paper_data['text'] is not None:
            if isinstance(paper_data['text'], dict):
                # 如果text是词频字典
                for word, count in paper_data['text'].items():
                    if isinstance(word, str) and isinstance(count, (int, float)):
                        stemmed_word = self.tokenizer.stemmer.stem(word)
                        text_counter[stemmed_word] += int(count)
            elif isinstance(paper_data['text'], str):
                # 如果text是字符串
                text = str(paper_data['text']).strip()
                if text and text.lower() != 'nan':
                    text_counter.update(self.tokenizer.text_to_bow(text))
        
        # 如果没有有效文本，返回空Counter
        if not texts and not text_counter:
            return Counter()
        
        # 合并所有文本
        if texts:
            content = self.tokenizer.text_to_bow(' '.join(texts))
        else:
            content = Counter()
        
        # 添加全文词频
        for word, count in text_counter.items():
            content[word] += count
        
        return content


class TPMSTwoStageMatcher(BaseMatcher):
    """TPMS两阶段匹配器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 默认配置
        self.default_config = {
            'regime': 'title+abstract',  # title, title+abstract, full
            'normalize_scores': True,
            'cache_profiles': True
        }
        
        # 先设置默认配置，然后更新用户配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 初始化组件
        self.tokenizer = TPMSTokenizer()
        self.profile_builder = TPMSPaperProfileBuilder(self.tokenizer)
        self.similarity_calculator = TPMSSimilarityCalculator()
        
        # 缓存
        self.reviewer_profiles = {}
        self.paper_profiles = {}
        self.idf_dict = {}
        self.is_fitted = False
        
        # 数据集类型相关
        self.dataset_type = None
        self.paper2reviewers = {}  # 存储论文-审稿人关系映射
        self.reviewer2papers = {}  # 存储审稿人-论文关系映射
    
    @property
    def matcher_type(self) -> MatcherType:
        return MatcherType.SELF_CONTAINED
    
    def fit(self, reviewers_df: pd.DataFrame, papers_df: pd.DataFrame, 
            metadata: Optional[Dict[str, Any]] = None) -> 'TPMSTwoStageMatcher':
        """训练匹配器（构建档案和计算IDF）"""
        print("🔧 Building TPMS profiles...")
        
        # 从metadata中获取数据集类型（自动从meta.json确定）
        self.dataset_type = metadata.get('qrel_format', 'closed') if metadata else 'closed'
        print(f"📋 Dataset type: {self.dataset_type} (auto-detected from meta.json)")
        
        # 构建论文-审稿人关系映射
        self._build_paper_reviewer_mapping(reviewers_df)
        
        # 构建审稿人档案
        self.reviewer_profiles = self._build_reviewer_profiles(reviewers_df, papers_df)
        
        # 构建论文档案
        self.paper_profiles = self._build_paper_profiles(papers_df)
        
        # 计算IDF
        print("📊 Computing IDF values...")
        all_profiles = list(self.reviewer_profiles.values()) + list(self.paper_profiles.values())
        self.idf_dict = self.similarity_calculator.compute_idf(all_profiles)
        
        self.is_fitted = True
        print(f"✅ TPMS matcher fitted with {len(self.reviewer_profiles)} reviewers and {len(self.paper_profiles)} papers")
        
        return self
    
    def _build_paper_reviewer_mapping(self, reviewers_df: pd.DataFrame):
        """构建论文-审稿人关系映射（用于open/close数据集区分）"""
        self.paper2reviewers = {}
        self.reviewer2papers = {}
        
        for _, reviewer_row in reviewers_df.iterrows():
            reviewer_id = reviewer_row['reviewer_id']
            authored_papers = reviewer_row.get('authored_paper_ids', [])
            
            self.reviewer2papers[reviewer_id] = authored_papers
            
            for paper_id in authored_papers:
                if paper_id not in self.paper2reviewers:
                    self.paper2reviewers[paper_id] = set()
                self.paper2reviewers[paper_id].add(reviewer_id)
        
        print(f"📊 Built paper-reviewer mapping: {len(self.paper2reviewers)} papers, {len(self.reviewer2papers)} reviewers")
    
    def _build_reviewer_profiles(self, reviewers_df: pd.DataFrame, papers_df: pd.DataFrame = None) -> Dict[str, Counter]:
        """构建审稿人档案"""
        profiles = {}
        
        for _, reviewer in reviewers_df.iterrows():
            reviewer_id = reviewer['reviewer_id']
            texts = []
            text_counter = Counter()
            
            # 方法1: 处理审稿人的论文（如果papers字段存在）
            if 'papers' in reviewer and isinstance(reviewer['papers'], list):
                for paper in reviewer['papers']:
                    if isinstance(paper, dict):
                        # 添加标题（确保是字符串）
                        if 'title' in paper and paper['title'] is not None:
                            title = str(paper['title']).strip()
                            if title and title.lower() != 'nan':
                                texts.append(title)
                        
                        # 根据模式添加摘要（确保是字符串）
                        if self.config['regime'] in ['title+abstract', 'full'] and 'abstract' in paper and paper['abstract'] is not None:
                            abstract = str(paper['abstract']).strip()
                            if abstract and abstract.lower() != 'nan':
                                texts.append(abstract)
                        
                        # 根据模式添加全文
                        if self.config['regime'] == 'full' and 'text' in paper and paper['text'] is not None:
                            if isinstance(paper['text'], dict):
                                for word, count in paper['text'].items():
                                    if isinstance(word, str) and isinstance(count, (int, float)):
                                        stemmed_word = self.tokenizer.stemmer.stem(word)
                                        text_counter[stemmed_word] += int(count)
                            elif isinstance(paper['text'], str):
                                text = str(paper['text']).strip()
                                if text and text.lower() != 'nan':
                                    text_counter.update(self.tokenizer.text_to_bow(text))
            
            # 方法2: 根据authored_paper_ids从papers_df获取论文数据
            elif 'authored_paper_ids' in reviewer and papers_df is not None:
                authored_paper_ids = reviewer.get('authored_paper_ids', [])
                for paper_id in authored_paper_ids:
                    # 在papers_df中查找对应的论文
                    paper_data = papers_df[papers_df['paper_id'] == paper_id]
                    if not paper_data.empty:
                        paper = paper_data.iloc[0]
                        
                        # 添加标题（确保是字符串）
                        if 'title' in paper and paper['title'] is not None:
                            title = str(paper['title']).strip()
                            if title and title.lower() != 'nan':
                                texts.append(title)
                        
                        # 根据模式添加摘要（确保是字符串）
                        if self.config['regime'] in ['title+abstract', 'full'] and 'abstract' in paper and paper['abstract'] is not None:
                            abstract = str(paper['abstract']).strip()
                            if abstract and abstract.lower() != 'nan':
                                texts.append(abstract)
            
            # 如果没有有效文本，创建空档案
            if not texts and not text_counter:
                profiles[reviewer_id] = Counter()
                continue
            
            # 构建档案
            if texts:
                profile = self.tokenizer.text_to_bow(' '.join(texts))
            else:
                profile = Counter()
            
            # 添加全文词频
            for word, count in text_counter.items():
                profile[word] += count
            
            profiles[reviewer_id] = profile
        
        return profiles
    
    def _build_paper_profiles(self, papers_df: pd.DataFrame) -> Dict[str, Counter]:
        """构建论文档案"""
        profiles = {}
        
        for _, paper in papers_df.iterrows():
            paper_id = paper['paper_id']
            profile = self.profile_builder.build_paper_profile(paper, self.config['regime'])
            profiles[paper_id] = profile
        
        return profiles
    
    def score(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame,
              metadata: Optional[Dict[str, Any]] = None, qrels_dict: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """根据任务类型计算匹配分数"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring. Call fit() first.")
        
        # 从metadata中获取任务类型
        task_type = metadata.get('task_type', 'paper-centric') if metadata else 'paper-centric'
        print(f"🎯 Computing TPMS similarities for {task_type} task...")
        print(f"📋 Dataset type: {self.dataset_type}")
        
        if task_type == 'reviewer-centric':
            return self._score_reviewer_centric(papers_df, reviewers_df, qrels_dict)
        else:
            return self._score_paper_centric(papers_df, reviewers_df, qrels_dict)
    
    def _get_candidate_reviewers_for_paper(self, paper_id: str, reviewers_df: pd.DataFrame, qrels_dict: Dict[str, pd.DataFrame] = None) -> set:
        """根据数据集类型获取论文的候选审稿人范围"""
        if self.dataset_type == "closed":
            # Closed: 考虑这个query在标注数据中有标注的审稿人
            if qrels_dict and 'raw' in qrels_dict:
                # 从qrels数据中获取该论文的标注审稿人
                qrels_df = qrels_dict['raw']
                paper_qrels = qrels_df[qrels_df['paper_id'] == paper_id]
                if not paper_qrels.empty:
                    candidate_reviewers = set(paper_qrels['reviewer_id'].tolist())
                    return candidate_reviewers
            
            # 如果无法从qrels获取，回退到所有审稿人
            print(f"⚠️ Warning: No qrels data found for paper {paper_id} in closed mode, using all reviewers")
            return set(reviewers_df['reviewer_id'].tolist())
        else:
            # Open: 考虑所有审稿人
            return set(reviewers_df['reviewer_id'].tolist())
    
    def _score_paper_centric(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame, qrels_dict: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Paper-centric: 论文作为query，审稿人档案作为待匹配项"""
        print("📝 Paper-centric mode: papers as queries, reviewers as candidates")
        
        # 创建结果DataFrame
        results = []
        
        for _, paper in papers_df.iterrows():
            paper_id = paper['paper_id']
            
            if paper_id not in self.paper_profiles:
                print(f"⚠️ Warning: Paper {paper_id} not found in fitted profiles")
                continue
            
            paper_profile = self.paper_profiles[paper_id]
            
            # 根据数据集类型确定候选审稿人范围
            candidate_reviewers = self._get_candidate_reviewers_for_paper(paper_id, reviewers_df, qrels_dict)
            
            for _, reviewer in reviewers_df.iterrows():
                reviewer_id = reviewer['reviewer_id']
                
                # 检查审稿人是否在候选范围内
                if reviewer_id not in candidate_reviewers:
                    continue
                
                if reviewer_id not in self.reviewer_profiles:
                    print(f"⚠️ Warning: Reviewer {reviewer_id} not found in fitted profiles")
                    continue
                
                reviewer_profile = self.reviewer_profiles[reviewer_id]
                
                # 计算相似度：审稿人档案 vs 论文档案
                similarity = self.similarity_calculator.compute_similarity(
                    reviewer_profile, paper_profile, self.idf_dict
                )
                
                results.append({
                    'paper_id': paper_id,
                    'reviewer_id': reviewer_id,
                    'score': similarity
                })
        
        # 转换为DataFrame
        scores_df = pd.DataFrame(results)
        
        if scores_df.empty:
            print("⚠️ Warning: No valid scores computed")
            return pd.DataFrame(columns=['paper_id', 'reviewer_id', 'score'])
        
        # 转换为宽格式：行=论文，列=审稿人
        scores_wide = scores_df.pivot(index='paper_id', columns='reviewer_id', values='score')
        
        # 填充缺失值
        scores_wide = scores_wide.fillna(0.0)
        
        # 分数归一化
        if self.config['normalize_scores']:
            scores_wide = self._normalize_scores(scores_wide)
        
        return scores_wide
    
    def _score_reviewer_centric(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame, qrels_dict: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Reviewer-centric: 审稿人档案作为query，论文作为待匹配项"""
        print("📝 Reviewer-centric mode: reviewers as queries, papers as candidates")
        
        # 创建结果DataFrame
        results = []
        
        for _, reviewer in reviewers_df.iterrows():
            reviewer_id = reviewer['reviewer_id']
            
            if reviewer_id not in self.reviewer_profiles:
                print(f"⚠️ Warning: Reviewer {reviewer_id} not found in fitted profiles")
                continue
            
            reviewer_profile = self.reviewer_profiles[reviewer_id]
            
            # 根据数据集类型确定候选论文范围
            candidate_papers = self._get_candidate_papers_for_reviewer(reviewer_id, papers_df, qrels_dict)
            
            for _, paper in papers_df.iterrows():
                paper_id = paper['paper_id']
                
                # 检查论文是否在候选范围内
                if paper_id not in candidate_papers:
                    continue
                
                if paper_id not in self.paper_profiles:
                    print(f"⚠️ Warning: Paper {paper_id} not found in fitted profiles")
                    continue
                
                paper_profile = self.paper_profiles[paper_id]
                
                # 计算相似度：审稿人档案 vs 论文档案（注意顺序与paper-centric一致）
                similarity = self.similarity_calculator.compute_similarity(
                    reviewer_profile, paper_profile, self.idf_dict
                )
                
                results.append({
                    'reviewer_id': reviewer_id,
                    'paper_id': paper_id,
                    'score': similarity
                })
        
        # 转换为DataFrame
        scores_df = pd.DataFrame(results)
        
        if scores_df.empty:
            print("⚠️ Warning: No valid scores computed")
            return pd.DataFrame(columns=['reviewer_id', 'paper_id', 'score'])
        
        # 转换为宽格式：行=审稿人，列=论文
        scores_wide = scores_df.pivot(index='reviewer_id', columns='paper_id', values='score')
        
        # 填充缺失值
        scores_wide = scores_wide.fillna(0.0)
        
        # 分数归一化
        if self.config['normalize_scores']:
            scores_wide = self._normalize_scores(scores_wide)
        
        return scores_wide
    
    def _get_candidate_papers_for_reviewer(self, reviewer_id: str, papers_df: pd.DataFrame, qrels_dict: Dict[str, pd.DataFrame] = None) -> set:
        """根据数据集类型获取审稿人的候选论文范围"""
        if self.dataset_type == "closed":
            # Closed: 考虑这个query在标注数据中有标注的论文
            if qrels_dict and 'raw' in qrels_dict:
                # 从qrels数据中获取该审稿人的标注论文
                qrels_df = qrels_dict['raw']
                reviewer_qrels = qrels_df[qrels_df['reviewer_id'] == reviewer_id]
                if not reviewer_qrels.empty:
                    candidate_papers = set(reviewer_qrels['paper_id'].tolist())
                    return candidate_papers

            # 如果无法从qrels获取，回退到所有论文
            print(f"⚠️ Warning: No qrels data found for reviewer {reviewer_id} in closed mode, using all papers")
            return set(papers_df['paper_id'].tolist())
        else:
            # Open: 考虑所有论文
            return set(papers_df['paper_id'].tolist())
    
    def _normalize_scores(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """归一化分数"""
        # 对每行（每篇论文）进行归一化
        for idx in scores_df.index:
            row_scores = scores_df.loc[idx]
            min_score, max_score = row_scores.min(), row_scores.max()
            
            if max_score > min_score:
                scores_df.loc[idx] = (row_scores - min_score) / (max_score - min_score)
            else:
                scores_df.loc[idx] = 0.5  # 如果所有分数相同，设为0.5
        
        return scores_df
    
    def predict(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame,
                top_k: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """根据任务类型进行预测"""
        scores_df = self.score(papers_df, reviewers_df, metadata)
        
        # 从metadata中获取任务类型
        task_type = metadata.get('task_type', 'paper-centric') if metadata else 'paper-centric'
        
        if task_type == 'reviewer-centric':
            return self._predict_reviewer_centric(scores_df, top_k)
        else:
            return self._predict_paper_centric(scores_df, top_k)

    # 对接统一入口：run_model_wrapper 会通过 hasattr(matcher, 'predict_reviewer_centric') 来判断是否走审稿人为查询的流程
    def predict_reviewer_centric(self, papers_df: pd.DataFrame, reviewers_df: pd.DataFrame,
                                 qrels_dict: Optional[Dict[str, pd.DataFrame]] = None,
                                 top_k: Optional[int] = None) -> Dict[str, List[Dict[str, float]]]:
        """
        Reviewer-centric 预测（公共方法供 run_model_wrapper 检测使用）。
        返回：{ reviewer_id: [{"id": paper_id, "score": float}, ...], ... }
        """
        # 确保已拟合
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # 直接调用内部 reviewer-centric 打分逻辑
        scores_df = self._score_reviewer_centric(papers_df, reviewers_df, qrels_dict)

        # 转换为期望的候选列表格式
        detailed_results: Dict[str, List[Dict[str, float]]] = {}
        for reviewer_id in scores_df.index:
            reviewer_scores = scores_df.loc[reviewer_id].sort_values(ascending=False)
            if top_k is not None:
                reviewer_scores = reviewer_scores.head(top_k)
            candidates = [{"id": str(paper_id), "score": float(score)} for paper_id, score in reviewer_scores.items()]
            detailed_results[str(reviewer_id)] = candidates

        return detailed_results
    
    def _predict_paper_centric(self, scores_df: pd.DataFrame, top_k: Optional[int] = None) -> pd.DataFrame:
        """Paper-centric预测：每篇论文的top-k审稿人推荐"""
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
    
    def _predict_reviewer_centric(self, scores_df: pd.DataFrame, top_k: Optional[int] = None) -> pd.DataFrame:
        """Reviewer-centric预测：每个审稿人的top-k论文推荐"""
        results = []
        for reviewer_id in scores_df.index:
            reviewer_scores = scores_df.loc[reviewer_id].sort_values(ascending=False)
            
            if top_k is not None:
                reviewer_scores = reviewer_scores.head(top_k)
            
            for paper_id, score in reviewer_scores.items():
                results.append({
                    'reviewer_id': reviewer_id,
                    'paper_id': paper_id,
                    'score': score
                })
        
        return pd.DataFrame(results)
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()
    
    def set_config(self, **kwargs) -> None:
        """更新配置参数"""
        self.config.update(kwargs)
