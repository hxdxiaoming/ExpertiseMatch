# TPMS Matcher

## 概述

TPMS (Text-based Paper Matching System) 是一个基于TF-IDF的文本相似度计算系统，主要用于计算审稿人与论文之间的相似度。该实现基于Xu等人2019年的IJCAI论文。

## 特性

- **多模式支持**: 支持三种信息模式
  - `title`: 仅使用标题
  - `title+abstract`: 使用标题和摘要
  - `full`: 使用标题、摘要和全文
- **任务类型支持**: 支持两种任务类型
  - `paper-centric`: 论文作为query，审稿人档案作为待匹配项
  - `reviewer-centric`: 审稿人档案作为query，论文作为待匹配项
- **智能文本预处理**: 
  - 停用词过滤
  - 词干提取 (Porter Stemmer)
  - 特殊字符处理
  - 重音符号标准化
- **TF-IDF相似度计算**: 使用平滑TF和IDF加权
- **分数归一化**: 支持多种归一化策略
- **缓存机制**: 缓存档案和IDF值以提高性能
- **数据集类型支持**: 自动检测open/close数据集类型

## 安装依赖

```bash
pip install nltk pandas numpy
```

## 使用方法

### 1. 基本使用

```python
from matchers.tpms.tpms_matcher import TPMSTwoStageMatcher

# 创建matcher
config = {
    'regime': 'title+abstract',  # 使用标题和摘要
    'normalize_scores': True,     # 启用分数归一化
    'cache_profiles': True        # 启用档案缓存
}

matcher = TPMSTwoStageMatcher(config)

# 训练matcher
matcher.fit(reviewers_df, papers_df, metadata)

# 计算分数
scores_df = matcher.score(papers_df, reviewers_df)

# 预测top-k审稿人
predictions = matcher.predict(papers_df, reviewers_df, top_k=5)
```

### 2. 通过run_model_wrapper使用

创建配置文件 `configs/tpms_config.json`:

```json
{
  "matcher_class": "TPMSTwoStageMatcher",
  "matcher_config": {
    "regime": "title+abstract",
    "normalize_scores": true,
    "cache_profiles": true
  },
  "description": "TPMS matcher with title+abstract regime",
  "experiment_name": "tpms_experiment"
}
```

运行实验:

```bash
python run_model_wrapper.py --dataset_name your_dataset --config_file configs/tpms_config.json --mode generate
```

### 3. 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `regime` | str | `'title+abstract'` | 信息模式: `title`, `title+abstract`, `full` |
| `normalize_scores` | bool | `True` | 是否归一化分数 |
| `cache_profiles` | bool | `True` | 是否缓存档案 |

**注意**: 
- 数据集类型（open/close）会自动从数据集的`meta.json`文件中的`qrel_format`字段检测，无需在配置中指定。
- 任务类型（paper-centric/reviewer-centric）会自动从数据集的`meta.json`文件中的`task_type`字段检测，无需在配置中指定。

## 算法原理

### 1. 文本预处理

1. **分词**: 使用正则表达式分割文本，只保留字母字符
2. **清理**: 去除重音符号和特殊字符
3. **过滤**: 移除停用词和单字符词汇
4. **词干提取**: 使用Porter Stemmer进行词干提取

### 2. 档案构建

- **审稿人档案**: 基于其论文集合构建
- **论文档案**: 根据选择的模式组合标题、摘要和全文
- **词频统计**: 使用Counter统计词频

### 3. 任务类型处理

- **Paper-centric**: 论文作为query，审稿人档案作为待匹配项
  - 计算每篇论文与所有审稿人的相似度
  - 输出矩阵：行=论文，列=审稿人
  
- **Reviewer-centric**: 审稿人档案作为query，论文作为待匹配项
  - 计算每个审稿人与所有论文的相似度
  - 输出矩阵：行=审稿人，列=论文

使用改进的TF-IDF余弦相似度:

```
TF = 0.5 + 0.5 * (词频 / 最大词频)
IDF = log(总文档数 / 包含该词的文档数)
相似度 = Σ(TF1 × TF2 × IDF²) / (√Σ(TF1 × IDF)² × √Σ(TF2 × IDF)²)
```

## 性能优化

- **IDF缓存**: 预计算并缓存IDF值
- **档案缓存**: 缓存构建的档案避免重复计算
- **批处理**: 支持批量计算多个论文-审稿人对的相似度

## 测试

运行测试脚本验证功能:

```bash
# 基本功能测试
python test_tpms_matcher.py

# Reviewer-centric功能测试
python test_tpms_reviewer_centric.py
```

## 注意事项

1. **数据格式**: 确保输入数据包含必要的字段（title, abstract, text等）
2. **内存使用**: 全文模式可能消耗较多内存
3. **性能**: 大规模数据集建议使用title或title+abstract模式
4. **依赖**: 需要安装NLTK的PorterStemmer

## 故障排除

### 常见问题

1. **ImportError**: 确保已安装所有依赖包
2. **ValueError**: 检查数据格式和必需字段
3. **MemoryError**: 考虑使用较少的信息模式或减少数据规模

### 调试模式

启用详细日志输出:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 参考文献

Xu, Y., Zhao, H., Shi, X., and Shah, N. (2019). On strategyproof conference review. In IJCAI.
