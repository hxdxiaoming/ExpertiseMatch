# evaluation.py

import json
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score, label_ranking_average_precision_score, roc_auc_score
from scipy.stats import kendalltau, spearmanr
from typing import Dict, List, Any, Tuple

class MetricsCalculator:
    """
    一个健壮的评估器，能够处理多种评估协议和标签类型，并计算全面的排序指标。
    """
    def __init__(self, qrels_df: pd.DataFrame, all_reviewer_ids: List[str], metadata: Dict[str, Any], qrel_type: str = None):
        """
        用 Ground Truth 和元数据初始化评估器。

        :param qrels_df: 包含 ['paper_id', 'reviewer_id', 'relevance_score'] 的DataFrame。
        :param all_reviewer_ids: 数据集中所有审稿人的ID列表。
        :param metadata: 从 meta.json 加载的元信息字典，用于决策。
        :param qrel_type: 当前使用的qrel类型（如'raw', 'easy', 'hard'），用于确定正确的标签类型
        """
        # 1. 预处理qrels，根据任务类型选择不同的索引方式
        self.task_type = metadata.get('task_type', 'paper-centric')

        if self.task_type == 'reviewer-centric':
            # reviewer-centric: 按审稿人分组，每个审稿人对应其标注的论文
            self.qrels_map = {rid: df.set_index('paper_id')['relevance_score']
                              for rid, df in qrels_df.groupby('reviewer_id')}
        else:
            # paper-centric: 按论文分组，每篇论文对应其标注的审稿人
            self.qrels_map = {pid: df.set_index('reviewer_id')['relevance_score']
                              for pid, df in qrels_df.groupby('paper_id')}

        self.all_reviewer_ids = all_reviewer_ids

        # 2. 从元数据中解析出评估协议和标签类型
        self.qrel_format = metadata.get('qrel_format', 'closed') # 默认为封闭世界

        # 根据当前使用的qrel_type确定标签类型
        qrels_profiles = metadata.get('qrels_profiles', {})
        self.label_type = "binary" # 默认值
        if qrels_profiles and qrel_type and qrel_type in qrels_profiles:
            self.label_type = qrels_profiles[qrel_type].get("label_type", "binary")
        elif qrels_profiles:
            # 如果没有指定qrel_type，使用第一个profile
            first_profile_key = list(qrels_profiles.keys())[0]
            self.label_type = qrels_profiles[first_profile_key].get("label_type", "binary")

        # 3. 为二元化指标设定一个阈值
        self.binarize_threshold = 1.0 # 默认将所有 >= 1 的分数视为相关

    def _prepare_data_for_eval(self, query_id: str, model_scores_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """为单个查询（论文或审稿人）对齐真实标签和预测分数，返回 y_true, y_pred。"""
        true_scores_series = self.qrels_map.get(query_id)
        if true_scores_series is None:
            return np.array([]), np.array([])

        if self.task_type == 'reviewer-centric':
            # reviewer-centric: query_id是审稿人ID，candidate_ids是论文ID
            candidate_ids = true_scores_series.index  # 该审稿人标注的论文ID列表
        else:
            # paper-centric: query_id是论文ID，candidate_ids是审稿人ID
            candidate_ids = self.all_reviewer_ids if self.qrel_format == 'open' else true_scores_series.index

        pred_scores = model_scores_series.reindex(candidate_ids, fill_value=0)
        true_scores = true_scores_series.reindex(candidate_ids, fill_value=0)

        return true_scores.values, pred_scores.values

    def calculate_all(self, scores_df: pd.DataFrame, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        主入口函数：输入模型预测分数，计算并返回所有适用的指标。
        
        :param scores_df: DataFrame, 对于reviewer-centric任务：index是reviewer_id, columns是paper_id；对于paper-centric任务：index是paper_id, columns是reviewer_id。
        :param k_values: 要评估的k值列表。
        :return: 包含所有平均指标的字典。
        """
        # 根据标签类型决定要计算的指标
        if self.label_type == 'graded':
            # graded (raw): 只计算排序质量指标，不计算二元分类指标
            metric_results: Dict[str, List] = {f'NDCG@{k}': [] for k in k_values}
            metric_results.update({'Kendalls_Tau': [], 'Spearman_Rho': []})
        else:
            # binary (soft/hard/authors/cite/simcite): 计算二元分类和排序指标，但不计算Kendall's Tau
            metric_results: Dict[str, List] = {f'P@{k}': [] for k in k_values}
            metric_results.update({f'Recall@{k}': [] for k in k_values})
            metric_results.update({f'NDCG@{k}': [] for k in k_values})
            metric_results.update({f'MRR@{k}': [] for k in k_values})
            metric_results.update({'MAP': [], 'AUC-ROC': []})

        if self.task_type == 'reviewer-centric':
            # reviewer-centric: 按审稿人遍历，每个审稿人对应一组候选论文
            for reviewer_id in self.qrels_map.keys():
                if reviewer_id not in scores_df.index:
                    continue  # 跳过没有预测分数的审稿人

                # 获取该审稿人对所有论文的预测分数
                model_scores_series = scores_df.loc[reviewer_id]
                y_true, y_pred = self._prepare_data_for_eval(reviewer_id, model_scores_series)

                if len(y_true) == 0:
                    continue

                self._calculate_metrics_for_query(y_true, y_pred, metric_results, k_values)
        else:
            # paper-centric: 按论文遍历，每篇论文对应一组候选审稿人
            for paper_id, model_scores_series in scores_df.iterrows():
                y_true, y_pred = self._prepare_data_for_eval(paper_id, model_scores_series)

                if len(y_true) == 0:
                    continue

                self._calculate_metrics_for_query(y_true, y_pred, metric_results, k_values)

        # 对所有查询的结果求平均，并过滤掉未计算的指标
        final_metrics = {
            metric: np.mean(values)
            for metric, values in metric_results.items() if values
        }

        return final_metrics

    def _calculate_metrics_for_query(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   metric_results: Dict[str, List], k_values: List[int]):
        """为单个查询计算指标"""
        # --- 根据标签类型计算相应的指标 ---
        if self.label_type == 'graded':
            # graded数据：只计算NDCG@k和Kendall's Tau

            # NDCG@k: 使用原始多级分数
            if len(y_true) > 1:
                for k in k_values:
                    try:
                        ndcg_val = ndcg_score([y_true], [y_pred], k=k)
                        metric_results[f'NDCG@{k}'].append(ndcg_val)
                    except ValueError:
                        # 如果出错，跳过这个查询的NDCG计算
                        pass

            # Kendall's Tau: 使用原始多级分数
            if len(y_true) > 1 and len(np.unique(y_true)) > 1:
                tau, _ = kendalltau(y_true, y_pred)
                if not np.isnan(tau):
                    metric_results['Kendalls_Tau'].append(tau)

            # Spearman's Rho: 使用原始多级分数，对并列值处理更好
            if len(y_true) > 1 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
                try:
                    rho, _ = spearmanr(y_true, y_pred)
                    if not np.isnan(rho):
                        metric_results['Spearman_Rho'].append(rho)
                except (ValueError, RuntimeWarning):
                    # 如果输入数组是常数或出现其他问题，跳过这个指标
                    pass

        else:
            # binary数据：计算二元分类指标和排序指标
            y_true_binary = (y_true >= self.binarize_threshold).astype(int)
            num_relevant = np.sum(y_true_binary)

            sorted_indices = np.argsort(y_pred)[::-1]
            y_true_sorted = y_true_binary[sorted_indices]

            if num_relevant > 0:
                # AUC-ROC: 只有当存在正负样本时才计算
                if len(np.unique(y_true_binary)) > 1:
                    metric_results['AUC-ROC'].append(roc_auc_score(y_true_binary, y_pred))

                # 计算MAP
                aps = []
                hits = 0
                for i, is_relevant in enumerate(y_true_sorted):
                    if is_relevant:
                        hits += 1
                        aps.append(hits / (i + 1))
                metric_results['MAP'].append(np.mean(aps) if aps else 0)

                # 计算P@k, Recall@k, MRR@k
                for k in k_values:
                    top_k_true = y_true_sorted[:k]
                    metric_results[f'P@{k}'].append(np.sum(top_k_true) / k)
                    metric_results[f'Recall@{k}'].append(np.sum(top_k_true) / num_relevant)

                    first_relevant_pos = np.where(top_k_true == 1)[0]
                    mrr_score = 1 / (first_relevant_pos[0] + 1) if len(first_relevant_pos) > 0 else 0
                    metric_results[f'MRR@{k}'].append(mrr_score)

            # NDCG@k: 对于binary数据也计算NDCG
            if len(y_true) > 1:
                for k in k_values:
                    try:
                        # 对于binary数据，使用二元化后的分数计算NDCG
                        ndcg_val = ndcg_score([y_true_binary], [y_pred], k=k)
                        metric_results[f'NDCG@{k}'].append(ndcg_val)
                    except ValueError:
                        pass

# --- 使用示例 ---
if __name__ == '__main__':
    # 这是一个模拟，展示如何使用这个类
    # 1. 准备模拟的 Ground Truth 和元数据
    qrels_data = [
        {'paper_id': 'p1', 'reviewer_id': 'r1', 'relevance_score': 3},
        {'paper_id': 'p1', 'reviewer_id': 'r3', 'relevance_score': 5},
    ]
    qrels_df = pd.DataFrame(qrels_data)
    all_revs = ['r1', 'r2', 'r3', 'r4']
    meta = {'name': 'mock_dataset', 'qrel_format': 'open', 'qrels_profiles': {'raw': {'label_type': 'graded'}}}

    # 2. 准备一个模拟的模型预测分数
    scores_data = {'p1': {'r1': 2.5, 'r2': 0.5, 'r3': 4.8, 'r4': 1.1}}
    scores_df = pd.DataFrame.from_dict(scores_data, orient='index')

    # 3. 初始化并调用评估器
    evaluator = MetricsCalculator(qrels_df, all_revs, meta)
    final_scores = evaluator.calculate_all(scores_df, k_values=[3, 4])

    print("Calculated Metrics:")
    print(json.dumps(final_scores, indent=4))