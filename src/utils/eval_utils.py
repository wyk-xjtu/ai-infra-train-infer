"""
评估工具模块 — 标准化的文本评估指标

支持指标：
- Token F1: 基于token overlap的F1分数（移除标点、小写化后计算）
- BLEU-4: 基于n-gram的翻译/生成质量指标
- Exact Match: 精确匹配率

使用方式：
    from src.utils.eval_utils import EvalMetrics

    evaluator = EvalMetrics()
    scores = evaluator.compute_all(predictions, references)
    # {'token_f1': 0.75, 'bleu4': 0.32, 'exact_match': 0.1}
"""
import re
import math
from typing import List, Dict, Optional, Tuple
from collections import Counter


def normalize_text(text: str) -> str:
    """标准化文本用于评估

    处理步骤：
    1. 转小写
    2. 移除标点符号
    3. 合并多余空格

    参考SQuAD评估脚本的标准做法
    """
    text = text.lower()
    # 移除标点符号（保留字母、数字、空格、中文字符）
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    # 合并多余空格
    return ' '.join(text.split())


def tokenize_for_eval(text: str) -> List[str]:
    """将文本切分为token用于评估

    中英文混合处理：
    - 英文按空格分词
    - 中文按字符分词
    """
    normalized = normalize_text(text)
    tokens = []
    for word in normalized.split():
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in word)
        if has_chinese:
            tokens.extend(list(word))  # 中文逐字
        else:
            tokens.append(word)  # 英文整词
    return tokens


def compute_token_f1(prediction: str, reference: str) -> float:
    """计算Token-level F1分数

    标准做法（参考SQuAD）：
    1. 标准化文本（小写+移除标点）
    2. 切分为token
    3. 计算precision/recall/F1
    """
    pred_tokens = tokenize_for_eval(prediction)
    ref_tokens = tokenize_for_eval(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # 交集token数
    common = sum((pred_counter & ref_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_bleu4(prediction: str, reference: str) -> float:
    """计算BLEU-4分数（简化版，不依赖nltk）

    BLEU = BP * exp(sum(1/4 * log(precision_n)) for n=1..4)
    BP = min(1, exp(1 - ref_len/pred_len))  # brevity penalty
    """
    pred_tokens = tokenize_for_eval(prediction)
    ref_tokens = tokenize_for_eval(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # 计算n-gram precision
    precisions = []
    for n in range(1, 5):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        if not pred_ngrams:
            precisions.append(0.0)
            continue

        common = sum((pred_ngrams & ref_ngrams).values())
        precision = common / sum(pred_ngrams.values())
        precisions.append(precision)

    # 平滑处理：短文本中高阶n-gram可能为空，使用下限值避免返回0
    # 参考NLTK的smoothing方法(method1): 将0精度替换为epsilon
    epsilon = 1e-8
    precisions = [max(p, epsilon) for p in precisions]

    # 几何平均
    log_avg = sum(math.log(p) for p in precisions) / 4

    # Brevity Penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))

    return bp * math.exp(log_avg)


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """提取n-gram并计数"""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_exact_match(prediction: str, reference: str) -> float:
    """精确匹配（标准化后比较）"""
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


class EvalMetrics:
    """评估指标计算器 — 支持批量计算和统计"""

    def __init__(self):
        self.results: List[Dict[str, float]] = []

    def compute_single(self, prediction: str, reference: str) -> Dict[str, float]:
        """计算单个样本的所有指标"""
        scores = {
            'token_f1': compute_token_f1(prediction, reference),
            'bleu4': compute_bleu4(prediction, reference),
            'exact_match': compute_exact_match(prediction, reference),
        }
        self.results.append(scores)
        return scores

    def compute_batch(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """批量计算并返回平均分"""
        assert len(predictions) == len(references)

        batch_scores = [self.compute_single(p, r) for p, r in zip(predictions, references)]

        avg = {}
        for key in batch_scores[0]:
            avg[key] = sum(s[key] for s in batch_scores) / len(batch_scores)
        return avg

    def get_summary(self) -> Dict[str, float]:
        """获取所有已计算结果的汇总统计"""
        if not self.results:
            return {}
        avg = {}
        for key in self.results[0]:
            values = [r[key] for r in self.results]
            avg[f'{key}_mean'] = sum(values) / len(values)
            avg[f'{key}_max'] = max(values)
            avg[f'{key}_min'] = min(values)
        return avg

    def reset(self):
        self.results.clear()
