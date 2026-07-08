"""
GSM8K 规则奖励函数

用于GRPO训练的奖励信号:
- 答案正确性: 提取模型回答中的数字答案，与标准答案比对
- 格式正确性: 检查是否按指定格式输出（如 "答案是: xxx"）

评分规则:
- 答案正确: +1.0
- 答案错误但格式正确: -0.5
- 答案错误且格式错误: -1.0
"""
import re
from typing import List


def extract_answer(text: str) -> str:
    """从模型输出中提取最终答案

    支持格式:
    - "#### 42"
    - "答案是: 42" / "答案是 42"
    - "The answer is 42" / "The answer is: 42"
    - "\\boxed{42}"

    Returns:
        提取到的答案字符串（数字），未找到返回空字符串
    """
    if not text:
        return ""

    # 优先匹配 #### 格式 (GSM8K标准格式)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return _normalize_number(match.group(1))

    # 匹配 \boxed{...} 格式 (LaTeX)
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return _normalize_number(match.group(1))

    # 匹配中文格式: "答案是: 42" 或 "答案是42" 或 "答案为42"
    match = re.search(r"答案[是为][:：]?\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return _normalize_number(match.group(1))

    # 匹配英文格式: "The answer is 42" 或 "The answer is: 42"
    match = re.search(r"[Tt]he answer is[:：]?\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return _normalize_number(match.group(1))

    # 回退: 尝试提取文本中最后一个独立数字
    matches = re.findall(r"(-?[\d,]+\.?\d*)", text)
    if matches:
        return _normalize_number(matches[-1])

    return ""


def _normalize_number(num_str: str) -> str:
    """规范化数字字符串：去除逗号和多余空格"""
    num_str = num_str.strip().replace(",", "")
    # 尝试转换为数字再转回字符串，以规范化格式
    try:
        val = float(num_str)
        # 如果是整数则去掉小数点
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return num_str


def check_format(text: str) -> bool:
    """检查输出是否包含正确的推理格式（分步骤）

    检测标准:
    1. 包含分步推理标记（如步骤编号、换行分段等）
    2. 包含最终答案标记（#### 或 "答案是" 等）
    """
    if not text:
        return False

    # 检查是否有分步推理迹象
    has_steps = bool(
        re.search(r"(步骤|Step|第\s*\d+\s*步)", text, re.IGNORECASE)
        or re.search(r"\d+[.)]\s+\S", text)  # "1. xxx" 或 "1) xxx"
        or text.count("\n") >= 2  # 至少有多行推理
    )

    # 检查是否有明确的答案标记
    has_answer_marker = bool(
        re.search(r"####", text)
        or re.search(r"\\boxed\{", text)
        or re.search(r"答案[是为]", text)
        or re.search(r"[Tt]he answer is", text)
    )

    return has_steps or has_answer_marker


def compute_reward(response: str, ground_truth: str) -> float:
    """计算单个回复的奖励分数

    Args:
        response: 模型生成的回答
        ground_truth: 标准答案（数字字符串）

    Returns:
        reward: -1.0 到 +1.0 之间的分数
            +1.0: 答案正确
            -0.5: 答案错误但格式正确
            -1.0: 答案错误且格式错误
    """
    predicted = extract_answer(response)
    expected = _normalize_number(ground_truth)
    format_ok = check_format(response)

    # 比较答案
    if predicted and predicted == expected:
        return 1.0
    elif format_ok:
        return -0.5
    else:
        return -1.0


def compute_batch_rewards(responses: List[str], ground_truths: List[str]) -> List[float]:
    """批量计算奖励

    Args:
        responses: 模型生成的回答列表
        ground_truths: 标准答案列表

    Returns:
        rewards: 每个回答的奖励分数列表
    """
    assert len(responses) == len(ground_truths), (
        f"responses({len(responses)}) and ground_truths({len(ground_truths)}) must have same length"
    )
    return [compute_reward(r, gt) for r, gt in zip(responses, ground_truths)]


class GSM8KRewardFunction:
    """GSM8K奖励函数封装

    提供统计信息（准确率、平均奖励等）
    """

    def __init__(self):
        self.total_count = 0
        self.correct_count = 0
        self.format_correct_count = 0
        self._total_reward = 0.0

    def __call__(self, responses: List[str], ground_truths: List[str]) -> List[float]:
        """计算批量奖励并更新统计

        Args:
            responses: 模型回答列表
            ground_truths: 标准答案列表

        Returns:
            奖励分数列表
        """
        rewards = []
        for response, gt in zip(responses, ground_truths):
            predicted = extract_answer(response)
            expected = _normalize_number(gt)
            format_ok = check_format(response)

            self.total_count += 1

            if predicted and predicted == expected:
                reward = 1.0
                self.correct_count += 1
                self.format_correct_count += 1
            elif format_ok:
                reward = -0.5
                self.format_correct_count += 1
            else:
                reward = -1.0

            self._total_reward += reward
            rewards.append(reward)

        return rewards

    @property
    def accuracy(self) -> float:
        """答案准确率"""
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count

    @property
    def format_accuracy(self) -> float:
        """格式正确率"""
        if self.total_count == 0:
            return 0.0
        return self.format_correct_count / self.total_count

    @property
    def average_reward(self) -> float:
        """平均奖励"""
        if self.total_count == 0:
            return 0.0
        return self._total_reward / self.total_count

    def reset_stats(self):
        """重置统计"""
        self.total_count = 0
        self.correct_count = 0
        self.format_correct_count = 0
        self._total_reward = 0.0

    def __repr__(self) -> str:
        return (
            f"GSM8KRewardFunction(total={self.total_count}, "
            f"accuracy={self.accuracy:.2%}, "
            f"format_accuracy={self.format_accuracy:.2%}, "
            f"avg_reward={self.average_reward:.3f})"
        )
