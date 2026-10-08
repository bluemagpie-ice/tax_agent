"""
评测指标模块
============

提供一组纯函数指标，用于量化 Agent 的表现。所有函数不依赖任何外部库，
可独立单元测试，也便于在不同评测模式下复用。

指标：
- intent_accuracy     意图路由准确率（预测意图 == 期望意图）
- keyword_recall      单条答案的关键词召回率（命中期望关键词的比例）
- macro_keyword_recall 所有答案关键词召回率的均值
- refusal_detected    判断一条答案是否属于"拒答"（未编造）
- refusal_accuracy    拒答正确率（该拒答的拒答了，该回答的回答了）
"""

from __future__ import annotations

from typing import Iterable, Sequence


# 拒答标志词：出现任一即认为模型在"如实告知无法回答"
_REFUSAL_MARKERS = (
    "无法回答", "未检索到", "未找到", "没有检索", "不掌握", "无法提供",
    "抱歉", "很抱歉", "暂未", "暂不能", "不足以", "未收录", "不了解",
    "无法判断", "不能提供",
)


def intent_accuracy(predicted: Sequence[str], expected: Sequence[str]) -> float:
    """
    意图路由准确率。

    :param predicted: 预测意图列表（与 expected 一一对应）
    :param expected:  期望意图列表
    :return: 命中比例 [0, 1]
    """
    if not expected:
        return 0.0
    correct = sum(1 for p, e in zip(predicted, expected) if p == e)
    return correct / len(expected)


def keyword_recall(answer: str, expected_keywords: Sequence[str]) -> float:
    """
    单条答案的关键词召回率（子串匹配）。

    :param answer: Agent 生成的答案文本
    :param expected_keywords: 期望出现的关键词列表
    :return: 命中比例 [0, 1]；无期望关键词时返回 1.0（不因空集扣分）
    """
    if not expected_keywords:
        return 1.0
    if not answer:
        return 0.0
    hit = sum(1 for kw in expected_keywords if kw in answer)
    return hit / len(expected_keywords)


def macro_keyword_recall(
    answers: Sequence[str], expected_keywords_list: Sequence[Sequence[str]]
) -> float:
    """所有样本关键词召回率的均值。"""
    if not answers:
        return 0.0
    scores = [
        keyword_recall(ans, kws)
        for ans, kws in zip(answers, expected_keywords_list)
    ]
    return sum(scores) / len(scores)


def refusal_detected(answer: str) -> bool:
    """判断一条答案是否属于拒答（未编造内容）。"""
    if not answer:
        return True
    return any(marker in answer for marker in _REFUSAL_MARKERS)


def refusal_accuracy(
    answers: Sequence[str], must_refuse_flags: Sequence[bool]
) -> float:
    """
    拒答正确率。

    对每个样本：
    - must_refuse=True  且 answer 拒答  → 正确
    - must_refuse=False 且 answer 未拒答 → 正确
    其余算错误。

    :return: 正确比例 [0, 1]
    """
    if not answers:
        return 0.0
    correct = 0
    for ans, must_refuse in zip(answers, must_refuse_flags):
        refused = refusal_detected(ans)
        if must_refuse == refused:
            correct += 1
    return correct / len(answers)


def compute_summary(
    *,
    predicted_intents: Sequence[str],
    expected_intents: Sequence[str],
    answers: Sequence[str],
    expected_keywords_list: Sequence[Sequence[str]],
    must_refuse_flags: Sequence[bool],
) -> dict:
    """
    汇总计算所有指标，返回一个可直接序列化的 dict。

    便于 run_eval.py 和未来的前端评测页复用同一套口径。
    """
    n = len(answers)
    return {
        "total_samples": n,
        "intent_accuracy": round(intent_accuracy(predicted_intents, expected_intents), 4),
        "macro_keyword_recall": round(
            macro_keyword_recall(answers, expected_keywords_list), 4
        ),
        "refusal_accuracy": round(refusal_accuracy(answers, must_refuse_flags), 4),
    }


def format_report(summary: dict, *, title: str = "评估报告") -> str:
    """把汇总 dict 渲染成人类可读的文本报告。"""
    lines = [
        f"===== {title} =====",
        f"样本总数         : {summary.get('total_samples', 0)}",
        f"意图路由准确率   : {summary.get('intent_accuracy', 0):.2%}",
        f"关键词召回率(宏) : {summary.get('macro_keyword_recall', 0):.2%}",
        f"拒答正确率       : {summary.get('refusal_accuracy', 0):.2%}",
    ]
    return "\n".join(lines)
