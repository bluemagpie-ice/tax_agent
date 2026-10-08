"""
评测运行入口
============

用法（在项目根目录 D:\\code\\tax_agent 下执行）：

    # 只评测意图路由（纯本地，不需要 LLM 与 Qdrant）
    python -m eval.run_eval --mode intent

    # 只评测答案质量（需要真实 LLM + Qdrant 知识库）
    python -m eval.run_eval --mode answer

    # 完整评测（意图 + 答案，默认）
    python -m eval.run_eval --mode all

    # 指定评测对象（默认 tax_agent 编排器）
    python -m eval.run_eval --agent knowledge_agent

    # 把明细结果导出 JSON
    python -m eval.run_eval --mode all --json eval_report.json

说明：
- intent 评测走 HybridIntentRouter.classify()，纯本地，无需 LLM/向量库。
- answer 评测走 Agent.run()，需要真实 LLM 与 Qdrant 知识库已灌入数据。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from eval.dataset import load_dataset
from eval.metrics import (
    intent_accuracy,
    keyword_recall,
    macro_keyword_recall,
    refusal_accuracy,
    refusal_detected,
)


def run_intent_eval(samples):
    """意图路由评测（离线）。返回 (predicted_intents, 明细)。"""
    from agents.intent_router import HybridIntentRouter

    router = HybridIntentRouter()
    predicted = []
    details = []
    for s in samples:
        plan = router.classify(s.query)
        predicted.append(plan["intent"])
        details.append({
            "id": s.id,
            "query": s.query,
            "expected_intent": s.expected_intent,
            "predicted_intent": plan["intent"],
            "match": plan["intent"] == s.expected_intent,
        })
    return predicted, details


async def run_answer_eval(samples, agent_name: str):
    """答案质量评测（需 LLM + Qdrant）。返回 (answers, 明细)。"""
    if agent_name == "tax_agent":
        from agents.tax_agent import TaxAgent
        agent = TaxAgent()
    elif agent_name == "knowledge_agent":
        from agents.knowledge_agent import KnowledgeAgent
        agent = KnowledgeAgent()
    else:
        raise ValueError(f"未知 agent: {agent_name}")

    answers = []
    details = []
    for s in samples:
        try:
            ans = await agent.run(s.query, context={"thread_id": f"eval_{s.id}"})
        except Exception as e:  # noqa: BLE001 —— 单条失败不中断整体评测
            ans = f"（评测执行出错: {e}）"
        answers.append(ans)
        details.append({
            "id": s.id,
            "query": s.query,
            "answer": ans,
            "keyword_recall": round(keyword_recall(ans, s.expected_keywords), 4),
            "refused": refusal_detected(ans),
            "must_refuse": s.must_refuse,
        })
    return answers, details


def main() -> None:
    parser = argparse.ArgumentParser(description="税务 Agent 评测")
    parser.add_argument("--mode", choices=["intent", "answer", "all"], default="all")
    parser.add_argument(
        "--agent", choices=["tax_agent", "knowledge_agent"], default="tax_agent"
    )
    parser.add_argument("--json", type=str, default="", help="明细结果 JSON 输出路径")
    args = parser.parse_args()

    samples = load_dataset()
    expected_intents = [s.expected_intent for s in samples]
    expected_keywords = [s.expected_keywords for s in samples]
    must_refuse = [s.must_refuse for s in samples]

    intent_details: list[dict] = []
    answer_details: list[dict] = []

    # ---------- 意图路由评测（离线） ----------
    if args.mode in ("intent", "all"):
        predicted_intents, intent_details = run_intent_eval(samples)
        acc = intent_accuracy(predicted_intents, expected_intents)
        hit = sum(1 for p, e in zip(predicted_intents, expected_intents) if p == e)
        print(f"意图路由准确率: {acc:.2%} ({hit}/{len(samples)})")
        for d in intent_details:
            if not d["match"]:
                print(
                    f"  错例 [{d['id']}] {d['query']}\n"
                    f"        期望={d['expected_intent']} 实际={d['predicted_intent']}"
                )

    # ---------- 答案质量评测（需 LLM + Qdrant） ----------
    if args.mode in ("answer", "all"):
        try:
            answers, answer_details = asyncio.run(run_answer_eval(samples, args.agent))
        except ImportError as e:
            print(
                f"[警告] 答案评测不可用（缺少依赖 {getattr(e, 'name', e)}），跳过。\n"
                f"       请在安装 langchain / langgraph 等依赖后，在你的开发环境运行。"
            )
            answers = []
        else:
            kr = macro_keyword_recall(answers, expected_keywords)
            ra = refusal_accuracy(answers, must_refuse)
            print(f"关键词召回率(宏): {kr:.2%}")
            print(f"拒答正确率      : {ra:.2%}")
            for d in answer_details:
                if d["refused"] != d["must_refuse"]:
                    print(
                        f"  拒答判定错误 [{d['id']}] {d['query']}\n"
                        f"        应拒答={d['must_refuse']} 实际拒答={d['refused']}"
                    )

    # ---------- 导出 JSON ----------
    if args.json and (intent_details or answer_details):
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {"intent": intent_details, "answer": answer_details},
                f, ensure_ascii=False, indent=2,
            )
        print(f"\n详细结果已写入 {args.json}")


if __name__ == "__main__":
    main()
