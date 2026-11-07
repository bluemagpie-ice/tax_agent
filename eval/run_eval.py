"""
评测运行入口
============

用法（在项目根目录 D:\\code\\tax_agent 下执行）：

    # 只评测意图路由（纯本地，不需要 LLM 与 Qdrant）
    python -m eval.run_eval --mode intent

    # 只评测答案质量（需要真实 LLM + Qdrant 知识库）
    python -m eval.run_eval --mode answer --live

    # 完整评测（意图 + 答案，默认）
    python -m eval.run_eval --mode all --live

    # 指定评测对象（默认 tax_agent 编排器）
    python -m eval.run_eval --agent knowledge_agent

    # 把明细结果导出 JSON
    python -m eval.run_eval --mode all --live --json eval_report.json

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


async def run_answer_eval(samples, agent_name: str, *, runtime_root=None):
    """Run with unique tenant, memory and checkpoint paths; errors score zero."""
    import uuid
    from pathlib import Path
    from agents.base_agent import runtime_memory, runtime_checkpoint_path, runtime_connections
    from agents.memory import ConversationMemory
    from services.paths import DATA_DIR
    from services.security import tenant_scope, read_only_retrieval
    run_id = uuid.uuid4().hex
    directory = Path(runtime_root) if runtime_root else DATA_DIR / "eval_runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    connections = []
    tokens = [runtime_memory.set(ConversationMemory(directory / "memory")),
              runtime_checkpoint_path.set(directory / "checkpoints.sqlite"), runtime_connections.set(connections)]
    read_only_token = read_only_retrieval.set(True)
    answers, details = [], []
    try:
        if agent_name == "tax_agent":
            from agents.tax_agent import TaxAgent
            agent = TaxAgent(workdir_root=str(directory))
        elif agent_name == "knowledge_agent":
            from agents.knowledge_agent import KnowledgeAgent
            agent = KnowledgeAgent(workdir_root=str(directory))
        else:
            raise ValueError(f"未知 agent: {agent_name}")
        with tenant_scope("eval-" + run_id):
            for sample in samples:
                try:
                    answer = await agent.run(sample.query, context={"thread_id": f"eval_{run_id}_{sample.id}"})
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("empty answer")
                    success, error_type = True, None
                except Exception as exc:
                    answer, success, error_type = "", False, type(exc).__name__
                answers.append(answer)
                refused = refusal_detected(answer) if success else None
                details.append({"id":sample.id, "query":sample.query, "answer":answer,
                    "execution_success":success, "error_type":error_type,
                    "keyword_recall":round(keyword_recall(answer, sample.expected_keywords),4) if success else 0,
                    "refused":refused, "must_refuse":sample.must_refuse,
                    "refusal_correct": bool(success and refused == sample.must_refuse)})
        return answers, details
    finally:
        for connection in connections:
            await connection.close()
        runtime_memory.reset(tokens[0]); runtime_checkpoint_path.reset(tokens[1]); runtime_connections.reset(tokens[2])
        read_only_retrieval.reset(read_only_token)


def main() -> None:
    parser = argparse.ArgumentParser(description="税务 Agent 评测")
    parser.add_argument("--mode", choices=["intent", "answer", "all"], default="intent")
    parser.add_argument(
        "--agent", choices=["tax_agent", "knowledge_agent"], default="tax_agent"
    )
    parser.add_argument("--json", type=str, default="", help="明细结果 JSON 输出路径")
    parser.add_argument("--live", action="store_true", help="显式启用真实模型评测，可能产生调用费用")
    args = parser.parse_args()
    if args.mode in ("answer", "all") and not args.live:
        parser.error("答案评测需要显式 --live；离线运行使用 --mode intent")

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
            kr = sum(d["keyword_recall"] for d in answer_details) / max(len(answer_details), 1)
            ra = refusal_accuracy(answers, must_refuse, execution_success=[d["execution_success"] for d in answer_details])
            print(f"关键词召回率(宏): {kr:.2%}")
            print(f"拒答正确率      : {ra:.2%}")
            print(f"执行失败样本数  : {sum(not d['execution_success'] for d in answer_details)}")
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
