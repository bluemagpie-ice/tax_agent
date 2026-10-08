"""
评估模块（eval）
================

用于量化税务 Agent 的表现，包含：
- dataset.py   评测样本结构 + 内置税务问答评测集
- metrics.py   纯函数指标（意图准确率 / 关键词召回 / 拒答正确率）
- run_eval.py  命令行评测入口

典型用法（在项目根目录 D:\\code\\tax_agent 下执行）：

    # 只评测意图路由（纯本地，不需要 LLM 与 Qdrant）
    python -m eval.run_eval --mode intent

    # 完整评测（意图 + 答案质量，需要 LLM 与 Qdrant 可用）
    python -m eval.run_eval --mode all

    # 只评测答案质量
    python -m eval.run_eval --mode answer
"""
