"""
Agent 可调用的工具包（LangChain StructuredTool + 纯函数双轨）
===========================================================

当前提供两类 Tool：
  1. 税法条文检索  - 查本地 RAG 知识库（Qdrant + BM25）
  2. 联网搜索     - 查财政部/税务总局官网最新政策

使用方式：
  # 纯函数（Agent 内部强制调用模式）
  from services.tools import search_tax_policy, search_web_policy

  # LangChain Tool（给 LLM ReAct 模式）
  from services.tools import TAX_POLICY_TOOL, WEB_SEARCH_TOOL
"""

try:
    from services.tools.kb_tools import (
        search_tax_policy,
        TAX_POLICY_TOOL,
        WEB_SEARCH_TOOL,
        ALL_TOOLS,
    )
    __all__ = [
        "search_tax_policy",
        "TAX_POLICY_TOOL",
        "WEB_SEARCH_TOOL",
        "ALL_TOOLS",
    ]
except ImportError:
    __all__ = []