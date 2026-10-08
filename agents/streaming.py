"""
流式输出工具模块
================

职责：
1. 把 LangGraph Agent 的 async generator（由 BaseAgent.stream_messages 产出）
   封装为 SSE（Server-Sent Events）格式的 async generator，供 FastAPI 的
   StreamingResponse 直接消费。
2. 处理 AIMessageChunk 的 content（可能是 str，也可能是 list[dict] 形式的
   multimodal 片段），统一输出纯文本 token。
3. 向前端推送统一的事件类型：
       {"type": "token",    "data": "..."}    — 单个 token / 文本片段
       {"type": "thought",  "data": "..."}    — 思维链 / 工具调用（预留）
       {"type": "citations","data": [...]}    — RAG 引用来源（预留）
       {"type": "done",     "data": null}     — 生成结束
       {"type": "error",    "data": "..."}    — 异常

给 FastAPI 路由的用法示例：
    from fastapi.responses import StreamingResponse
    from agents.knowledge_agent import KnowledgeAgent
    from agents.streaming import agent_to_sse

    @app.post("/api/chat/stream")
    async def chat_stream(...):
        agent = KnowledgeAgent()
        async def gen():
            async for line in agent_to_sse(
                agent.stream_messages(messages, input_context=ctx)
            ):
                yield line
        return StreamingResponse(gen(), media_type="text/event-stream")
"""

from __future__ import annotations

import json
import traceback
from typing import AsyncIterator, Any

# --------------------------------------------------------------- #
# 1. 从 LangGraph astream 事件里提取纯文本
# --------------------------------------------------------------- #

def _extract_text(chunk: Any) -> str:
    """
    从 LangGraph astream(stream_mode="messages") 产出的 event 中提取文本。

    event 有两种常见形态：
        a) AIMessageChunk 实例（langchain_core）
        b) 普通 str（极少数节点直接 yield 字符串）

    AIMessageChunk.content 可能是：
        - str                          — 最常见
        - list[dict]                   — multimodal，如 [{"type":"text","text":"..."}]
        - None                         — 空内容（仅 tool_call / reasoning）
    """
    if chunk is None:
        return ""

    # 已经是字符串
    if isinstance(chunk, str):
        return chunk

    # AIMessageChunk / BaseMessage
    content = getattr(chunk, "content", None)
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    # multimodal list[dict] —— 抽取 type=="text" 的片段
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                # 兼容 LangChain 新版的 reasoning_content 字段
                elif item.get("type") == "reasoning" and item.get("reasoning"):
                    # 思维链目前不单独推送，合并到 token 里（前端再拆分）
                    parts.append(str(item["reasoning"]))
        return "".join(parts)

    return str(content)

# --------------------------------------------------------------- #
# 2. 把 LangGraph astream 迭代器 → SSE 格式迭代器
# --------------------------------------------------------------- #
async def agent_to_sse(
    agent_stream: AsyncIterator[Any],
    *,
    include_metadata: bool = False,
) -> AsyncIterator[str]:
    """
    消费 BaseAgent.stream_messages() 产出的 async generator，
    逐行产出符合 SSE 协议的字符串（"data: {json}\\n\\n" 格式）。

    :param agent_stream:    BaseAgent.stream_messages(...) 的返回值
    :param include_metadata: 是否把 LangGraph 附带的 metadata 一起推送
                             （调试用，生产环境建议 False）
    """
    try:
        async for event in agent_stream:
            # ---------- LangGraph stream_mode="messages" 的 event 形态 ----------
            # 标准形态：(AIMessageChunk, metadata_dict)
            # 也可能直接是 AIMessageChunk / str（某些自定义节点）
            chunk: Any
            metadata: dict = {}

            if isinstance(event, tuple) and len(event) >= 1:
                chunk = event[0]
                metadata = event[1] if len(event) > 1 and isinstance(event[1], dict) else {}
            else:
                chunk = event

            text = _extract_text(chunk)

            # 跳过空 token（tool_call 触发的 chunk 等）
            if not text.strip():
                # 但如果 chunk 有 reasoning_content，也推送 thought 事件
                reasoning = getattr(chunk, "additional_kwargs", {}).get("reasoning_content") if not isinstance(chunk, str) else None
                if reasoning:
                    yield _sse_line("thought", reasoning)
                continue

            payload: dict = {}
            if include_metadata and metadata:
                payload["meta"] = metadata
            payload["text"] = text

            yield _sse_line("token", payload)

    except Exception as exc:
        # 异常也通过 SSE 推送，而不是让连接断开
        err_msg = f"{type(exc).__name__}: {exc}"
        yield _sse_line("error", err_msg)
        # 把 traceback 打到服务端日志，前端只看到简短错误
        traceback.print_exc()
        return

    # 正常结束
    yield _sse_line("done", None)
    # SSE 注释行（可选，由前端决定是否关掉）
    yield ": stream closed\n\n"

# --------------------------------------------------------------- #
# 3. 辅助：构造一条 SSE 事件
# --------------------------------------------------------------- #

def _sse_line(event_type: str, data: Any) -> str:
    """
    拼一条 SSE 格式字符串。

    SSE 协议：
        event: <type>\n
        data: <json>\n
        \n

    注意：data 里的换行会被 SSE 协议解释为多段，
    所以统一用 json.dumps 转义，保证一行一条 data。
    """
    # ensure_ascii=False —— 中文原样输出，不转成 \uXXXX
    data_str = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data_str}\n\n"

# --------------------------------------------------------------- #
# 4. 预留：非流式结果 → 一次性 SSE（供未来切换/降级使用）
# --------------------------------------------------------------- #
async def result_to_sse(
    result: Any,
) -> AsyncIterator[str]:
    """
    把 invoke_messages() 的非流式结果也包装成 SSE。
    用法：当 Agent 暂不支持流式时做降级兜底。
    """
    # 从结果里提取最后一条 AI 消息
    text = ""
    if isinstance(result, dict):
        for msg in reversed(result.get("messages", [])):
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            if role in ("ai", "assistant"):
                text = _extract_text(msg)
                break
    elif hasattr(result, "content"):
        text = _extract_text(result)

    # 模拟逐 token 吐出（前端体验一致）
    for ch in text:
        yield _sse_line("token", {"text": ch})

    yield _sse_line("done", None)