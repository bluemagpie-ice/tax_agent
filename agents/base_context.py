"""
Base Context
定义税务智能体的基础配置模板
"""

import uuid
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class BaseContext:
    """
    定义基础 Context 类，供各类 graph / agent 继承。

    字段说明：
    - thread_id:     会话线程唯一标识（用于多轮对话隔离）
    - uid:           用户唯一标识
    - system_prompt: 系统提示词，用于驱动税务智能体的角色与行为
    - model:         驱动模型名称，留空时使用系统默认模型
    - tools:         智能体可调用的工具列表
    - knowledges:    智能体可访问的知识库列表
    """

    thread_id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"name": "线程ID", "configurable": False, "description": "唯一标识一个对话线程"},
    )


    uid: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"name": "UID", "configurable": False, "description": "唯一标识一个用户"},
    )

    system_prompt: str = field(
        default=(
            "你是一个专业的税务AI助手，擅长解答税务咨询、"
            "税务计算、税务筹划、发票处理、合规检查、申报指导等问题。"
            "回答要求准确、清晰、易懂，并在涉及具体税法规定时标注参考依据。"
        ),
        metadata={"name": "系统提示词", "description": "描述智能体的角色和行为（税务场景）", "kind": "prompt"},
    )

    model: str = field(
        default="",
        metadata={
            "name": "智能体模型",
            "options": [],
            "description": "智能体的驱动模型，留空时使用系统默认模型。",
            "kind": "llm",
        },
    )

    tools: list[str] | None = field(
        default=None,
        metadata={
            "name": "工具",
            "description": "智能体可调用的工具列表，默认选择当前用户可用的全部工具。",
            "type": "list",
            "kind": "tools",
        },
    )

    knowledges: list[str] | None = field(
        default=None,
        metadata={
            "name": "知识库",
            "description": "智能体可访问的知识库列表，默认选择当前用户可访问的全部知识库。",
            "type": "list",
            "kind": "knowledges",
        },
    )

    source_types: list[str] | None = field(
        default=None,
        metadata={
            "name": "知识来源",
            "description": "知识来源筛选：default / user_upload / web_search",
        },
    )