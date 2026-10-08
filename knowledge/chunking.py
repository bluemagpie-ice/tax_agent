"""
分块策略

todo:增加"条文合并"策略(同一法规相邻条文关联性强时合并为更大块)；
todo:增加针对"通知公告"类型文档的分块策略(此类文档通常无"第X条"结构)。
"""
import uuid
from copy import deepcopy
from typing import Any
import logging


from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_PRESET_ID = "general"

# todo：分块中的语义分块和严格分隔的含义

CHUNK_PRESETS: dict[str,dict[str,str]] = {
    "general":{
        "label":"General",
        "description":"通用分块：按分隔符和长度切分，适合大多数普通文档。",
    },
    "qa":{
        "label":"QA",
        "description":"问答分块：优先抽取问题-回答结构，适合 FAQ、题库、问答手册。",
    },
    "book":{
        "label":"Book",
        "description":"书籍分块：按章节、段落等结构切分，适合书籍等结构化文档。",
    },
    "laws":{
        "label":"Laws",
        "description":"法律分块：按法律条文、法律条款等结构切分，适合法律文档。",
    },
"semantic": {
        "label": "Semantic",
        "description": "语义分块：利用嵌入和聚类算法进行语义切分，并自动增强标题上下文。",
    },
    "separator": {
        "label": "Separator",
        "description": "严格分隔：命中分隔符即切分，仅超长片段内部继续按长度切分。",
    },
}

CHUNK_PRESET_ID = set(CHUNK_PRESETS)

CHUNK_ENGINE_VERSION = "ragflow_like_v1"
GENERAL_INTERNAL_PARSER_ID = "naive"

def deep_merge(base:dict[str,Any],override:dict[str,Any]) -> dict[str,Any]:
    """
    实现深拷贝
    """
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key,{}), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def normalize_chunk_preset_id(value:str | None) -> str:
    """
    归一化分块策略ID
    """
    if not value:
        return DEFAULT_CHUNK_PRESET_ID

    normalized = str(value).strip().lower()
    if normalized == GENERAL_INTERNAL_PARSER_ID:
        return GENERAL_INTERNAL_PARSER_ID

    if normalized in CHUNK_PRESET_ID:
        return normalized

    logging.warning(f"Unknown chunk preset ID: {value}")
    return DEFAULT_CHUNK_PRESET_ID

def map_to_internal_parser_id(preset_id:str) -> str:
    """
    将分块策略ID映射为内部解析器ID
    """
    normalized = normalize_chunk_preset_id(preset_id)
    if normalized == GENERAL_INTERNAL_PARSER_ID:
        return GENERAL_INTERNAL_PARSER_ID
    return normalized

def get_default_chunk_parser_config(preset_id: str) -> dict[str, Any]:
    normalize_chunk_preset_id(preset_id)
    return {}

def ensure_chunk_parser_config(additional_params: dict[str, Any] | None) -> dict[str, Any]:
    """
    对传入的参数进行检验，自定义的名字：chunk_preset_id, chunk_parser_config
    """
    params = dict(additional_params or {})
    params["chunk_preset_id"] = normalize_chunk_preset_id(params.get("chunk_preset_id"))

    if "chunk_parser_config" in params and not isinstance(params["chunk_parser_config"], dict):
        logging.warning(f"Invalid chunk_parser_config: {params['chunk_parser_config']}")
        params["chunk_parser_config"] = {}

    return params

def resolve_chunk_processing_params(
        kb_additional_params: dict[str, Any] | None,
        file_processing_params: dict[str, Any] | None,
        request_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    将多个层次的分块的参数配置，按照优先级合并
    kb_additional_params：知识库额外参数
    file_processing_params：文件处理参数
    request_params：请求参数
    """
    kb_additional = ensure_chunk_parser_config(kb_additional_params)
    file_params = dict(file_processing_params or {})
    request = dict(request_params or {})

    preset_id = normalize_chunk_preset_id(
        request.get("chunk_preset_id") or file_params.get("chunk_preset_id") or kb_additional.get("chunk_preset_id")
    )

    parser_config = get_default_chunk_parser_config(preset_id)
    kb_parser_config = kb_additional.get("chunk_parser_config")
    if isinstance(kb_parser_config, dict):
        parser_config = deep_merge(parser_config, kb_parser_config)

    file_parser_config = file_params.get("chunk_parser_config")
    if isinstance(file_parser_config, dict):
        parser_config = deep_merge(parser_config, file_parser_config)

    req_parser_config = request.get("chunk_parser_config")
    if isinstance(req_parser_config, dict):
        parser_config = deep_merge(parser_config, req_parser_config)

    return {
        "chunk_preset_id": preset_id,
        "chunk_parser_config": parser_config,
        "chunk_engine_version": CHUNK_ENGINE_VERSION,
    }

def split_text(text: str, preset_id: str, file_id=None, chunk_size: int = 1024, chunk_overlap: int = 128) -> list[dict[str, Any]]:
    """
    对文本进行分块。

    laws preset 专用：按「第X条 / 第X章 / 总则 / 附则」等法律结构性标记切分，
    保证一个条文是一个独立的 chunk，不被切断。
    超长条文（> chunk_size）内部再按长度补切。
    """
    if preset_id == "laws":
        return _split_law_text(text, file_id, chunk_size, chunk_overlap)

    preset_params = CHUNK_PRESETS.get(preset_id, {})
    chunk_size = preset_params.get("chunk_size", chunk_size)
    chunk_overlap = preset_params.get("chunk_overlap", chunk_overlap)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "\r\n", "。", "！", "？", "，", "、",
                    ". ", ", ", " ", ""],
    )
    raw_chunks = text_splitter.split_text(text)

    chunks = []
    for i, content in enumerate(raw_chunks):
        if not content.strip():
            continue
        chunks.append({
            "id": str(uuid.uuid4()),
            "content": content,
            "file_id": file_id,
            "chunk_index": i,
            "preset_id": preset_id,
        })
    logging.info(f"分块完成：{len(chunks)}个分块，分块方法：{preset_id}")
    return chunks


# ============================================================
# 法律文档专用分块：按「第X条 / 第X章 / 总则 / 附则」结构切分
# ============================================================
import re as _re

_LAW_STRUCT_PATTERN = _re.compile(
    r"(^|\n)\s*"
    r"(第[一二三四五六七八九十百千万0-9]+[章节条篇部]|总则|附则|前言|序言|目录)"
)


def _split_law_text(text: str, file_id, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    """
    法律文档专用分块：按条文结构切分，保证一个条文不被切断。
    超长条文内部再按长度切分。
    """
    if not text:
        return []

    # 用正则 finditer 找到所有结构性标记的位置
    markers = list(_LAW_STRUCT_PATTERN.finditer(text))

    # 如果一个结构性标记都没找到 → fallback 到通用切分
    if len(markers) < 2:
        logging.info("[laws chunking] 未检测到法律结构性标记，回退通用切分")
        return _fallback_split(text, file_id, chunk_size, chunk_overlap, "laws")

    # 按标记位置切分
    chunks_raw: list[str] = []
    for idx, m in enumerate(markers):
        start = m.start()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        segment = text[start:end].strip()
        if segment:
            chunks_raw.append(segment)

    # 处理超长条文：内部再按长度补切（保留条文开头标记在每块里）
    final_chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for seg in chunks_raw:
        if len(seg) <= chunk_size:
            final_chunks.append({
                "id": str(uuid.uuid4()),
                "content": seg,
                "file_id": file_id,
                "chunk_index": chunk_index,
                "preset_id": "laws",
            })
            chunk_index += 1
        else:
            # 超长 → 内部按句子切（保留 overlapping）
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "；", ", ", ""],
            )
            sub_parts = sub_splitter.split_text(seg)
            for part in sub_parts:
                if not part.strip():
                    continue
                final_chunks.append({
                    "id": str(uuid.uuid4()),
                    "content": part.strip(),
                    "file_id": file_id,
                    "chunk_index": chunk_index,
                    "preset_id": "laws",
                })
                chunk_index += 1

    logging.info(f"[laws chunking] 按条文结构切分完成：{len(final_chunks)} 个分块")
    return final_chunks


def _fallback_split(text: str, file_id, chunk_size: int, chunk_overlap: int, preset_id: str) -> list[dict[str, Any]]:
    """laws preset 回退通用切分的辅助函数"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""],
    )
    raw = text_splitter.split_text(text)
    out = []
    for i, content in enumerate(raw):
        if not content.strip():
            continue
        out.append({
            "id": str(uuid.uuid4()),
            "content": content,
            "file_id": file_id,
            "chunk_index": i,
            "preset_id": preset_id,
        })
    return out


def get_chunk_preset_options() -> list[dict[str, str]]:
    """
    连接前端
    """
    return [
        {"value": preset_id, "label": preset["label"], "description": preset["description"]}
        for preset_id, preset in CHUNK_PRESETS.items()
    ]