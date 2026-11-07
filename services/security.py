"""
多租户隔离模块

基于 contextvars 实现请求级租户上下文绑定，确保每条会话/记忆
都严格隔离在各自的租户（企业/用户）范围内。

对外提供：
    - current_tenant_id()  获取当前线程/协程的租户 ID
    - bind_tenant()        绑定租户上下文（返回 token，用于 reset）
    - tenant_scope()       上下文管理器，方便 with 块绑定
    - record_visible()    判定某条记录当前是否对本租户可见
"""

from contextlib import contextmanager
from contextvars import ContextVar

read_only_retrieval = ContextVar("read_only_retrieval", default=False)
from typing import Any, Dict, Optional
import logging
import re
import os

# 租户上下文变量；默认值 "default" 表示尚未绑定具体租户，
# 此时任何受租户保护的写入都应被拒绝。
_tenant_context: ContextVar[str] = ContextVar("tenant_id", default="default")
document_session: ContextVar[str] = ContextVar("document_session", default="")
blocked_document_ids = set()
# 每轮单独创建；仅保存通过权限校验的实际资料，不把模型草稿当作证据。
answer_sources: ContextVar[list | None] = ContextVar("answer_sources", default=None)
agent_workspace: ContextVar[dict | None] = ContextVar("agent_workspace", default=None)


def collect_answer_sources(records: list[dict]) -> list[dict]:
    visible = [r for r in records if document_visible(r)]
    if len(visible) != len(records):
        logging.warning("[guardrails] blocked_source_count=%d", len(records) - len(visible))
    sources = answer_sources.get()
    if sources is not None:
        for record in visible:
            if record not in sources:
                sources.append(record)
    return visible


def redact_identifiers(text: str) -> str:
    """Mask direct identifiers without changing substantive claims."""
    text = re.sub(r"(?<![A-Za-z0-9])\d{17}[\dXx](?![A-Za-z0-9])", "[身份标识已脱敏]", text)
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d|(?:\.\d+)?\s*(?:元|万元|亿元))", "[手机号已脱敏]", text)
    text = re.sub(r"((?:税号|纳税人识别号|统一社会信用代码|银行卡号|银行账号)\s*[：:=]?\s*)[A-Za-z0-9]{15,22}(?![A-Za-z0-9])", r"\1[标识已脱敏]", text)
    for name, value in os.environ.items():
        if name.endswith(("API_KEY", "AUTH_USER_TOKEN", "AUTH_ADMIN_TOKEN", "DEVELOPER_TOKEN")) and len(value) >= 8:
            text = text.replace(value, "[密钥已脱敏]")
    return text


def check_answer(text: str, sources: list[dict] | None = None, *, citation_details=False, allow_nonfiling=False) -> tuple[str, dict]:
    """确定性输出检查：资料内文号/条款匹配、敏感标识脱敏、明显危险建议拦截。

    匹配只证明本轮可见资料有该引用，不证明法规全球存在或现行有效。
    不读取其他用户的数据来比对，不将用户问题或计算器静态注释视为法规证据。
    """
    sources = sources if sources is not None else (answer_sources.get() or [])
    original_answer = text
    canonical = lambda value: re.sub(r"[\s第〔〕\[\]【】（）()]", "", value)
    article_pattern = r"第[一二三四五六七八九十百千万零〇\d]+条(?:之[一二三四五六七八九十\d]+)?"
    documents = []
    for source in sources:
        if document_visible(source):
            raw = str(source.get("document_number", "")) + " " + str(source.get("title", "")) + " " + str(source.get("content", ""))
            documents.append((canonical(raw), {canonical(article) for article in re.findall(article_pattern, raw)}))
    number_pattern = r"(?:财税|税总发|税总函|税总办发|财政部(?:\s*税务总局)?公告|国家税务总局公告|税务总局公告)\s*[〔\[【（(]?\s*(?:19|20)\d{2}\s*[〕\]】）)年]?\s*第?\s*\d+\s*号"
    missing = [m.group() for m in re.finditer(number_pattern, text)
               if not any(canonical(m.group()) in doc for doc, _ in documents)]
    for match in re.finditer(article_pattern, text):
        if re.match(r"(?:建议|消息|记录|数据|方案|政策源)", text[match.end():]):
            continue
        preceding = text[:match.start()]
        # 最近明确引用的法规/文号必须与条号在同一份资料里；不靠另一法规的同号条款放行。
        anchors = list(re.finditer(r"《([^》\n]{1,100})》|" + number_pattern, preceding))
        anchor = anchors[-1].group() if anchors else ""
        anchor = anchor.strip("《》")
        if not any(canonical(match.group()) in articles and (not anchor or canonical(anchor) in doc) for doc, articles in documents):
            missing.append(match.group())
    issues = []
    if missing:
        issues.append("unsupported_citation")
        text = "引用校验未通过：本轮可见知识库或已读取原文未能支持所引用的文号/条款，不能据此确定税务结论。请提供政策原文或重新检索核验。"
    # 否定/风险说明保留；明确建议做违法动作或保证法律结论时拒绝发布整份草稿。
    risky = False
    for match in re.finditer(r"虚开(?:增值税)?发票|隐瞒收入|少报收入|做两套账|无需申报|保证免税|绝对合法|保证(?:不会|不被)(?:查|追究)", text):
        if allow_nonfiling and match.group() == "无需申报":
            continue
        before = text[max(0, match.start()-10):match.start()]
        after = text[match.end():match.end()+10]
        negated = bool(re.search(r"(?:不能|不得|不要|禁止|严禁|切勿|避免|不建议|不应|拒绝|不可以|无法|不代表|没有依据)(?:再|去|通过|直接|随意|擅自|采取|进行|建议|保证|认定|认为|该方案|这个方案|此方案|的|是|为|\s){0,4}$", before)
                       or re.match(r"[，：,:\s]*(?:属于违法|是违法|可能违法|违法|违规|有风险|有法律风险|不可取|被禁止)", after))
        if not negated:
            risky = True
    if risky:
        issues.append("unsafe_tax_advice")
        text = "合规校验拦截了可能涉及违法或无依据保证的税务建议。请按真实业务记录收入和发票，并结合政策原文由财税人员核对申报条件。"
    original = text
    text = redact_identifiers(text)
    if text != original:
        issues.append("sensitive_identifier")
    report = {"ok": not issues, "issues": issues, "citation_checks": len(list(re.finditer(number_pattern, original_answer))) + len(list(re.finditer(article_pattern, original_answer))),
              "changed": bool(issues)}
    if citation_details:
        report["unmatched_citations"] = list(dict.fromkeys(missing))
    logging.info("[guardrails] ok=%s issues=%s", report["ok"], issues)
    return text, report


def document_visible(metadata: dict) -> bool:
    """RAG 文档权限：系统资料共享；用户文件必须有归属，单次文件限定会话。"""
    if metadata.get("file_id") in blocked_document_ids:
        return False
    if metadata.get("status", "active") != "active":
        return False
    if metadata.get("source_type") == "default" or metadata.get("scope") == "system":
        return True
    owner = metadata.get("owner_id")
    if not owner or str(owner) != current_tenant_id():
        return False
    return metadata.get("scope") == "global" or (
        metadata.get("scope") == "session" and bool(document_session.get())
        and metadata.get("session_id") == document_session.get())


def checkpoint_thread_id(thread_id: str) -> str:
    tenant = current_tenant_id()
    return f"{tenant}:{thread_id}" if tenant != "default" else thread_id

# 公共可见的租户占位 —— 用于标识"租户无关"的公开数据
_PUBLIC_TENANT = "public"

def current_tenant_id() -> str:
    """获取当前上下文的租户 ID。"""
    return _tenant_context.get()

def bind_tenant(tenant_id: str):
    """绑定租户 ID，返回可用于 reset 的 token。"""
    return _tenant_context.set(tenant_id)

def reset_tenant(token) -> None:
    """将租户上下文恢复到 set 之前的状态。"""
    _tenant_context.reset(token)


@contextmanager
def tenant_scope(tenant_id: str):
    """
    使用 `with tenant_scope("tenant_xxx"):` 形式绑定租户上下文，
    退出块后自动恢复，便于在 FastAPI 依赖或测试中使用。
    """
    token = bind_tenant(tenant_id)
    try:
        yield tenant_id
    finally:
        reset_tenant(token)

def require_tenant() -> str:
    """
    断言当前已绑定真实租户，否则抛出异常。
    供写操作入口（如 RecordStore.put）调用。
    """
    tenant_id = current_tenant_id()
    if tenant_id == "default":
        raise PermissionError("未绑定租户上下文：请先在请求入口调用 bind_tenant(...)")
    return tenant_id

def record_visible(record: Optional[Dict[str, Any]]) -> bool:
    """
    判断某条记录对当前租户是否可见。

    规则：
        1) 记录为空 → 不可见
        2) 记录未声明 tenant_id（历史数据/公开数据）→ 可见，由上层决定是否迁移
        3) 记录 tenant_id 为 "public" → 对所有租户可见
        4) 当前租户已绑定，且匹配记录 tenant_id → 可见
        5) 其他情况 → 不可见
    """
    if not record:
        return False
    record_tenant = record.get("tenant_id")
    if record_tenant in (None, "", _PUBLIC_TENANT):
        return True
    current = current_tenant_id()
    if current == "default":
        # 未绑定租户时看不到任何租户隔离数据
        return False
    return str(record_tenant) == current
