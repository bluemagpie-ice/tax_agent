"""Evidence normalization is deliberately conservative about legal validity."""
import hashlib
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlparse, urlsplit, urlunsplit

from services.research_store import utcnow

POLICY_METADATA_FIELDS = (
    "tax_type", "document_number", "issuer", "effective_date", "is_current", "owner_id",
    "session_id", "source_url", "url", "publish_date", "published_date", "region",
    "expiry_date", "superseded_by", "law_type", "articles",
)


def official_url(url):
    if not isinstance(url, str) or any(c.isspace() or c == "\\" for c in url):
        return False
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None or port not in (None, 80, 443):
        return False
    return parsed.scheme in ("http", "https") and any(
        host == root or host.endswith("." + root) for root in ("chinatax.gov.cn", "mof.gov.cn", "gov.cn"))


def normalize_evidence(record, *, kind="policy_rag"):
    content = str(record.get("content") or "")
    url = record.get("source_url") or record.get("url") or ""
    identity = str(record.get("id") or record.get("file_id") or url) + "\0" + content
    return {
        "evidence_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
        "kind": kind, "title": record.get("title") or record.get("file_name") or "未标注标题的资料", "source_url": url or None,
        "issuer": record.get("issuer") or None, "document_number": record.get("document_number") or None,
        "publish_date": record.get("publish_date") or record.get("published_date") or None,
        "response_date": record.get("response_date"), "document_date": record.get("document_date"),
        "effective_date": record.get("effective_date") or None, "expiry_date": record.get("expiry_date") or None,
        "region": record.get("region") or None, "is_current": record.get("is_current"),
        "superseded_by": record.get("superseded_by") or None,
        "articles": re.findall(r"第[一二三四五六七八九十百零〇\d]+条[^\n]*", content),
        "content": content, "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        "retrieved_at": utcnow(), "read_status": "read_excerpt" if content else "not_read",
        "truncated": bool(record.get("truncated")),
        "source_check": "official_domain" if official_url(url) else "unverified",
        "validity_check": "superseded" if record.get("superseded_by") else record.get("validity_check", "unknown"), "applicability_check": "unknown",
        "source_type": record.get("source_type", kind),
        "file_id": record.get("file_id"),
    }


def source_version(evidence):
    """Group excerpts of a file, without conflating different files by title."""
    if evidence.get("file_id"):
        return "file:" + evidence["file_id"]
    if evidence.get("source_url"):
        parts = urlsplit(evidence["source_url"])
        return urlunsplit(parts._replace(fragment="")) + ":" + evidence["content_hash"]
    return evidence["content_hash"]


def restore_policy_metadata(records):
    """Fill fields missing from old chunks using an exact tracker file ID match."""
    from services.paths import DATA_DIR
    path = DATA_DIR / "policy_tracker.sqlite3"
    ids = {record.get("file_id") for record in records} - {None, ""}
    if not ids or not path.is_file():
        return
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1) as connection:
            rows = connection.execute(
                "SELECT file_id, title, source_url, published_date FROM policy_record WHERE file_id IN ("
                + ",".join("?" for _ in ids) + ")", tuple(ids)
            ).fetchall()
    except sqlite3.Error:
        return
    by_id = {row[0]: dict(zip(("title", "source_url", "publish_date"), row[1:])) for row in rows}
    for record in records:
        for key, value in by_id.get(record.get("file_id"), {}).items():
            if value and not record.get(key):
                record[key] = value


def compact_evidence(evidence, focus=""):
    """Keep short source-bound excerpts so several policies fit one model call."""
    keys = ("evidence_id", "title", "source_url", "document_number", "publish_date", "document_date", "response_date", "effective_date",
            "expiry_date", "region", "source_check", "is_current", "superseded_by", "validity_check", "source_type")

    def bigrams(value):
        return {pair for word in re.findall(r"[\u4e00-\u9fff]+", value) for pair in zip(word, word[1:])}

    topic = bigrams(focus)
    years = set(re.findall(r"(?:19|20)\d{2}", focus))
    for record in evidence:
        content = record["content"]
        if len(content) > 850:
            normalized = re.sub(r"\s+", "", content)
            sentences = re.split(r"(?<=[。；！？])", normalized)
            scored = []
            for index, sentence in enumerate(sentences):
                if len(sentence) < 8:
                    continue
                date = any(year in sentence for year in years) and re.search(r"施行|执行|日起|期间|日前|年底|至\d{4}", sentence)
                scored.append((len(topic & bigrams(sentence)) + (8 if date else 0), index, sentence))
            selected = sorted(sorted(scored, reverse=True)[:5], key=lambda row: row[1])
            content = "\n[… ]\n".join(row[2][:200] for row in selected) or normalized[:850]
        yield {**{key: record.get(key) for key in keys}, "content": content}


def assessment_evidence(records, focus):
    """Number short, source-bound passages so the assessor can select IDs."""
    counter = 0
    for record in compact_evidence(records, focus):
        passages = {}
        seen = set()
        for sentence in re.split(r"\n\[… \]\n|(?<=[。；！？])", record.pop("content")):
            quote = sentence.strip()
            key = "".join(quote.split())
            if len(key) < 6 or key in seen:
                continue
            seen.add(key)
            counter += 1
            passages[f"p{counter}"] = quote
        if passages:
            yield {**record, "passages": passages}


def traceable_evidence(record):
    """A readable official URL or an uploaded policy with an exact stored file ID."""
    return bool(record.get("content") and (official_url(record.get("source_url"))
                or (record.get("kind") == "policy_rag" and record.get("file_id"))))


EVIDENCE_GAP_CODES = {"insufficient_evidence", "policy_source_unverified", "policy_original_unavailable"}


def legacy_failed_report(data):
    scope = (data.get("report") or {}).get("scope", "")
    return scope in {"初稿校验失败且修正预算耗尽", "预算耗尽，研究尚未完成"}


def validate_report(report, evidence):
    """Each analysis cites readable policy evidence. Never promotes model validity claims."""
    available = {e["evidence_id"]: e for e in evidence if e.get("content")}
    for item in report.get("analysis", []):
        refs = item.get("evidence_ids", [])
        if any(ref not in available for ref in refs):
            raise ValueError("报告引用了不存在或未读取的政策证据")
        if not refs:
            item["qualification"] = "缺少政策依据，仅为待核验分析"
        else:
            item["qualification"] = "AI 初步分析，政策效力及业务适用性仍需人工核实"
    report["policy_evidence"] = evidence
    report["notice"] = "事实来自本任务输入；政策资料为实际读取的节选；AI 分析未经专业定稿。未知信息不代表已核实。"
    return report
