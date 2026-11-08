"""Model-visible excerpts from the official pages already read for this turn."""

import json
import re

from langchain_core.messages import HumanMessage


WEB_POLICY = "web_policy"
_SOURCE_FIELDS = ("title", "document_number", "issuer", "publish_date", "document_date", "effective_date", "url")
_CONDITION = re.compile(r"条件|同时符合|适用|不超过|不含|不得|除外|减按|税率|免征|减征|施行|执行至|有效期|从业人数|资产总额|应纳税所得额")
_ITEM_NUMBER = r"[一二三四五六七八九十百零〇]+"
_ANNOUNCEMENT_NUMBER = re.compile(r"\d{4}年?第\d+号")


def policy_messages(web_policy):
    messages = []
    seen_urls, seen_bodies = set(), set()
    for source in (web_policy or {}).get("sources", []):
        body = source.get("content", "").strip()
        if source["page_status"] != "success" or not body or source["url"] in seen_urls or body in seen_bodies:
            continue
        seen_urls.add(source["url"])
        seen_bodies.add(body)
        record = {key: source[key] for key in _SOURCE_FIELDS if source.get(key)}
        record["content"] = body
        if source.get("truncated"):
            record["context_notice"] = "网页读取不完整，仅为正文片段。缺失的适用条件、例外及期限未核验。"
        messages.append(HumanMessage(content=json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                                     additional_kwargs={"context_optional": True, "context_kind": WEB_POLICY}))
    return messages


def policy_terms(text):
    return set(re.findall(r"[a-z0-9_]+", text.lower())) | {
        run[i:i + 2] for run in re.findall(r"[\u4e00-\u9fff]+", text) for i in range(len(run) - 1)
    }


def policy_relevance(record, query, terms):
    body = record["content"]
    body_matches = len(terms & policy_terms(body))
    has_relief_terms = bool(re.search(r"减按|减征|免征|税率", body))
    return ("优惠" in query and has_relief_terms and body_matches >= 4,
            len(terms & policy_terms(record.get("title", ""))), body_matches)


def normalize_policy_item_citations(answer, sources):
    """Use the published item label when a line explicitly identifies that announcement."""
    lines = answer.split("\n")
    for i, line in enumerate(lines):
        compact = re.sub(r"\s+", "", line)
        cited_numbers = set(_ANNOUNCEMENT_NUMBER.findall(compact))
        if len(cited_numbers) != 1:
            continue
        cited = [s for s in sources if s.get("document_number")
                 and re.sub(r"\s+", "", s["document_number"]) in compact]
        if len(cited) != 1:
            continue
        source = cited[0]
        if any(title != source.get("title") for title in re.findall(r"《([^》]+)》", line)):
            continue
        items = set(re.findall(rf"(?m)^\s*({_ITEM_NUMBER})、", source["content"]))
        lines[i] = re.sub(rf"(?<!法)(?<!条例)(?<!规定)第({_ITEM_NUMBER})条(?!之)",
                          lambda m: f"“{m[1]}、”项" if m[1] in items else m[0], line)
    return "\n".join(lines)


def fit_policy_message(message, query_terms, budget, cost):
    """Keep complete text blocks and the source envelope. Mark omissions explicitly."""
    record = json.loads(message.content)
    body = record.pop("content")
    was_truncated = bool(record.get("context_notice"))
    record["context_notice"] = "证据不完整，正文已按预算节选。缺失的适用条件、例外及期限未核验。"

    def render(blocks, kept=()):
        pieces = []
        previous = None
        for i in sorted(kept):
            if previous is not None and i != previous + 1:
                pieces.append("【中间正文已省略】")
            pieces.append(blocks[i])
            previous = i
        content = "\n".join(pieces)
        return message.model_copy(update={"content": json.dumps({**record, "content": content},
                                                                ensure_ascii=False, separators=(",", ":"))})

    # A fetch can end mid-clause. Exclude its incomplete tail when selecting excerpts.
    if was_truncated and not re.search(r"[。！？；.!?]$", body):
        ends = list(re.finditer(r"[。！？；.!?](?=\s|$)", body))
        body = body[:ends[-1].end()] if ends else ""
    # Preserve paragraphs and combine wrapped lines that do not end a clause.
    blocks = [part.strip() for part in re.split(r"(?<=[。！？])\s*|(?<=[；.!?])\s+|\n\s*\n", body) if part.strip()]
    if not blocks:
        return render([])
    overlap = [len(query_terms & policy_terms(part)) for part in blocks]
    best = max(overlap)
    conditions = {i for i, block in enumerate(blocks) if _CONDITION.search(block)}
    candidates = {i for i, score in enumerate(overlap) if score >= max(1, best // 2)} if best else conditions
    kept = set()
    for i in sorted(candidates, key=lambda j: (overlap[j], j in conditions, -j), reverse=True):
        # Keep neighboring conditions and exceptions together with the matching clause.
        bundle = {i} | ({i - 1, i + 1} & conditions)
        if cost(render(blocks, kept | bundle)) <= budget:
            kept.update(bundle)
    return render(blocks, kept)
