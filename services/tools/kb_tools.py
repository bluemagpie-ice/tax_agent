"""
税法条文检索 Tool + 联网搜索 Tool
=================================

设计原则：
  1. 纯函数入口（给 Agent 强制调用）和 LangChain StructuredTool（给 LLM ReAct）共享底层实现
  2. 检索结果必须包含：条文内容 + 标题 + 文号 + 发布机关 + 生效日期（LLM 要用来溯源）
  3. 联网搜索用 Bing（国内可访问、无需 API Key；DuckDuckGo 在国内不可用）
  4. 所有 Tool 返回 JSON 字符串（LangChain StructuredTool 标准格式）
"""

import json
import logging
import os
import base64
import time
import socket
import ipaddress
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from urllib.parse import urlparse, parse_qs, urljoin
from typing import Literal, Optional

import requests

from services.cache_utils import TTLCache
from services.security import blocked_document_ids, current_tenant_id, document_session
from services.config_manager import ConfigManager

logger = logging.getLogger(__name__)

_search_cache = TTLCache(ttl_seconds=300, maxsize=128)

# ==============================================================
# 1. 税法条文检索（本地 RAG 知识库）
# ==============================================================

def search_tax_policy(
    query: str,
    tax_type: Optional[str] = None,
    limit: int = 5,
    is_current: Optional[bool] = None,
    *, research: bool = False,
) -> list[dict]:
    """
    从本地税务知识库检索相关条文（纯函数入口，Agent 内部强制调用）。

    :param query: 自然语言查询，如 "增值税进项抵扣"
    :param tax_type: 可选税种过滤，如 "vat" / "corporate_income" / "individual_income" / "general"
    :param limit: 返回前 N 条
    :param is_current: 可选，True=只搜现行有效，False=只搜已废止
    :return: 列表，每项包含 id / score / content / file_id 以及 title / document_number / issuer / effective_date / law_type / tax_type / is_current 等元数据
             注意：is_current 缺失时保留 None（未知不等于有效，也不等于已废止）
    """
    try:
        from knowledge.embedding import EmbeddingService, PersistentDocumentStore, QdrantNotConfigured

        # 组装 Qdrant 过滤条件
        filters: dict = {}
        # Legal metadata on historical chunks may be missing. Preferences are soft.

        cache_key = (current_tenant_id(), document_session.get(), query.strip(), tax_type or "", limit, is_current, research,
                     ConfigManager().get_embedding_config()["model"])
        cached = _search_cache.get(cache_key)
        if cached is not None:
            logger.info(f"[search_tax_policy] 缓存命中 → {len(cached)} 条")
            return deepcopy([item for item in cached if item.get("file_id") not in blocked_document_ids])

        store = PersistentDocumentStore()
        options = {"qdrant_read_only": True, "use_bm25": False, "use_neo4j": False} if research else {}
        try:
            svc = EmbeddingService(store=store, use_qdrant=True, skip_rebuild=True, **options)
        except QdrantNotConfigured:
            if research and not any(chunk.metadata.get("source_type") == "default" for chunk in store.get_chunks()):
                return []
            raise
        if research and not svc.qdrant.has_policy_chunks():
            return []
        if research and not svc.embedding_model:
            raise RuntimeError("有政策资料，但尚未配置向量模型")

        # 加一点检索增强（让 RRF 的 BM25 分支也受益）
        enhanced = query if research else f"税法 政策 条文 {query}"
        results = svc.search(enhanced, limit=limit, mode="semantic" if research else "rrf", filters=filters or None)

        results.sort(key=lambda item: (item.get("score", 0), item.get("tax_type") == tax_type, item.get("is_current") is is_current), reverse=True)
        # 格式化输出
        formatted = []
        for r in results:
            formatted.append({
                "id": r.get("id", ""),
                "file_id": r.get("file_id", ""),
                "title": r.get("title", "") or r.get("file_name", ""),
                "document_number": r.get("document_number", ""),
                "issuer": r.get("issuer", ""),
                "effective_date": r.get("effective_date", ""),
                "law_type": r.get("law_type", ""),
                "tax_type": r.get("tax_type", ""),
                "is_current": r.get("is_current"),
                "score": r.get("score", 0),
                "content": r.get("content", ""),
                **{key: r.get(key) for key in ("source_url", "url", "publish_date", "published_date",
                    "region", "expiry_date", "superseded_by", "articles", "source_type", "owner_id", "scope", "session_id", "status")
                   if key in r},
            })
        _search_cache.set(cache_key, formatted)
        return deepcopy(formatted)

    except Exception as e:
        if research:
            raise
        logger.error(f"[search_tax_policy] 检索失败: {e}", exc_info=True)
        raise RuntimeError("知识库检索失败，结论未核验") from e

# ---- LangChain Tool 封装 ----

try:
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class TaxPolicyInput(BaseModel):
        """税法条文检索 Tool 的参数 schema"""
        query: str = Field(description="要查询的税务问题或关键词，如 '增值税进项抵扣' / '小型微利企业所得税优惠'")
        tax_type: Optional[Literal["vat", "corporate_income", "individual_income", "stamp_duty", "surcharges", "general"]] = Field(
            default=None,
            description="可选的税种过滤：vat=增值税 / corporate_income=企业所得税 / individual_income=个人所得税 / stamp_duty=印花税 / surcharges=附加税费 / general=通用",
        )
        limit: int = Field(default=5, description="返回的最相关条文数量，默认 5，最多 10")
        is_current: Optional[bool] = Field(
            default=None,
            description="可选，True=只搜现行有效文件，False=只搜已废止文件",
        )

    def _tool_search_tax_policy(
        query: str,
        tax_type: Optional[str] = None,
        limit: int = 5,
        is_current: Optional[bool] = None,
    ) -> str:
        try:
            results = search_tax_policy(query, tax_type=tax_type, limit=limit, is_current=is_current)
        except RuntimeError:
            return json.dumps({"status":"error", "found":0, "message":"政策检索失败，结论未核验，请稍后重试"}, ensure_ascii=False)
        if not results:
            return json.dumps({"found": 0, "message": "知识库中未检索到相关条文"}, ensure_ascii=False)

        # 截断超长 content（LLM context 有限）
        for r in results:
            if len(r["content"]) > 800:
                r["content"] = r["content"][:800] + "..."

        return json.dumps({"found": len(results), "results": results}, ensure_ascii=False, indent=2)

    TAX_POLICY_TOOL = StructuredTool.from_function(
        func=_tool_search_tax_policy,
        name="search_tax_policy",
        description=(
            "从本地税法知识库检索相关条文（高精度 RAG，多路召回+RRF 融合）。\n"
            "支持按税种过滤（增值税/企业所得税/个人所得税/印花税/附加税费），\n"
            "支持只搜现行有效文件。返回每条条文的标题、文号、发布机关、生效日期和内容片段。\n"
            "如果用户问 '根据XX法第X条'、'XX税政策是什么'，优先用此 Tool。"
        ),
        args_schema=TaxPolicyInput,
        return_direct=False,
    )
except ImportError:
    TAX_POLICY_TOOL = None

# ==============================================================
# 2. 联网搜索最新政策
# ==============================================================

_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
}


def _bing_search(query: str, limit: int = 5) -> list[dict]:
    """解析 Bing HTML，将无法识别的响应与明确的零结果区分。"""
    from bs4 import BeautifulSoup

    search_url = os.getenv("WEB_SEARCH_URL") or "https://cn.bing.com/search"
    resp = requests.get(search_url, params={"q": query, "count": limit},
                        headers=_SEARCH_HEADERS, timeout=(3, 6))
    resp.raise_for_status()

    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    entries = soup.select("li.b_algo")
    for li in entries:
        a = li.select_one("h2 a")
        if not a:
            continue
        raw_href = a.get("href", "")
        if not raw_href:
            continue
        href = urljoin(search_url, raw_href)
        # Bing 会把真实链接编码在 /ck/a?u=a1... 中，先解码再做官网过滤。
        if urlparse(href).hostname in ("www.bing.com", "cn.bing.com"):
            encoded = parse_qs(urlparse(href).query).get("u", [""])[0]
            if encoded.startswith("a1"):
                try:
                    payload = encoded[2:]
                    href = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode("utf-8")
                except (ValueError, UnicodeError):
                    continue
        if urlparse(href).scheme not in ("http", "https"):
            continue
        title = a.get_text(" ", strip=True)
        if not href or not title:
            continue
        cap = li.select_one(".b_caption p") or li.select_one("p")
        snippet = cap.get_text(" ", strip=True) if cap else ""
        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= limit:
            break
    if not results and not soup.select_one("#b_results .b_no, li.b_no"):
        raise ValueError("Bing HTML 未返回可识别的搜索结果页")
    return results


def _bing_rss_search(query: str, limit: int = 5) -> list[dict]:
    """HTML 结果缺失时读取 Bing RSS，保留明确失败日志。"""
    response = requests.get(os.getenv("WEB_SEARCH_URL") or "https://cn.bing.com/search",
                            params={"q": query, "format": "rss", "count": limit},
                            headers=_SEARCH_HEADERS, timeout=(3, 6))
    response.raise_for_status()
    root = ET.fromstring(response.text)
    channel = root.find("channel")
    if root.tag != "rss" or channel is None:
        raise ValueError("Bing RSS 响应缺少搜索频道")
    return [{"title": item.findtext("title", ""), "url": item.findtext("link", ""),
             "snippet": item.findtext("description", "")}
            for item in channel.findall("item")
            if urlparse(item.findtext("link", "")).scheme in ("http", "https")][:limit]


class PolicyOriginalUnavailable(ValueError):
    pass


def read_web_page(url: str, max_chars: int = 10000, *, official_only: bool = False,
                  require_article: bool = False, national_only: bool = True) -> dict:
    """读取公开网页正文；校验每次跳转、限制大小，失败返回可见的结构化错误。"""
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin
    from services.research_evidence import official_url
    max_chars = max(200, min(int(max_chars), 15000))
    started = time.monotonic()
    try:
        for _ in range(4):
            parsed = urlparse(url)
            if require_article and national_only and not _national_policy_site(url):
                raise ValueError("政策原文读取只接受全国性政策官网及其官方跳转")
            if official_only and not official_url(url):
                raise ValueError("政策原文读取只接受官方站点及其官方跳转")
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("仅支持无账号密码的 HTTP/HTTPS 网页链接")
            addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
            if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
                raise ValueError("只能读取公开互联网地址")
            with requests.get(url, headers=_SEARCH_HEADERS, timeout=(3, 6), stream=True, allow_redirects=False) as response:
                if response.status_code == 403 and parsed.scheme == "http":
                    url = parsed._replace(scheme="https").geturl()
                    continue
                if response.is_redirect:
                    url = urljoin(url, response.headers["Location"])
                    continue
                response.raise_for_status()
                mime = response.headers.get("Content-Type", "").lower()
                is_pdf = official_only and ("application/pdf" in mime or parsed.path.lower().endswith(".pdf"))
                if mime and not is_pdf and not any(t in mime for t in ("text/", "html", "xml")):
                    raise ValueError("此链接不是可读取的文本网页，请上传对应文档")
                content = bytearray()
                for piece in response.iter_content(16384):
                    content.extend(piece)
                    if len(content) > 2_000_000 or time.monotonic() - started > 12:
                        raise ValueError("网页过大或读取超过时间预算")
                metadata = {}
                if is_pdf:
                    from io import BytesIO
                    from pypdf import PdfReader
                    from pypdf.errors import PdfReadError
                    try:
                        pdf = PdfReader(BytesIO(content))
                        text = "\n".join(page.extract_text() or "" for page in pdf.pages[:12])
                    except (PdfReadError, ValueError) as exc:
                            raise PolicyOriginalUnavailable("官方 PDF 附件未取得可读取正文") from exc
                    title = next((line.strip() for line in text.splitlines() if len(line.strip()) >= 6), "官方政策附件")
                    metadata["truncated"] = len(pdf.pages) > 12
                else:
                    if require_article:
                        try:
                            html = content.decode("utf-8-sig")
                        except UnicodeDecodeError:
                            html = bytes(content)
                    else:
                        html = bytes(content)
                    soup = BeautifulSoup(html, "html.parser")
                    title = soup.title.get_text(" ", strip=True) if soup.title else parsed.hostname
                    for node in soup(["script", "style", "noscript", "nav", "footer", "header"]):
                        node.decompose()
                    if official_only:
                        title_tag = soup.select_one("meta[name='ArticleTitle'], meta[property='og:title']")
                        if title_tag and title_tag.get("content"):
                            title = title_tag["content"].strip()
                        published = soup.select_one("meta[name='PubDate'], meta[property='article:published_time']")
                        if published:
                            metadata["publish_date"] = published.get("content")
                        if require_article:
                            document_number = soup.select_one(".actfwzh") or soup.find(
                                ["h4", "h5"], string=re.compile(r"公告\s*\d{4}\s*年\s*第\s*\d+\s*号"))
                            if document_number and document_number.get_text(strip=True):
                                metadata["document_number"] = document_number.get_text(" ", strip=True)
                            generated = soup.select_one("meta[name='others']")
                            if (generated and metadata.get("publish_date") and "页面生成时间" in generated.get("content", "")
                                    and metadata.get("publish_date") in generated["content"]):
                                metadata.pop("publish_date")
                        for selector, key in ((".dfsj", "response_date"), (".arc_date", "document_date")):
                            label = soup.select_one(selector)
                            if label:
                                date = re.search(r"\d{4}[-年/]\d{1,2}[-月/]\d{1,2}日?", label.get_text())
                                if date:
                                    metadata[key] = date.group()
                    if require_article:
                        body = next((node for selector in (
                            ".arc_cont", ".TRS_Editor", "#fontzoom", "#zoom", "#artibody", ".article-content",
                            ".xxgk_content", ".Custom_UnionStyle", ".contentLeft", "article",
                        ) if (node := soup.select_one(selector)) is not None), None)
                        if body is None:
                            raise PolicyOriginalUnavailable("未找到政策正文，页面可能是栏目、验证页或动态加载页")
                    else:
                        body = (soup.select_one(".TRS_Editor, .contentLeft, #zoom, #artibody, .article-content")
                                if official_only else None) or soup.select_one("article, main, #content") or soup.body or soup
                    text = body.get_text("\n", strip=True)
                    if official_only:
                        metadata["attachments"] = list(dict.fromkeys(
                            link for tag in body.select("a[href]")
                            if official_url(link := urljoin(url, tag["href"])) and urlparse(link).path.lower().endswith(".pdf")
                        ))[:2]
                if official_only:
                    date = re.search(r"(?:答复时间|发布日期|发布时间)\s*[：:]?\s*(\d{4}[-年/]\d{1,2}[-月/]\d{1,2}日?)", text)
                    if date and not metadata.get("publish_date"):
                        metadata["publish_date"] = date.group(1)
                    number = re.search(r"(?m)^\s*(?:财政部\s*税务总局|国家税务总局|税务总局)\s*公告\s*\d{4}\s*年\s*第\s*\d+\s*号", text[:1300])
                    if number and (require_article or not metadata.get("document_number")):
                        metadata.setdefault("document_number", number.group().strip())
                        metadata["issuer"] = "财政部、税务总局" if number.group().strip().startswith("财政部") else "国家税务总局"
                    if require_article:
                        signed = re.search(r"((?:财政部[\s、]*税务总局|国家税务总局))\s*\n\s*(\d{4}年\s*\d{1,2}月\s*\d{1,2}日)\s*$", text)
                        if signed:
                            metadata["issuer"] = re.sub(r"\s+", "、", signed.group(1))
                            metadata.setdefault("document_date", signed.group(2))
                    effective = re.search(r"(?:本(?:公告|办法|规定|条例)|现予公布[，,]\s*)自\s*(\d{4}年\s*\d{1,2}月\s*\d{1,2}日)起施行", text)
                    if effective:
                        metadata["effective_date"] = re.sub(r"\s+", "", effective.group(1))
                if not text or (require_article and (len(text.strip()) < 30 or text.strip() == title)):
                    raise PolicyOriginalUnavailable("网页没有可读取正文，可能需要登录或动态加载")
                return {**metadata, "status": "success", "url": url, "title": title, "content": text[:max_chars],
                        "truncated": len(text) > max_chars or metadata.get("truncated", False)}
        raise ValueError("网页跳转次数过多")
    except (requests.RequestException, ValueError, OSError, KeyError) as exc:
        logger.warning("网页读取失败（%s）", type(exc).__name__)
        return {"status": "error", "url": url, "content": "",
                "error_code": "no_original" if isinstance(exc, PolicyOriginalUnavailable) else "read_failed",
                "message": str(exc) if isinstance(exc, ValueError) else "网页读取失败，请检查网络或链接"}


def search_web_policy(
    query: str,
    source: Literal["duckduckgo", "gov_cn", "all"] = "all",
    limit: int = 5,
) -> list[dict]:
    """
    联网搜索最新税务政策（纯函数入口）。

    :param query: 搜索关键词，如 "2024 年小规模纳税人增值税优惠"
    :param source: 搜索源：duckduckgo（底层实为 Bing 通用搜索）/ gov_cn=财政部/税务总局官网 / all=全部
    :param limit: 返回前 N 条
    :return: 列表，每项包含 title / url / snippet / source
    """
    query = query.strip()
    if not query or source not in ("duckduckgo", "gov_cn", "all"):
        return []
    limit = max(1, min(limit, 10))
    results: list[dict] = []

    # ---- 源 1：通用网页搜索（Bing，国内可访问）----
    if source in ("duckduckgo", "all"):
        _, candidates = _try_search([
            (_bing_search, query, "bing", lambda _: True),
            (_bing_rss_search, query, "bing", lambda _: True),
        ], limit)
        results.extend(candidates)

    # ---- 源 2：财政部官网 / 国家税务总局官网（定向）----
    if source in ("gov_cn", "all"):
        site_results = _search_gov_sites(query, limit=limit)
        results.extend(site_results)

    # 去重 + 截断
    seen_urls = set()
    deduped = []
    for r in results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped.append(r)
        if len(deduped) >= limit:
            break

    return deduped


def search_official_research(query: str, limit: int = 2, *, by_title: bool = False,
                             policy_only: bool = False, rank_titles: bool = False) -> list[dict]:
    """Use the public search endpoint used by the STA policy library page."""
    from bs4 import BeautifulSoup
    from services.research_evidence import official_url

    params = {"searchWord": query, "siteCode": "bm29000002", "indexCode": 1, "searchSiteName": "GSFFK",
              "column": "政策法规,政策解读,政策指引", "pageNum": 0, "pageSize": 5, "participleRule": 0}
    if by_title:
        params.update(column="政策法规", wordPlace=1, type=0)
    elif policy_only:
        params.update(column="政策法规", type=0)
    elif rank_titles:
        params["pageSize"] = 10
    response = requests.get("https://www.chinatax.gov.cn/search5/search/s",
                            headers={**_SEARCH_HEADERS, "Referer": "https://fgk.chinatax.gov.cn/"},
                            params=params, timeout=(3, 6), allow_redirects=False)
    response.raise_for_status()
    if response.status_code != 200:
        raise ValueError("税务总局搜索返回意外状态")
    rows = response.json()["searchResultAll"]["searchTotal"]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("税务总局搜索结果格式不正确")
    results = []
    for row in rows:
        url = row.get("url") or ""
        if not official_url(url):
            continue
        parsed = urlparse(url)
        host = "fgk.chinatax.gov.cn" if parsed.hostname == "www.chinatax.gov.cn" and parsed.path.startswith("/zcfgk/") else parsed.netloc
        url = parsed._replace(scheme="https", netloc=host, fragment="").geturl()
        if policy_only and (host != "fgk.chinatax.gov.cn" or not re.fullmatch(r"/zcfgk/c\d+/c\d+/content\.html", parsed.path)):
            continue
        if any(item["url"] == url for item in results):
            continue
        title = BeautifulSoup(row.get("title") or "", "html.parser").get_text()
        results.append({"title": title, "url": url, "source": "sta_policy_search", "rank": len(results) + 1})
    def title_key(title):
        return re.sub(r"[\s《》〈〉]", "", title)
    if by_title:
        results = [item for item in results if title_key(query) in title_key(item["title"])]
    elif policy_only:
        results.sort(key=lambda item: title_key(query) not in title_key(item["title"]))
    elif rank_titles:
        phrase = title_key(query)
        terms = {key for term in query.split() if len(key := title_key(term)) > 1}
        results.sort(key=lambda item: (phrase in (title := title_key(item["title"])),
                                       sum(term in title for term in terms)), reverse=True)
        for rank, item in enumerate(results, 1):
            item["rank"] = rank
    return results[:limit]

def _national_policy_site(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        return (parsed.scheme in ("https", "http") and not parsed.username and not parsed.password
                and parsed.port in (None, 80, 443)
                and (host in ("gov.cn", "www.gov.cn", "chinatax.gov.cn", "www.chinatax.gov.cn", "fgk.chinatax.gov.cn")
                     or host == "mof.gov.cn" or host.endswith(".mof.gov.cn")))
    except ValueError:
        return False


def _try_search(attempts, limit):
    succeeded = False
    for provider, query, source, accepts in attempts:
        try:
            candidates = provider(query, limit=limit)
            if not isinstance(candidates, list):
                raise ValueError("搜索结果格式不正确")
            results = {}
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError("搜索结果条目格式不正确")
                url = candidate.get("url", "")
                if url and accepts(url):
                    url = urlparse(url)._replace(fragment="").geturl()
                    results.setdefault(url, {**candidate, "url": url, "source": source})
            succeeded = True
            if results:
                return "success", list(results.values())[:limit]
        except (OSError, ET.ParseError, ValueError, KeyError, TypeError) as exc:
            logger.warning("政策搜索服务 %s 失败（%s）", source, type(exc).__name__)
    return ("no_results" if succeeded else "request_failed"), []


def _government_search_attempts(query):
    return [
        (_bing_search, f"(site:mof.gov.cn OR site:chinatax.gov.cn) {query}", "gov_cn", _national_policy_site),
        (_bing_rss_search, f"site:chinatax.gov.cn {query}", "gov_cn", _national_policy_site),
        (_bing_rss_search, f"site:mof.gov.cn {query}", "gov_cn", _national_policy_site),
    ]


def _search_gov_sites(query: str, limit: int = 5) -> list[dict]:
    return _try_search(_government_search_attempts(query), limit)[1]


def search_live_policy_candidates(query: str, limit: int = 3) -> tuple[str, list[dict]]:
    """聊天专用，官网法规库查询失败或无结果时使用 Bing 定向检索。"""
    query = " ".join(query.split())[:500]
    def search_sta(term, limit):
        return search_official_research(term, limit=limit, policy_only=True)

    # The official index favors policy subjects over a full conversational question.
    match = re.match(r"^([\u4e00-\u9fff]{4,24}?)(?:有什么|有哪些|有何|如何享受|能享受什么|适用哪些|适用什么)", query)
    subject = match.group(1) if match else None
    def search_subject(term, limit):
        rows = [row for row in search_sta(term, limit) if term in re.sub(r"\s+", "", row.get("title") or "")]
        if term == "小型微利企业" and "优惠" in query:
            # The administering and substantive announcements use different names for this subject.
            _, related = _try_search([
                (lambda term, limit: search_official_research(term, limit=limit, by_title=True),
                 "小微企业和个体工商户发展", "sta_policy_search", _national_policy_site)
            ], 1)
            rows = [*related, *rows]
        return rows[:limit]

    subject_attempt = [(search_subject, subject, "sta_policy_search", _national_policy_site)] if subject else []
    return _try_search([
        *subject_attempt,
        (search_sta, query, "sta_policy_search", _national_policy_site),
        *_government_search_attempts(query),
    ], limit)


LIVE_POLICY_STATUS = {
    "disabled": "本轮未开启联网检索。",
    "success": "已读取本次联网获取的官方政策正文。",
    "no_results": "搜索已完成，未找到匹配的官方政策。联网检索未成功，本次回答仅参考本地知识库。",
    "read_failed": "找到了官方页面，但未能读取政策正文。联网检索未成功，本次回答仅参考本地知识库。",
    "request_failed": "政策搜索服务请求失败。联网检索未成功，本次回答仅参考本地知识库。",
}


async def retrieve_live_policy(query, enabled, emit=None):
    import asyncio

    def status(text, **fields):
        if emit:
            emit("status", {"text": text, **fields})

    outcome, sources = "disabled", []
    if enabled:
        urls = re.findall(r'https?://[^\s<>"，。；）)]+', query)
        if urls:
            candidates = [{"url": url, "title": "用户提供的官方页面"} for url in dict.fromkeys(
                urlparse(url)._replace(fragment="").geturl() for url in urls if _national_policy_site(url)
            )][:3]
            outcome = "read_failed"
        else:
            status("正在搜索官方政策…", web_search_status="searching")
            outcome, candidates = await asyncio.to_thread(search_live_policy_candidates, query)
        if candidates:
            status("正在读取政策正文…", web_search_status="reading")
            pages = await asyncio.gather(*(asyncio.to_thread(
                read_web_page, candidate["url"], 3500, official_only=True, require_article=True
            ) for candidate in candidates))
            by_url = {}
            for candidate, page in zip(candidates, pages):
                url = page.get("url") or candidate["url"]
                if not _national_policy_site(url):
                    url = candidate["url"]
                url = urlparse(url)._replace(fragment="").geturl()
                host = urlparse(url).hostname
                title = page.get("title")
                read_ok = page.get("status") == "success" and bool(page.get("content"))
                if url in by_url and (by_url[url]["page_status"] == "success" or not read_ok):
                    continue
                by_url[url] = {
                    "title": title if title and title != host else candidate.get("title") or "官方政策",
                    "url": url,
                    "website": host,
                    "page_status": "success" if read_ok else "error",
                    "publish_date": _published_day(page.get("publish_date")),
                    "document_date": _published_day(page.get("document_date")),
                    "document_number": page.get("document_number"),
                    "issuer": page.get("issuer"),
                    "effective_date": _published_day(page.get("effective_date")),
                    "snippet": candidate.get("snippet", ""),
                    "content": page["content"] if read_ok else "",
                    "truncated": bool(page.get("truncated")),
                }
            sources = list(by_url.values())
            outcome = "success" if any(source["page_status"] == "success" for source in sources) else "read_failed"
    status(LIVE_POLICY_STATUS[outcome], web_search_status=outcome)
    return {"status": outcome, "sources": sources}


def _published_day(value):
    from datetime import date

    match = re.match(r"(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})(?!\d)", str(value or "").strip())
    if match:
        try:
            return date(*map(int, match.groups())).isoformat()
        except ValueError:
            pass
    return None

# ---- LangChain Tool 封装 ----

try:
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class WebSearchInput(BaseModel):
        """联网搜索 Tool 的参数 schema"""
        query: str = Field(min_length=1, max_length=500, description="要搜索的税务政策关键词，如 '2024 年小规模纳税人增值税减免政策'")
        source: Literal["all", "duckduckgo", "gov_cn"] = Field(
            default="all",
            description="搜索源：all=全部 / duckduckgo=通用网页搜索（Bing）/ gov_cn=只搜财政部和税务总局官网（更权威）",
        )
        limit: int = Field(default=5, ge=1, le=10, description="返回前 N 条，默认 5，最多 10")

    def _tool_search_web(
        query: str,
        source: str = "all",
        limit: int = 5,
    ) -> str:
        results = search_web_policy(query, source=source, limit=limit)
        if not results:
            return json.dumps({"found": 0, "message": "联网搜索未返回结果，请稍后重试或换关键词"}, ensure_ascii=False)
        return json.dumps({"found": len(results), "results": results}, ensure_ascii=False, indent=2)

    WEB_SEARCH_TOOL = StructuredTool.from_function(
        func=_tool_search_web,
        name="search_web_policy",
        description=(
            "联网搜索最新税务政策（实时获取，覆盖财政部/税务总局官网 + 通用搜索）。\n"
            "当用户问到 '最新政策' / '今年有没有新规定' / '2024 年优惠' 等时效性强的问题时必须用此 Tool。\n"
            "注意：本地知识库可能未收录最新文件，此 Tool 可作为补充。返回标题、URL 和内容摘要。"
        ),
        args_schema=WebSearchInput,
        return_direct=False,
    )
except ImportError:
    WEB_SEARCH_TOOL = None


class WebPageInput(BaseModel):
    url: str = Field(min_length=1, max_length=2048, description="需要读取正文的公开网页 URL")
    max_chars: int = Field(default=10000, ge=200, le=15000)


WEB_PAGE_TOOL = StructuredTool.from_function(
    func=lambda url, max_chars=10000: json.dumps(read_web_page(url, max_chars), ensure_ascii=False),
    name="read_web_page", description="读取网页正文，核对搜索摘要对应的政策原文；返回来源 URL 或明确错误。",
    args_schema=WebPageInput,
)

# ==============================================================
# 3. 统一导出（给 ReAct 模式用）
# ==============================================================

ALL_TOOLS = [t for t in [TAX_POLICY_TOOL, WEB_SEARCH_TOOL, WEB_PAGE_TOOL] if t is not None]
