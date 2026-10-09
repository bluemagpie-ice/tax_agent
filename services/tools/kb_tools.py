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
import xml.etree.ElementTree as ET
from copy import deepcopy
from urllib.parse import urlparse, parse_qs
from typing import Literal, Optional

import requests

from services.cache_utils import TTLCache
from services.security import current_tenant_id, document_session
from services.config_manager import ConfigManager

_search_cache = TTLCache(ttl_seconds=300, maxsize=128)

from services.cache_utils import TTLCache

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
) -> list[dict]:
    """
    从本地税务知识库检索相关条文（纯函数入口，Agent 内部强制调用）。

    :param query: 自然语言查询，如 "增值税进项抵扣"
    :param tax_type: 可选税种过滤，如 "vat" / "corporate_income" / "individual_income" / "general"
    :param limit: 返回前 N 条
    :param is_current: 可选，True=只搜现行有效，False=只搜已废止
    :return: 列表，每项包含 id / score / content / file_id 以及 title / document_number / issuer / effective_date / law_type / tax_type / is_current 等元数据
             注意：is_current 缺失时默认 False（不盲目认定为现行有效，避免已废止文件误导）
    """
    try:
        from knowledge.embedding import EmbeddingService, PersistentDocumentStore

        # 组装 Qdrant 过滤条件
        filters: dict = {}
        if tax_type:
            filters["tax_type"] = tax_type
        if is_current is not None:
            filters["is_current"] = is_current

        cache_key = (current_tenant_id(), document_session.get(), query.strip(), tax_type or "", limit, is_current,
                     ConfigManager().get_embedding_config()["model"])
        cached = _search_cache.get(cache_key)
        if cached is not None:
            logger.info(f"[search_tax_policy] 缓存命中 → {len(cached)} 条")
            return deepcopy(cached)

        store = PersistentDocumentStore()
        svc = EmbeddingService(store=store, use_qdrant=True, skip_rebuild=True)

        # 加一点检索增强（让 RRF 的 BM25 分支也受益）
        enhanced = f"税法 政策 条文 {query}"
        results = svc.search(enhanced, limit=limit, filters=filters or None)

        # 格式化输出
        formatted = []
        for r in results:
            formatted.append({
                "id": r.get("id", ""),
                "file_id": r.get("file_id", ""),
                "title": r.get("title", ""),
                "document_number": r.get("document_number", ""),
                "issuer": r.get("issuer", ""),
                "effective_date": r.get("effective_date", ""),
                "law_type": r.get("law_type", ""),
                "tax_type": r.get("tax_type", ""),
                "is_current": r.get("is_current", False),
                "score": r.get("score", 0),
                "content": r.get("content", ""),
            })
        _search_cache.set(cache_key, formatted)
        return deepcopy(formatted)

    except Exception as e:
        logger.error(f"[search_tax_policy] 检索失败: {e}", exc_info=True)
        return []

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
        results = search_tax_policy(query, tax_type=tax_type, limit=limit, is_current=is_current)
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
    """用 Bing 搜索并解析结果，返回 [{title, url, snippet}]。

    超时策略：连接 5s，读取 8s；读取超时不是整次请求的总时间上限。
    失败时返回空列表，不影响主流程。
    """
    from bs4 import BeautifulSoup

    search_url = os.getenv("WEB_SEARCH_URL") or "https://cn.bing.com/search"
    try:
        resp = requests.get(
            search_url,
            params={"q": query, "count": limit},
            headers=_SEARCH_HEADERS,
            timeout=(5, 8),
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        logger.warning(f"[Bing] 搜索超时: {search_url}")
        return []
    except requests.exceptions.ConnectionError:
        logger.warning(f"[Bing] 无法连接搜索服务，请检查网络")
        return []
    except requests.exceptions.RequestException as e:
        logger.warning(f"[Bing] 搜索请求失败: {e}")
        return []

    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a")
        if not a:
            continue
        href = a.get("href", "")
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
    return results


def _bing_rss_search(query: str, limit: int = 5) -> list[dict]:
    """HTML 结果缺失时读取 Bing RSS，保留明确失败日志。"""
    try:
        response = requests.get(os.getenv("WEB_SEARCH_URL") or "https://cn.bing.com/search",
                                params={"q": query, "format": "rss", "count": limit},
                                headers=_SEARCH_HEADERS, timeout=(3, 6))
        response.raise_for_status()
        root = ET.fromstring(response.text)
        return [{"title": item.findtext("title", ""), "url": item.findtext("link", ""),
                 "snippet": item.findtext("description", "")}
                for item in root.findall("./channel/item")
                if urlparse(item.findtext("link", "")).scheme in ("http", "https")][:limit]
    except (requests.RequestException, ET.ParseError) as exc:
        logger.warning("Bing RSS 检索失败（%s）", type(exc).__name__)
        return []


def read_web_page(url: str, max_chars: int = 10000) -> dict:
    """读取公开网页正文；校验每次跳转、限制大小，失败返回可见的结构化错误。"""
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin
    max_chars = max(200, min(int(max_chars), 15000))
    started = time.monotonic()
    try:
        for _ in range(4):
            parsed = urlparse(url)
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
                if mime and not any(t in mime for t in ("text/", "html", "xml")):
                    raise ValueError("此链接不是可读取的文本网页，请上传对应文档")
                content = bytearray()
                for piece in response.iter_content(16384):
                    content.extend(piece)
                    if len(content) > 2_000_000 or time.monotonic() - started > 12:
                        raise ValueError("网页过大或读取超过时间预算")
                soup = BeautifulSoup(bytes(content), "html.parser")
                title = soup.title.get_text(" ", strip=True) if soup.title else parsed.hostname
                for node in soup(["script", "style", "noscript", "nav", "footer", "header"]):
                    node.decompose()
                body = soup.select_one("article, main, .TRS_Editor, #zoom, #content") or soup.body or soup
                text = body.get_text("\n", strip=True)
                if not text:
                    raise ValueError("网页没有可读取正文，可能需要登录或动态加载")
                return {"status": "success", "url": url, "title": title, "content": text[:max_chars],
                        "truncated": len(text) > max_chars}
        raise ValueError("网页跳转次数过多")
    except (requests.RequestException, ValueError, OSError) as exc:
        logger.warning("网页读取失败（%s）", type(exc).__name__)
        return {"status": "error", "url": url, "content": "", "message": str(exc) if isinstance(exc, ValueError) else "网页读取失败，请检查网络或链接"}


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
        try:
            for r in (_bing_search(query, limit=limit) or _bing_rss_search(query, limit=limit)):
                results.append({
                    "title": r["title"],
                    "url": r["url"],
                    "snippet": r["snippet"],
                    "source": "bing",
                })
        except Exception as e:
            logger.warning(f"[search_web_policy] Bing 搜索失败: {e}")

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

def _search_gov_sites(query: str, limit: int = 5) -> list[dict]:
    """定向搜索财政部 / 国家税务总局官网。"""
    site_query = f"(site:mof.gov.cn OR site:chinatax.gov.cn) {query}"

    try:
        results = []
        for r in _bing_search(site_query, limit=limit * 2):
            host = (urlparse(r["url"]).hostname or "").lower()
            if any(host == domain or host.endswith("." + domain) for domain in ("mof.gov.cn", "chinatax.gov.cn")):
                results.append({
                    "title": r["title"],
                    "url": r["url"],
                    "snippet": r["snippet"],
                    "source": "gov_cn",
                })
        # 某些 Bing 页面忽略 OR 限定；没有官网结果时再尝试单站点检索。
        if not results:
            for domain in ("chinatax.gov.cn", "mof.gov.cn"):
                for r in _bing_rss_search(f"site:{domain} {query}", limit=10):
                    host = (urlparse(r["url"]).hostname or "").lower()
                    if host == domain or host.endswith("." + domain):
                        results.append({**r, "source": "gov_cn"})
                if results:
                    break
        return results[:limit]
    except Exception as e:
        logger.warning(f"[_search_gov_sites] 定向搜索失败: {e}")
        return []

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
