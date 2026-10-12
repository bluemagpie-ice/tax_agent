"""
政策自动更新
定期扫描政策源 -> 与已有库去重 -> 增量索引 -> 通知用户

核心流程：
    1. PolicySource   : 抽象"政策源"，支持本地目录 / HTTP 抓取 / 自定义扩展
    2. PolicyTracker  : 基于 SQLite 记录已入库政策，用于去重（基于 URL/文件名 + 内容哈希）
    3. PolicyUpdater  : 编排"扫描 → 去重 → 增量入库 → 通知"的完整流水线
    4. run_scheduler  : 使用 APScheduler 或后台线程启动定期任务

去重策略：
    - 一级：source_id + 外部标识（URL/文件名）完全匹配
    - 二级：内容 SHA256 哈希匹配（防止同一文件内容重复入库）

增量索引：
    - 复用 knowledge.kb.QdrantKnowledgeBase.ingest_file 完成解析→分块→向量化→入库
# todo:知识库查询上，假如政策条目相同，但是表达的文本形式不同，是否需要合并处理？
"""

from __future__ import annotations

import abc
import difflib
import hashlib
import json
import logging
import os
import sqlite3
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence
import re
import requests


logger = logging.getLogger(__name__)

# ============================================================ #
# 1. 政策条目与政策源（抽象层）
# ============================================================ #

@dataclass
class PolicyItem:
    """
    一条待入库的政策条目。

    Attributes:
        source_id : 来源标识（如 "local_dir_1" / "tax_gov_cn"）
        item_id   : 在来源中的唯一标识（URL / 文件名 / ID）
        title     : 政策标题
        content   : 政策正文或 Markdown 文本
        url       : 原始链接（可用于展示）
        published : 发布时间
        category  : 分类（如 增值税/企业所得税）
        file_path : 若为本地文件，记录物理路径
        metadata  : 附加元数据
    """
    source_id: str
    item_id: str
    title: str = ""
    content: str = ""
    url: str = ""
    published: Optional[datetime] = None
    category: str = ""
    file_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    published_date: str | None = None

    def __post_init__(self) -> None:
        # 发布日期不是文件修改时间；优先显式日期，其次正文标签和文件名。
        explicit = self.published_date or self.metadata.get("publish_date") or self.metadata.get("published_date")
        if not explicit and self.published:
            self.published_date = self.published.date().isoformat()
            return
        labelled = re.search(r"(?:发布日期|发布时间|发布于)\s*[:：]?\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})", self.content or "")
        filename = Path(self.file_path or self.item_id).name
        candidates = [explicit, labelled.group(1) if labelled else None, filename]
        for value in candidates:
            match = re.search(r"(?<!\d)(\d{4})[-_/年](\d{1,2})[-_/月](\d{1,2})(?!\d)", str(value or ""))
            if match:
                self.published = datetime(*map(int, match.groups()))
                self.published_date = self.published.date().isoformat()
                return
        if explicit:
            raise ValueError(f"无法解析政策发布日期：{explicit}")
        if self.published:
            self.published_date = self.published.date().isoformat()

    @property
    def content_hash(self) -> str:
        """内容哈希，用于二次去重"""
        return hashlib.sha256(
            (self.content or self.title or "").encode("utf-8")
        ).hexdigest()

    @property
    def unique_key(self) -> str:
        """一级去重键"""
        return f"{self.source_id}::{self.item_id}"


class PolicySource(abc.ABC):
    """
    政策源抽象基类。
    新的数据源（如爬取财政部官网、接 RSS 订阅）只需实现 scan()。
    """

    source_type: str = "base"

    @abc.abstractmethod
    def scan(self) -> list[PolicyItem]:
        """扫描数据源，返回若干条候选政策"""
        ...

# ---------------- 内置源：本地目录 ---------------- #
@dataclass
class LocalDirectorySource(PolicySource):
    """
    从本地目录扫描政策文件。
    支持的扩展名参考 knowledge/parser.py 中的 ParserFactory。
    """
    directory: str
    source_id: str = "local_dir"
    recursive: bool = True
    category: str = ""

    source_type: str = "local_directory"

    SCAN_EXTENSIONS = {
        ".txt", ".md", ".pdf", ".docx", ".xlsx", ".pptx", ".csv",
        ".jpg", ".jpeg", ".png",
    }

    def scan(self) -> list[PolicyItem]:
        self.last_errors = []
        if not os.path.isdir(self.directory):
            logger.warning(f"本地政策目录不存在：{self.directory}")
            return []

        pattern = "**/*" if self.recursive else "*"
        items: list[PolicyItem] = []
        for path in Path(self.directory).glob(pattern):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.SCAN_EXTENSIONS:
                continue
            try:
                stat = path.stat()
                if path.suffix.lower() in {".md", ".txt"}:
                    content = path.read_text(encoding="utf-8-sig")
                else:
                    from knowledge.parser import ParserFactory
                    content = ParserFactory.get_parser(str(path)).parse()
                if not content or not content.strip() or content.lstrip().startswith(
                    ("文件解析错误", "解析错误", "PDF 错误", "OCR 错误", "OCR 依赖错误", "OCR 初始化错误", "依赖错误", "错误：")
                ):
                    raise ValueError(f"政策文件未解析出有效正文：{path.name}")
                metadata = {"size": stat.st_size, "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()}
                # 独立验收脚本生成的模拟政策必须隔离，不能作为所有用户的政策依据。
                if content.startswith("<!-- policy-update-demo -->"):
                    metadata.update(source_type="user_upload", owner_id="policy_update_demo", scope="global")
                heading = re.search(r"^ {0,3}#[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", content, re.MULTILINE) if path.suffix.lower() == ".md" else None
                items.append(PolicyItem(
                    source_id=self.source_id,
                    item_id=str(path.resolve()),
                    title=heading.group(1).strip() if heading else path.stem,
                    content=content,
                    file_path=str(path.resolve()),
                    category=self.category,
                    metadata=metadata,
                ))
            except Exception as e:
                self.last_errors.append(f"{path}: {e}")
                logger.error("政策文件扫描失败：%s", self.last_errors[-1])
        logger.info(f"[LocalDirectorySource] 扫描到 {len(items)} 个文件")
        return items

# ---------------- 内置源：HTTP 抓取 ---------------- #
@dataclass
class HttpCrawlSource(PolicySource):
    """
    通过 HTTP 抓取的政策源（示例实现）。
    子类或 config 中可自定义 fetch / parse 逻辑。
    """
    source_id: str
    urls: Sequence[str]
    category: str = ""
    fetcher: Optional[Callable[[str], tuple[str, str]]] = None
    # fetcher(url) -> (title, markdown_text)

    source_type: str = "http"

    def scan(self) -> list[PolicyItem]:
        items: list[PolicyItem] = []
        self.last_errors = []
        for url in self.urls:
            try:
                if self.fetcher:
                    title, content = self.fetcher(url)
                else:
                    title, content = self._default_fetch(url)
                if not content or not content.strip():
                    raise ValueError("正文为空")
                items.append(PolicyItem(
                    source_id=self.source_id,
                    item_id=url,
                    title=title or url,
                    content=content,
                    url=url,
                    category=self.category,
                ))
            except Exception as e:
                logger.error(f"[HttpCrawlSource] 抓取失败 {url}: {e}")
                self.last_errors.append(f"{url}: {e}")
        logger.info(f"[HttpCrawlSource] 抓取到 {len(items)} 条政策")
        return items

    @staticmethod
    def _default_fetch(url: str) -> tuple[str, str]:
        """
        默认抓取实现（纯文本）。
        生产环境建议使用 requests + BeautifulSoup / Readability 解析。
        """
        source = TaxGovCnSource(url, request_timeout=15)
        item = source._fetch_policy_page(url)
        return item.title, item.content

# ============================================================ #
# 2. 政策入库追踪器（去重持久化）
# ============================================================ #

class TaxGovCnSource(PolicySource):
    """国家税务总局法规库爬取源

    自动爬取法规列表页，解析每一条法规的标题、文号、发布日期、正文，
    去重后返回 PolicyItem 列表。

    典型用法::

        source = TaxGovCnSource(
            listing_url="https://fgk.chinatax.gov.cn/zcfgk/c100027/list.html",
            source_id="chinatax_vat",
            category="增值税",
            max_pages=3,
        )
    """

    source_type = "tax_gov_cn"

    def __init__(
        self,
        listing_url: str,
        source_id: str = "chinatax",
        category: str = "",
        max_pages: int = 5,
        link_selector: str = "a[href*='content.html']",
        title_selector: str = "h3, .title, .xxgk_title",
        date_selector: str = ".info, .time, .date, .pubinfo",
        content_selector: str = "#fontzoom, .content, .xxgk_content, .Custom_UnionStyle, .main-content",
        encoding: str = "utf-8",
        request_timeout: int = 30,
        max_items: int | None = None,
    ):
        self.listing_url = listing_url
        self.source_id = source_id
        self.category = category
        self.max_pages = max_pages
        self.link_selector = link_selector
        self.title_selector = title_selector
        self.date_selector = date_selector
        self.content_selector = content_selector
        self.encoding = encoding
        self.request_timeout = request_timeout
        self.max_items = max_items
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })

    def scan(self) -> list:
        """爬取法规列表页及其分页，返回 PolicyItem 列表"""
        from bs4 import BeautifulSoup

        items: list = []
        self.last_errors = []
        seen_urls: set = set()

        for page in range(self.max_pages):
            url = self._build_page_url(page)
            logger.info(f"[TaxGovCnSource] 抓取列表页 {page + 1}: {url}")
            try:
                resp = self._session.get(url, timeout=self.request_timeout)
                resp.raise_for_status()
                resp.encoding = self._detect_encoding(resp)
                soup = BeautifulSoup(resp.text, "lxml")
            except Exception as e:
                logger.warning(f"[TaxGovCnSource] 列表页抓取失败: {url} — {e}")
                self.last_errors.append(f"列表页 {url}: {e}")
                break

            links_found = 0
            try:
                links = self._select_links(soup)
            except ValueError as e:
                self.last_errors.append(f"列表页 {url}: {e}")
                logger.error("%s", self.last_errors[-1])
                break
            for a_tag in links:
                if self.max_items is not None and len(seen_urls) >= self.max_items:
                    break
                href = a_tag.get("href", "")
                if not href:
                    continue
                full_url = self._resolve_url(href)
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)
                links_found += 1

                try:
                    policy_item = self._fetch_policy_page(full_url)
                    if policy_item:
                        items.append(policy_item)
                except Exception as e:
                    logger.warning(f"[TaxGovCnSource] 正文抓取失败: {full_url} — {e}")
                    self.last_errors.append(f"正文页 {full_url}: {e}")

            logger.info(f"[TaxGovCnSource] 第 {page + 1} 页发现 {links_found} 个链接")
            if self.max_items is not None and len(seen_urls) >= self.max_items:
                break
            if links_found == 0:
                break

        logger.info(f"[TaxGovCnSource] 共抓取 {len(items)} 条政策")
        return items

    def _select_links(self, soup) -> list:
        links = soup.select(self.link_selector)
        if not links:
            links = soup.select("a[href*='/content.html']")
            if links:
                logger.warning("link_selector=%r 失效，已适配 a[href*='/content.html']", self.link_selector)
        if not links:
            raise ValueError(f"link_selector={self.link_selector!r} 匹配不到政策链接（已尝试法规库选择器）")
        return links

    def _build_page_url(self, page: int) -> str:
        if page == 0:
            return self.listing_url
        base, _, rest = self.listing_url.rpartition("/")
        if "index" in rest or "list" in rest:
            name, dot, ext = rest.partition(".")
            return f"{base}/{name}_{page}{dot}{ext}"
        return f"{base}/index_{page}.html"

    def _resolve_url(self, href: str) -> str:
        from urllib.parse import urljoin
        return urljoin(self.listing_url, href)

    def _detect_encoding(self, resp) -> str:
        ct = resp.headers.get("Content-Type", "")
        if "charset=" in ct:
            return ct.split("charset=")[-1].split(";")[0].strip()
        return self.encoding

    def _fetch_policy_page(self, url: str) -> PolicyItem | None:
        from bs4 import BeautifulSoup

        resp = self._session.get(url, timeout=self.request_timeout)
        resp.raise_for_status()
        resp.encoding = self._detect_encoding(resp)
        soup = BeautifulSoup(resp.text, "lxml")

        title_el = soup.select_one("meta[name='ArticleTitle']") or soup.select_one(self.title_selector)
        if not title_el:
            title_el = soup.select_one("meta[name='ArticleTitle'], h1, h3, .article-title")
            if title_el:
                logger.warning("title_selector=%r 失效，已适配正文标题", self.title_selector)
        title = (title_el.get("content", "") if title_el and title_el.name == "meta"
                 else title_el.get_text(strip=True) if title_el else "")
        if not title:
            raise ValueError(f"title_selector={self.title_selector!r} 匹配不到非空标题：{url}")

        date_el = soup.select_one(self.date_selector)
        date_text = date_el.get_text(strip=True) if date_el else ""
        date_match = re.search(r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})", date_text)
        date_meta = soup.select_one("meta[name='PubDate']")
        publish_date = date_meta.get("content", "") if date_meta else date_match.group(1) if date_match else ""

        content_el = soup.select_one(self.content_selector)
        if not content_el:
            content_el = soup.select_one(".article, .TRS_Editor, #zoom, #artibody")
            if content_el:
                logger.warning("content_selector=%r 失效，已适配正文容器", self.content_selector)
        if content_el:
            article = content_el.select_one(".article, .TRS_Editor, #zoom, #artibody")
            if article:
                content_el = article
            for unwanted in content_el.select("script, style, noscript"):
                unwanted.decompose()
        content_text = content_el.get_text("\n", strip=True) if content_el else ""

        if not content_text:
            raise ValueError(f"content_selector={self.content_selector!r} 匹配不到非空正文：{url}")

        item_id = self._extract_item_id(url)

        return PolicyItem(
            source_id=self.source_id,
            item_id=item_id,
            title=title or item_id,
            content=content_text,
            url=url,
            category=self.category,
            metadata={"url": url, "publish_date": publish_date},
        )

    @staticmethod
    def _extract_item_id(url: str) -> str:
        m = re.search(r"/([a-z]\d+)/content\.html", url, re.IGNORECASE)
        if m:
            return m.group(1)
        return hashlib.md5(url.encode()).hexdigest()[:12]


class PolicyTracker:
    """
    使用 SQLite 记录已入库政策，用于去重。

    表结构：
        policy_record(
            unique_key TEXT PRIMARY KEY,   -- source_id::item_id
            source_id   TEXT,
            item_id     TEXT,
            title       TEXT,
            content_hash TEXT,
            status      TEXT,              -- pending / ingested / failed
            file_id     TEXT,              -- 入库后的 file_id
            created_at  TEXT,
        )
    """

    def __init__(self, db_path: str):
        self.db_path = db_path # 数据库文件路径
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS policy_record (
                    unique_key            TEXT PRIMARY KEY,
                    source_id             TEXT NOT NULL,
                    item_id               TEXT NOT NULL,
                    title                 TEXT,
                    content_hash          TEXT,
                    previous_content_hash TEXT,
                    content_snapshot      TEXT,
                    previous_content_snapshot TEXT,
                    published_date        TEXT,
                    source_url            TEXT,
                    source_name           TEXT,
                    status                TEXT NOT NULL DEFAULT 'pending',
                    file_id               TEXT,
                    created_at            TEXT NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS update_report (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at  TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    scanned     INTEGER DEFAULT 0,
                    new_count   INTEGER DEFAULT 0,
                    ingested    INTEGER DEFAULT 0,
                    duplicated  INTEGER DEFAULT 0,
                    failed      INTEGER DEFAULT 0,
                    errors      TEXT DEFAULT '[]',
                    created_at  TEXT NOT NULL
                )
            """)
            report_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(update_report)")}
            if "changed" not in report_columns:
                self._conn.execute("ALTER TABLE update_report ADD COLUMN changed INTEGER DEFAULT 0")
            if "filtered" not in report_columns:
                self._conn.execute("ALTER TABLE update_report ADD COLUMN filtered INTEGER DEFAULT 0")
            policy_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(policy_record)")}
            # SQLite 不支持 ADD COLUMN IF NOT EXISTS；检查列后再 ALTER，重复启动可安全迁移。
            for column in ("previous_content_hash", "content_snapshot", "previous_content_snapshot",
                           "published_date", "source_url", "source_name"):
                if column not in policy_columns:
                    self._conn.execute(f"ALTER TABLE policy_record ADD COLUMN {column} TEXT")
            self._conn.commit()

    # ---------- 查询 ---------- #
    def exists(self, item: PolicyItem) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM policy_record WHERE unique_key = ? AND content_hash = ? AND status = 'ingested'",
                (item.unique_key, item.content_hash),
            ).fetchone()
            if row:
                return True
            # 二级：内容哈希
            if item.content_hash:
                row2 = self._conn.execute(
                    "SELECT 1 FROM policy_record WHERE content_hash = ? AND status = 'ingested'",
                    (item.content_hash,),
                ).fetchone()
                return row2 is not None
            return False


    def get_pending(self, limit: int = 500) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM policy_record WHERE status='pending' LIMIT ?",
                (limit,),
            ).fetchall()

    # ---------- 写入 ---------- #
    def upsert_pending(self, item: PolicyItem, content_text: str = "") -> tuple[bool, bool]:
        """若已存在且内容未变化则跳过；若内容变化则标记为 changed 并仍返回需要更新。

        Returns (is_new, is_changed)
        """
        with self._lock:
            if not item.unique_key or not item.content_hash:
                return False, False
            key = item.unique_key
            now = datetime.now().isoformat(timespec="seconds")
            snapshot = content_text or item.content or ""

            existing = self._conn.execute(
                "SELECT * FROM policy_record WHERE unique_key = ?", (key,)
            ).fetchone()

            if existing is None:
                if self._conn.execute("SELECT 1 FROM policy_record WHERE content_hash=? AND status='ingested'",
                                      (item.content_hash,)).fetchone():
                    return False, False
                self._conn.execute(
                    """INSERT INTO policy_record
                       (unique_key, source_id, item_id, title, content_hash, status, content_snapshot, created_at,
                        published_date, source_url, source_name)
                       VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
                    (key, item.source_id, item.item_id, item.title, item.content_hash, snapshot, now,
                     item.published_date, item.url or item.metadata.get("url", ""),
                     item.metadata.get("source_name") or item.metadata.get("issuer") or item.category),
                )
                self._conn.commit()
                return True, False

            old_hash = existing["content_hash"]
            # 同正文重扫也补齐历史元数据，不改变哈希、旧快照或检测时间。
            self._conn.execute(
                """UPDATE policy_record SET title=COALESCE(NULLIF(?, ''), title),
                   published_date=COALESCE(?, published_date), source_url=COALESCE(NULLIF(?, ''), source_url),
                   source_name=COALESCE(NULLIF(?, ''), source_name) WHERE unique_key=?""",
                (item.title, item.published_date, item.url or item.metadata.get("url", ""),
                 item.metadata.get("source_name") or item.metadata.get("issuer") or item.category, key),
            )
            if old_hash == item.content_hash:
                if existing["status"] == "ingested":
                    self._conn.commit()
                    return False, False
                self._conn.execute("UPDATE policy_record SET status='pending' WHERE unique_key=?", (key,))
                self._conn.commit()
                return True, bool(existing["previous_content_hash"])

            # 重试失败版本时仍与最后成功版本比较，不能把失败草稿当作旧政策。
            previous_hash = old_hash
            previous_snapshot = existing["content_snapshot"]
            if existing["status"] != "ingested" and existing["previous_content_hash"]:
                previous_hash = existing["previous_content_hash"]
                previous_snapshot = existing["previous_content_snapshot"]
            self._conn.execute(
                """UPDATE policy_record
                   SET title = ?, content_hash = ?, previous_content_hash = ?,
                       content_snapshot = ?, previous_content_snapshot = ?, status = 'pending', created_at = ?
                   WHERE unique_key = ?""",
                (item.title, item.content_hash, previous_hash, snapshot, previous_snapshot, now, key),
            )
            self._conn.commit()
            return True, True

    def mark_ingested(self, unique_key: str, file_id: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE policy_record SET status='ingested', file_id=? WHERE unique_key=?",
                (file_id, unique_key),
            )
            self._conn.commit()

    def mark_failed(self, unique_key: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE policy_record SET status='failed' WHERE unique_key=?",
                (unique_key,),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------- 报告持久化 ---------- #
    def save_report(self, report_dict: dict) -> int:
        """持久化一份更新报告，返回自增 ID"""
        with self._lock:
            now = datetime.now().isoformat(timespec="seconds")
            cursor = self._conn.execute(
                """INSERT INTO update_report
                   (started_at, finished_at, scanned, new_count, ingested, duplicated, failed, errors, created_at, changed, filtered)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report_dict.get("started_at", now),
                    report_dict.get("finished_at", now),
                    report_dict.get("scanned", 0),
                    report_dict.get("new", 0),
                    report_dict.get("ingested", 0),
                    report_dict.get("duplicated", 0),
                    report_dict.get("failed", 0),
                    json.dumps(report_dict.get("errors", []) or [], ensure_ascii=False),
                    now,
                    report_dict.get("changed", 0),
                    report_dict.get("filtered", 0),
                ),
            )
            self._conn.commit()
            return cursor.lastrowid

    def get_reports(self, limit: int = 20) -> list[dict]:
        """获取历史更新报告列表（按时间倒序）"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM update_report ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "started_at": row["started_at"],
                    "finished_at": row["finished_at"],
                    "scanned": row["scanned"],
                    "new": row["new_count"],
                    "changed": row["changed"],
                    "filtered": row["filtered"],
                    "ingested": row["ingested"],
                    "duplicated": row["duplicated"],
                    "failed": row["failed"],
                    "errors": json.loads(row["errors"] or "[]"),
                }
                for row in rows
            ]

    def get_pending_count(self) -> int:
        """获取待入库的政策数量"""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM policy_record WHERE status='pending'"
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def get_changes(self, limit: int = 20) -> list[dict]:
        """获取内容发生变更的政策列表（previous_content_hash IS NOT NULL）"""
        with self._lock:
            rows = self._conn.execute(
                """SELECT unique_key, source_id, item_id, title,
                          previous_content_hash, content_hash, content_snapshot, previous_content_snapshot, created_at,
                          published_date, source_url, source_name
                   FROM policy_record
                   WHERE previous_content_hash IS NOT NULL
                   ORDER BY created_at DESC LIMIT ?""",
                (max(1, int(limit)),),
            ).fetchall()
            source_names = {"local_policies": "本地政策库", "local_dir": "本地政策库",
                            "chinatax": "国家税务总局", "mof": "财政部"}
            changes = []
            for row in rows:
                old_text, new_text = row["previous_content_snapshot"], row["content_snapshot"] or ""
                if old_text is None:
                    summary = "历史版本缺少旧正文，无法生成变更摘要。"
                else:
                    lines = list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), n=0, lineterm=""))[2:]
                    added = [line[1:].strip() for line in lines if line.startswith("+") and line[1:].strip()]
                    removed = [line[1:].strip() for line in lines if line.startswith("-") and line[1:].strip()]
                    summary = re.sub(r"\s+", " ", "；".join(added) if added else "删除：" + "；".join(removed) if removed else "仅空白或格式变更。")
                published_date = row["published_date"]
                if not published_date:
                    try:
                        published_date = PolicyItem(row["source_id"], row["item_id"], content=new_text).published_date
                    except ValueError as exc:
                        logger.warning("旧政策发布日期无效，显示未知日期：%s (%s)", row["unique_key"], exc)
                heading = re.search(r"^ {0,3}#[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", new_text, re.MULTILINE) if row["item_id"].lower().endswith(".md") else None
                title = row["title"]
                if heading and (not title or title == Path(row["item_id"]).stem):
                    title = heading.group(1).strip()
                url = row["source_url"] or (row["item_id"] if row["item_id"].startswith(("https://", "http://")) else "")
                changes.append({
                    "unique_key": row["unique_key"],
                    "source_id": row["source_id"],
                    "item_id": row["item_id"],
                    "title": title,
                    "published_date": published_date,
                    "source_name": source_names.get(row["source_id"]) or row["source_name"] or row["source_id"],
                    "url": url,
                    "summary": summary[:99] + "…" if len(summary) > 100 else summary,
                    "old_hash": row["previous_content_hash"],
                    "new_hash": row["content_hash"],
                    "snapshot": (row["content_snapshot"] or "")[:500],
                    "updated_at": row["created_at"],
                })
            return changes

    def get_change_diff(self, unique_key: str | None = None, source_id: str | None = None, item_id: str | None = None) -> dict | None:
        """生成某条政策的文字级 diff"""
        with self._lock:
            if unique_key:
                row = self._conn.execute(
                    "SELECT * FROM policy_record WHERE unique_key = ?", (unique_key,)
                ).fetchone()
            elif source_id and item_id:
                row = self._conn.execute(
                    "SELECT * FROM policy_record WHERE source_id = ? AND item_id = ?",
                    (source_id, item_id),
                ).fetchone()
            else:
                return None

            if not row or not row["previous_content_hash"]:
                return None

            old_text = row["previous_content_snapshot"]
            new_text = row["content_snapshot"] or ""
            diff_lines = list(difflib.unified_diff(
                (old_text or "").splitlines(),
                new_text.splitlines(),
                fromfile=f"{row['source_id']}/{row['item_id']} (旧)",
                tofile=f"{row['source_id']}/{row['item_id']} (新)",
                lineterm="",
            ))
            return {
                "unique_key": row["unique_key"],
                "title": row["title"],
                "old_hash": row["previous_content_hash"],
                "new_hash": row["content_hash"],
                "old_snapshot_available": old_text is not None,
                "status": row["status"],
                "diff": "\n".join(diff_lines) if old_text is not None else "历史版本未保存旧正文，无法还原文本差异；后续变更将保存完整快照。",
            }

@dataclass
class UpdateReport:
    """一次更新任务的结果摘要"""
    started_at: str
    finished_at: str
    scanned: int = 0
    new: int = 0
    ingested: int = 0
    duplicated: int = 0
    changed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    filtered: int = 0

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "scanned": self.scanned,
            "new": self.new,
            "ingested": self.ingested,
            "duplicated": self.duplicated,
            "changed": self.changed,
            "failed": self.failed,
            "errors": self.errors,
            "filtered": self.filtered,
        }

class PolicyUpdater:
    """
    定期扫描政策源 -> 与已有库去重 -> 增量索引 -> 通知用户。

    使用示例：
        updater = PolicyUpdater(
            tracker_db="data/policy_tracker.sqlite3",
            kb_work_dir="data/kb",
            kb_id="tax_policy",
        )
        updater.add_source(LocalDirectorySource(directory="data/policies"))
        report = updater.run_once()
        updater.start_scheduler(interval_seconds=3600)
    """

    def __init__(
        self,
        tracker_db: str = "data/policy_tracker.sqlite3",
        kb_work_dir: str = "data/kb",
        kb_id: str = "tax_policy",
        notify_hooks: Optional[list[Callable[[UpdateReport], None]]] = None,
        chunk_preset_id: str = "laws",
        since_date: str | None = None,
    ):
        self.since_date = datetime.strptime(since_date, "%Y-%m-%d").date() if since_date else None
        self.tracker = PolicyTracker(tracker_db)
        self.kb_work_dir = kb_work_dir
        self.kb_id = kb_id
        self.chunk_preset_id = chunk_preset_id
        self.notify_hooks: list[Callable[[UpdateReport], None]] = notify_hooks or []
        self._sources: list[PolicySource] = []

        # 调度器相关
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._run_lock = threading.Lock()
        self._schedule_wakeup = threading.Event()
        self._interval_seconds = 6 * 3600
        self._progress = {"stage": "idle", "source": "", "completed": 0, "total": 0}

    # ---------------- 配置 ---------------- #
    def add_source(self, source: PolicySource) -> None:
        if any(getattr(src, "source_id", "") == getattr(source, "source_id", "") for src in self._sources):
            raise ValueError("来源 ID 已存在，请使用其他 ID")
        self._sources.append(source)
        logger.info(f"[PolicyUpdater] 添加政策源: {getattr(source, 'source_id', source.__class__.__name__)}")

    def remove_source(self, source_id: str) -> bool:
        """按 source_id 移除政策源，返回是否成功"""
        for i, src in enumerate(self._sources):
            if getattr(src, "source_id", "") == source_id:
                self._sources.pop(i)
                logger.info(f"[PolicyUpdater] 移除政策源: {source_id}")
                return True
        return False

    def add_notify_hook(self, hook: Callable[[UpdateReport], None]) -> None:
        self.notify_hooks.append(hook)

    # ---------------- 核心流程 ---------------- #
    def run_once(self) -> UpdateReport:
        """执行一次完整更新流程"""
        with self._run_lock:
            started_at = datetime.now().isoformat(timespec="seconds")
            report = UpdateReport(started_at=started_at, finished_at="")

            if not self._sources:
                logger.warning("未配置任何政策源，跳过本次更新")
                report.finished_at = datetime.now().isoformat(timespec="seconds")
                self.tracker.save_report(report.to_dict())
                self._notify(report)
                return report

            # Step 1: 扫描所有源
            all_items: list[PolicyItem] = []
            for source in list(self._sources):
                self._progress = {"stage": "scanning", "source": source.source_id, "completed": 0, "total": 0}
                try:
                    all_items.extend(source.scan())
                    for error in getattr(source, "last_errors", []):
                        report.errors.append(f"scan[{source.source_type}]: {error}")
                        report.failed += 1
                except Exception as e:
                    logger.error(f"扫描 {source.source_type} 失败: {e}")
                    report.errors.append(f"scan[{source.source_type}]: {e}")
                    report.failed += 1
            report.scanned = len(all_items)
            logger.info(f"共扫描到 {report.scanned} 条候选政策")

            # Step 2: 去重 + 记录 pending
            pending_items: list[PolicyItem] = []
            seen_keys, seen_hashes = set(), set()
            for item in all_items:
                if self.since_date and (not item.published or item.published.date() < self.since_date):
                    report.filtered += 1
                    if not item.published:
                        report.errors.append(f"{item.unique_key}: 缺少发布日期，无法确认时间窗口，未入库")
                    continue
                if item.unique_key in seen_keys or item.content_hash in seen_hashes:
                    report.duplicated += 1
                    continue
                seen_keys.add(item.unique_key)
                seen_hashes.add(item.content_hash)
                content_text = item.content or ""
                is_new, is_changed = self.tracker.upsert_pending(item, content_text)
                if is_new:
                    pending_items.append(item)
                    if is_changed:
                        report.changed += 1
                    else:
                        report.new += 1
                else:
                    report.duplicated += 1

            logger.info(f"去重后新增 {report.new} 条，变更 {report.changed} 条，重复 {report.duplicated} 条")

            # Step 3: 增量入库
            for index, item in enumerate(pending_items):
                self._progress = {"stage": "ingesting", "source": item.source_id,
                                  "completed": index, "total": len(pending_items)}
                try:
                    result = self._ingest_item(item)
                    if result.get("status") != "success" or result.get("chunk_count", 0) < 1:
                        raise RuntimeError(result.get("message") or "政策入库未返回有效分块")
                    self.tracker.mark_ingested(item.unique_key, file_id=item.metadata.get("file_id", ""))
                    report.ingested += 1
                except Exception as e:
                    logger.error(f"入库失败 {item.unique_key}: {e}")
                    self.tracker.mark_failed(item.unique_key)
                    report.failed += 1
                    report.errors.append(f"{item.unique_key}: {e}")

            report.finished_at = datetime.now().isoformat(timespec="seconds")
            logger.info(
                f"[PolicyUpdater] 完成：扫描 {report.scanned}，新增 {report.new}，"
                f"变更 {report.changed}，入库 {report.ingested}，重复 {report.duplicated}，失败 {report.failed}"
            )

            # Step 4: 持久化报告
            try:
                self.tracker.save_report(report.to_dict())
            except Exception as e:
                logger.error(f"[PolicyUpdater] 保存报告失败: {e}")
                report.errors.append(f"save_report: {e}")

            # Step 5: 通知
            self._notify(report)
            self._progress = {"stage": "idle", "source": "", "completed": len(pending_items), "total": len(pending_items)}
            return report

    # ---------------- 入库实现 ---------------- #
    def _ingest_item(self, item: PolicyItem) -> dict:
        """
        将单条 PolicyItem 增量入库。

        策略：
        - 若有 file_path 则直接走 kb.ingest_file（parser→chunk→embedding→qdrant）
        - 若仅有 content（如 HTTP 抓取），先写入临时文件再入库
        """
        from knowledge.kb import KnowledgeBaseFactory

        kb = KnowledgeBaseFactory.create("qdrant", self.kb_work_dir, kb_id=self.kb_id)
        if kb._ensure_embedding().qdrant is None:
            raise RuntimeError("Qdrant 不可用，政策未入库；已保留失败记录，下次扫描可重试")

        if item.file_path and os.path.exists(item.file_path) and not item.content:
            return self._ingest_file(item, kb)

        # 文本型：写入临时 .md 文件再入库
        if item.content:
            temp_dir = os.path.join(self.kb_work_dir, "_temp_policy")
            os.makedirs(temp_dir, exist_ok=True)
            safe_title = _safe_filename(item.title or item.item_id)
            temp_path = os.path.join(temp_dir, f"{item.content_hash[:16]}_{safe_title}.md")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(item.content)

            result = self._ingest_file(item, kb, override_path=temp_path)
            # 清理临时文件
            try:
                os.remove(temp_path)
            except OSError as exc:
                logger.warning("政策临时文件清理失败（%s）：%s", type(exc).__name__, temp_path)
            return result

        raise ValueError(f"PolicyItem 缺少 file_path 和 content，无法入库: {item.unique_key}")

    def _ingest_file(
        self,
        item: PolicyItem,
        kb,
        override_path: Optional[str] = None,
    ) -> dict:
        """
        通过 QdrantKnowledgeBase.ingest_file 入库。
        注意 ingest_file 是 async，这里用 asyncio 驱动。
        """
        import asyncio

        file_path = override_path or item.file_path
        file_id = "policy_" + hashlib.sha256(f"{item.unique_key}\0{item.content_hash}".encode("utf-8")).hexdigest()[:32]
        params = {key: item.metadata[key] for key in
                  ("tax_type", "document_number", "issuer", "effective_date", "is_current", "owner_id", "session_id")
                  if key in item.metadata}
        params.update(file_id=file_id, chunk_preset_id=self.chunk_preset_id, title=item.title,
                      source_type=item.metadata.get("source_type", "default"),
                      scope=item.metadata.get("scope", "system"), file_name=Path(item.file_path or file_path).name)

        # 兼容无 running loop 的场景
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                result = ex.submit(asyncio.run, kb.ingest_file(file_path, **params)).result(timeout=600)
        else:
            result = asyncio.run(kb.ingest_file(file_path, **params))

        if result.get("status") != "success" or result.get("chunk_count", 0) < 1:
            raise RuntimeError(result.get("message") or "政策入库未返回成功或有效分块")

        item.metadata["file_id"] = file_id
        return result

    # ---------------- 通知 ---------------- #
    def _notify(self, report: UpdateReport) -> None:
        for hook in self.notify_hooks:
            try:
                hook(report)
            except Exception as e:
                logger.error(f"通知钩子执行失败: {e}")
                report.errors.append(f"notify[{getattr(hook, '__name__', 'hook')}]: {e}")

    # ---------------- 查询接口 ---------------- #
    def get_sources_info(self) -> list[dict]:
        """返回已配置的政策源信息"""
        result = []
        for src in self._sources:
            info = {
                "source_type": getattr(src, "source_type", "unknown"),
            }
            if isinstance(src, LocalDirectorySource):
                info.update({
                    "directory": src.directory,
                    "source_id": src.source_id,
                    "recursive": src.recursive,
                    "category": src.category,
                })
            elif isinstance(src, HttpCrawlSource):
                info.update({
                    "source_id": src.source_id,
                    "urls": list(src.urls) if hasattr(src, "urls") else [],
                    "category": src.category,
                })
            elif isinstance(src, TaxGovCnSource):
                info.update({"source_id": src.source_id, "category": src.category, "listing_url": src.listing_url,
                             "link_selector": src.link_selector, "title_selector": src.title_selector,
                             "content_selector": src.content_selector, "date_selector": src.date_selector,
                             "max_pages": src.max_pages, "max_items": src.max_items})
            else:
                info["source_id"] = getattr(src, "source_id", "unknown")
            result.append(info)
        return result

    def get_latest_report(self) -> dict | None:
        """获取最近一次更新报告"""
        reports = self.tracker.get_reports(limit=1)
        return reports[0] if reports else None

    def get_report_history(self, limit: int = 20) -> list[dict]:
        """获取历史更新报告列表"""
        return self.tracker.get_reports(limit=limit)

    def get_scheduler_status(self) -> dict:
        """获取调度器运行状态"""
        return {
            "running": bool(self._scheduler_thread and self._scheduler_thread.is_alive()),
            "sources_count": len(self._sources),
            "pending_count": self.tracker.get_pending_count(),
            "changed_count": len(self.tracker.get_changes(limit=1)),
            "interval_seconds": self._interval_seconds,
            "updating": self._run_lock.locked(),
            "progress": dict(self._progress),
        }

    # ---------------- 调度 ---------------- #
    def start_scheduler(self, interval_seconds: int = 3600, run_immediately: bool = True) -> None:
        """
        启动后台线程，按固定周期运行 run_once。
        若环境安装了 APScheduler，可改用 APScheduler 实现更精细的调度。
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self.set_interval(interval_seconds)
            return

        self.set_interval(interval_seconds)
        self._stop_event.clear()
        self._schedule_wakeup.clear()

        def _loop():
            logger.info(f"[PolicyUpdater] 调度器启动，间隔 {interval_seconds}s")
            # 启动时先跑一次
            if run_immediately:
                try:
                    self.run_once()
                except Exception as e:
                    logger.error(f"[PolicyUpdater] 启动时任务失败: {e}")

            while not self._stop_event.is_set():
                changed = self._schedule_wakeup.wait(self._interval_seconds)
                self._schedule_wakeup.clear()
                if self._stop_event.is_set():
                    break
                if changed:
                    continue  # 从新设置的间隔重新计时，不中断正在执行的扫描。
                try:
                    self.run_once()
                except Exception as e:
                    logger.error(f"[PolicyUpdater] 定时任务失败: {e}")

        self._scheduler_thread = threading.Thread(target=_loop, name="policy_updater", daemon=True)
        self._scheduler_thread.start()

    def set_interval(self, interval_seconds: int) -> None:
        if not isinstance(interval_seconds, (int, float)) or interval_seconds <= 0:
            raise ValueError("更新间隔必须大于 0")
        self._interval_seconds = interval_seconds
        self._schedule_wakeup.set()

    def stop_scheduler(self) -> None:
        self._stop_event.set()
        self._schedule_wakeup.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
            if self._scheduler_thread.is_alive():
                raise RuntimeError("政策更新仍在执行，停止信号已发送；调度线程尚未退出")
            self._scheduler_thread = None
        logger.info("[PolicyUpdater] 调度器已停止")

    # ---------------- 上下文支持 ---------------- #
    def __enter__(self) -> "PolicyUpdater":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_scheduler()
        self.tracker.close()

# ============================================================ #
# 4. 内置通知钩子
# ============================================================ #

def logging_notify_hook(report: UpdateReport) -> None:
    """将更新报告写入日志"""
    logger.info("[政策更新报告] %s", json.dumps(report.to_dict(), ensure_ascii=False))

def webhook_notify_hook(webhook_url: str) -> Callable[[UpdateReport], None]:
    """
    将更新报告 POST 到 webhook（可对接企业微信/钉钉/飞书通知）。
    """
    import requests

    def _hook(report: UpdateReport) -> None:
        try:
            requests.post(
                webhook_url,
                json={"text": f"[政策更新] {json.dumps(report.to_dict(), ensure_ascii=False)}"},
                timeout=10,
            )
        except Exception as e:
            logger.error(f"webhook 发送失败: {e}")

    return _hook

# ============================================================ #
# 5. 辅助工具
# ============================================================ #

def _safe_filename(name: str) -> str:
    bad = '<>:"/\\|?*\r\n\t'
    out = "".join("_" if c in bad else c for c in name)
    return out[:120] or "untitled"

# ============================================================ #
# 6. 对外便捷入口
# ============================================================ #

def run_scheduler(
    source: PolicySource,
    tracker_db: str = "data/policy_tracker.sqlite3",
    kb_work_dir: str = "data/kb",
    kb_id: str = "tax_policy",
    interval_seconds: int = 3600,
    notify: Optional[Callable[[UpdateReport], None]] = None,
) -> PolicyUpdater:
    """
    便捷启动函数：一行调用即可启动定期更新任务。

        updater = run_scheduler(
            source=LocalDirectorySource(directory="data/policies"),
            interval_seconds=1800,
        )
    """
    updater = PolicyUpdater(
        tracker_db=tracker_db,
        kb_work_dir=kb_work_dir,
        kb_id=kb_id,
        notify_hooks=[notify] if notify else [logging_notify_hook],
    )
    updater.add_source(source)
    updater.start_scheduler(interval_seconds=interval_seconds)
    return updater

# ============================================================ #
# 7. CLI 自测入口
# ============================================================ #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    demo_dir = os.path.join(os.path.dirname(__file__), "..", "data", "policies")
    os.makedirs(demo_dir, exist_ok=True)

    updater = PolicyUpdater(
        tracker_db=os.path.join(os.path.dirname(__file__), "..", "data", "policy_tracker.sqlite3"),
        kb_work_dir=os.path.join(os.path.dirname(__file__), "..", "data", "kb"),
        kb_id="tax_policy",
        notify_hooks=[logging_notify_hook],
    )
    updater.add_source(LocalDirectorySource(directory=demo_dir))

    print("=== 单次运行测试 ===")
    report = updater.run_once()
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    # 如需启动定期任务：
    # updater.start_scheduler(interval_seconds=60)
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     updater.stop_scheduler()
