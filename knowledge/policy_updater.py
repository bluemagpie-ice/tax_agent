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
            stat = path.stat()
            items.append(PolicyItem(
                source_id=self.source_id,
                item_id=str(path.resolve()),
                title=path.stem,
                file_path=str(path.resolve()),
                published=datetime.fromtimestamp(stat.st_mtime),
                category=self.category,
                metadata={"size": stat.st_size},
            ))
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
        for url in self.urls:
            try:
                if self.fetcher:
                    title, content = self.fetcher(url)
                else:
                    title, content = self._default_fetch(url)
                if not content:
                    continue
                items.append(PolicyItem(
                    source_id=self.source_id,
                    item_id=url,
                    title=title or url,
                    content=content,
                    url=url,
                    published=datetime.now(),
                    category=self.category,
                ))
            except Exception as e:
                logger.error(f"[HttpCrawlSource] 抓取失败 {url}: {e}")
        logger.info(f"[HttpCrawlSource] 抓取到 {len(items)} 条政策")
        return items

    @staticmethod
    def _default_fetch(url: str) -> tuple[str, str]:
        """
        默认抓取实现（纯文本）。
        生产环境建议使用 requests + BeautifulSoup / Readability 解析。
        """
        try:
            import requests
            resp = requests.get(url, timeout=15)
            resp.encoding = resp.apparent_encoding or "utf-8"
            return url, resp.text
        except Exception as e:
            logger.error(f"默认抓取失败: {e}")
            return url, ""

# ============================================================ #
# 2. 政策入库追踪器（去重持久化）
# ============================================================ #

class TaxGovCnSource(PolicySource):
    """国家税务总局法规库爬取源

    自动爬取法规列表页，解析每一条法规的标题、文号、发布日期、正文，
    去重后返回 PolicyItem 列表。

    典型用法::

        source = TaxGovCnSource(
            listing_url="https://www.chinatax.gov.cn/chinatax/n810214/n810641/n810651/index.html",
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
        self._session = requests.Session()
        self._session.headers.update({
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
        seen_urls: set = set()

        for page in range(self.max_pages):
            url = self._build_page_url(page)
            logger.info(f"[TaxGovCnSource] 抓取列表页 {page + 1}: {url}")
            try:
                resp = self._session.get(url, timeout=self.request_timeout)
                resp.encoding = self._detect_encoding(resp)
                soup = BeautifulSoup(resp.text, "lxml")
            except Exception as e:
                logger.warning(f"[TaxGovCnSource] 列表页抓取失败: {url} — {e}")
                break

            links_found = 0
            for a_tag in soup.select(self.link_selector):
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

            logger.info(f"[TaxGovCnSource] 第 {page + 1} 页发现 {links_found} 个链接")
            if links_found == 0:
                break

        logger.info(f"[TaxGovCnSource] 共抓取 {len(items)} 条政策")
        return items

    def _build_page_url(self, page: int) -> str:
        if page == 0:
            return self.listing_url
        base, _, rest = self.listing_url.rpartition("/")
        if "index" in rest:
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
        resp.encoding = self._detect_encoding(resp)
        soup = BeautifulSoup(resp.text, "lxml")

        title_el = soup.select_one(self.title_selector)
        title = title_el.get_text(strip=True) if title_el else ""

        date_el = soup.select_one(self.date_selector)
        date_text = date_el.get_text(strip=True) if date_el else ""
        date_match = re.search(r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})", date_text)
        publish_date = date_match.group(1) if date_match else ""

        content_el = soup.select_one(self.content_selector)
        if not content_el:
            content_el = soup.body
        content_text = content_el.get_text("\n", strip=True) if content_el else ""

        if not content_text:
            return None

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
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS policy_record (
                unique_key            TEXT PRIMARY KEY,
                source_id             TEXT NOT NULL,
                item_id               TEXT NOT NULL,
                title                 TEXT,
                content_hash          TEXT,
                previous_content_hash TEXT,
                content_snapshot      TEXT,
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
        self._conn.commit()

    # ---------- 查询 ---------- #
    def exists(self, item: PolicyItem) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM policy_record WHERE unique_key = ? AND status = 'ingested'",
            (item.unique_key,),
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
        return self._conn.execute(
            "SELECT * FROM policy_record WHERE status='pending' LIMIT ?",
            (limit,),
        ).fetchall()

    # ---------- 写入 ---------- #
    def upsert_pending(self, item: PolicyItem, content_text: str = "") -> tuple[bool, bool]:
        """若已存在且内容未变化则跳过；若内容变化则标记为 changed 并仍返回需要更新。

        Returns (is_new, is_changed)
        """
        if not item.unique_key or not item.content_hash:
            return False, False
        key = item.unique_key
        now = datetime.now().isoformat(timespec="seconds")
        snapshot = (content_text or "")[:2000]

        existing = self._conn.execute(
            "SELECT content_hash, title FROM policy_record WHERE unique_key = ?", (key,)
        ).fetchone()

        if existing is None:
            self._conn.execute(
                """INSERT INTO policy_record
                   (unique_key, source_id, item_id, title, content_hash, status, content_snapshot, created_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (key, item.source_id, item.item_id, item.title, item.content_hash, snapshot, now),
            )
            self._conn.commit()
            return True, False

        old_hash = existing["content_hash"]
        if old_hash == item.content_hash:
            return False, False

        self._conn.execute(
            """UPDATE policy_record
               SET title = ?, content_hash = ?, previous_content_hash = ?,
                   content_snapshot = ?, status = 'pending', created_at = ?
               WHERE unique_key = ?""",
            (item.title, item.content_hash, old_hash, snapshot, now, key),
        )
        self._conn.commit()
        return True, True

    def mark_ingested(self, unique_key: str, file_id: str = "") -> None:
        self._conn.execute(
            "UPDATE policy_record SET status='ingested', file_id=? WHERE unique_key=?",
            (file_id, unique_key),
        )
        self._conn.commit()

    def mark_failed(self, unique_key: str) -> None:
        self._conn.execute(
            "UPDATE policy_record SET status='failed' WHERE unique_key=?",
            (unique_key,),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ---------- 报告持久化 ---------- #
    def save_report(self, report_dict: dict) -> int:
        """持久化一份更新报告，返回自增 ID"""
        now = datetime.now().isoformat(timespec="seconds")
        cursor = self._conn.execute(
            """INSERT INTO update_report
               (started_at, finished_at, scanned, new_count, ingested, duplicated, failed, errors, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_reports(self, limit: int = 20) -> list[dict]:
        """获取历史更新报告列表（按时间倒序）"""
        rows = self._conn.execute(
            "SELECT * FROM update_report ORDER BY created_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "scanned": row["scanned"],
                "new": row["new_count"],
                "ingested": row["ingested"],
                "duplicated": row["duplicated"],
                "failed": row["failed"],
                "errors": json.loads(row["errors"] or "[]"),
            }
            for row in rows
        ]

    def get_pending_count(self) -> int:
        """获取待入库的政策数量"""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM policy_record WHERE status='pending'"
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_changes(self, limit: int = 20) -> list[dict]:
        """获取内容发生变更的政策列表（previous_content_hash IS NOT NULL）"""
        rows = self._conn.execute(
            """SELECT unique_key, source_id, item_id, title,
                      previous_content_hash, content_hash, content_snapshot, created_at
               FROM policy_record
               WHERE previous_content_hash IS NOT NULL
               ORDER BY created_at DESC LIMIT ?""",
            (max(1, int(limit)),),
        ).fetchall()
        return [
            {
                "unique_key": row["unique_key"],
                "source_id": row["source_id"],
                "item_id": row["item_id"],
                "title": row["title"],
                "old_hash": row["previous_content_hash"],
                "new_hash": row["content_hash"],
                "snapshot": (row["content_snapshot"] or "")[:500],
                "updated_at": row["created_at"],
            }
            for row in rows
        ]

    def get_change_diff(self, unique_key: str | None = None, source_id: str | None = None, item_id: str | None = None) -> dict | None:
        """生成某条政策的文字级 diff"""
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

        old_text = (row["content_snapshot"] or "")[:2000]
        new_text = old_text
        diff_lines = list(difflib.unified_diff(
            old_text.splitlines(keepends=True) if old_text else [""],
            new_text.splitlines(keepends=True) if new_text else [""],
            fromfile=f"{row['source_id']}/{row['item_id']} (旧)",
            tofile=f"{row['source_id']}/{row['item_id']} (新)",
        ))
        return {
            "unique_key": row["unique_key"],
            "title": row["title"],
            "old_hash": row["previous_content_hash"],
            "new_hash": row["content_hash"],
            "diff": "".join(diff_lines) if len(diff_lines) > 3 else "（仅哈希变更，快照未捕获到文本差异）",
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
    ):
        self.tracker = PolicyTracker(tracker_db)
        self.kb_work_dir = kb_work_dir
        self.kb_id = kb_id
        self.chunk_preset_id = chunk_preset_id
        self.notify_hooks: list[Callable[[UpdateReport], None]] = notify_hooks or []
        self._sources: list[PolicySource] = []

        # 调度器相关
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ---------------- 配置 ---------------- #
    def add_source(self, source: PolicySource) -> None:
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
        started_at = datetime.now().isoformat(timespec="seconds")
        report = UpdateReport(started_at=started_at, finished_at="")

        if not self._sources:
            logger.warning("未配置任何政策源，跳过本次更新")
            report.finished_at = datetime.now().isoformat(timespec="seconds")
            self._notify(report)
            return report

        # Step 1: 扫描所有源
        all_items: list[PolicyItem] = []
        for source in self._sources:
            try:
                all_items.extend(source.scan())
            except Exception as e:
                logger.error(f"扫描 {source.source_type} 失败: {e}")
                report.errors.append(f"scan[{source.source_type}]: {e}")
        report.scanned = len(all_items)
        logger.info(f"共扫描到 {report.scanned} 条候选政策")

        # Step 2: 去重 + 记录 pending
        pending_items: list[PolicyItem] = []
        for item in all_items:
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
        for item in pending_items:
            try:
                self._ingest_item(item)
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

        # Step 5: 通知
        self._notify(report)
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

        if item.file_path and os.path.exists(item.file_path):
            return self._ingest_file(item, kb)

        # 文本型：写入临时 .md 文件再入库
        if item.content:
            temp_dir = os.path.join(self.kb_work_dir, "_temp_policy")
            os.makedirs(temp_dir, exist_ok=True)
            safe_title = _safe_filename(item.title or item.item_id)
            temp_path = os.path.join(temp_dir, f"{int(time.time())}_{safe_title}.md")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(f"# {item.title}\n\n{item.content}")

            result = self._ingest_file(item, kb, override_path=temp_path)
            # 清理临时文件
            try:
                os.remove(temp_path)
            except OSError:
                pass
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
        file_id = f"policy_{int(time.time())}_{abs(hash(item.unique_key)) % 10000}"

        # 兼容无 running loop 的场景
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(
                        asyncio.run,
                        kb.ingest_file(
                            file_path,
                            file_id=file_id,
                            chunk_preset_id=self.chunk_preset_id,
                        ),
                    )
                    result = future.result(timeout=600)
            else:
                result = loop.run_until_complete(kb.ingest_file(
                    file_path,
                    file_id=file_id,
                    chunk_preset_id=self.chunk_preset_id,
                ))
        except RuntimeError:
            result = asyncio.run(kb.ingest_file(
                file_path,
                file_id=file_id,
                chunk_preset_id=self.chunk_preset_id,
            ))

        item.metadata["file_id"] = file_id
        return result

    # ---------------- 通知 ---------------- #
    def _notify(self, report: UpdateReport) -> None:
        for hook in self.notify_hooks:
            try:
                hook(report)
            except Exception as e:
                logger.error(f"通知钩子执行失败: {e}")

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
        }

    # ---------------- 调度 ---------------- #
    def start_scheduler(self, interval_seconds: int = 3600) -> None:
        """
        启动后台线程，按固定周期运行 run_once。
        若环境安装了 APScheduler，可改用 APScheduler 实现更精细的调度。
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("调度器已在运行")
            return

        self._stop_event.clear()

        def _loop():
            logger.info(f"[PolicyUpdater] 调度器启动，间隔 {interval_seconds}s")
            # 启动时先跑一次
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"[PolicyUpdater] 启动时任务失败: {e}")

            while not self._stop_event.is_set():
                self._stop_event.wait(interval_seconds)
                if self._stop_event.is_set():
                    break
                try:
                    self.run_once()
                except Exception as e:
                    logger.error(f"[PolicyUpdater] 定时任务失败: {e}")

        self._scheduler_thread = threading.Thread(target=_loop, name="policy_updater", daemon=True)
        self._scheduler_thread.start()

    def stop_scheduler(self) -> None:
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
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