"""
轻量 TTL 缓存工具（零外部依赖，纯标准库实现）。

用于 RAG 检索结果和 Embedding 向量的短期缓存，
避免短时间内重复调用外部 API 或重复计算。
"""

import time
import threading
from typing import Any, Optional

class TTLCache:
    """带 TTL（Time-To-Live）的线程安全内存缓存。"""

    def __init__(self, ttl_seconds: int, maxsize: int = 256):
        """
        :param ttl_seconds: 缓存条目有效期（秒）
        :param maxsize:     最大条目数，超出时淘汰最旧的
        """
        self._ttl = ttl_seconds
        self._maxsize = maxsize
        self._data: dict = {}
        self._lock = threading.Lock()

    def get(self, key) -> Optional[Any]:
        """获取缓存值；已过期或不存在则返回 None。"""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.time() - ts > self._ttl:
                del self._data[key]
                return None
            return value

    def set(self, key, value) -> None:
        """写入缓存值；超出 maxsize 时淘汰最旧条目。"""
        with self._lock:
            if len(self._data) >= self._maxsize:
                oldest_key = min(self._data, key=lambda k: self._data[k][0])
                del self._data[oldest_key]
            self._data[key] = (time.time(), value)

    def clear(self) -> None:
        """文档变更或模型切换时失效现有缓存。"""
        with self._lock:
            self._data.clear()
