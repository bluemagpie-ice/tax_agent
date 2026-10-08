"""
多知识库管理器
================
- KnowledgeBaseManager 负责：
    1. 维护多个 KnowledgeBase 实例的缓存
    2. 基于用户权限过滤知识库列表
    3. 协调 Repository（数据库层）与 KnowledgeBase（向量化层）

- 真正的知识库能力在 knowledge/kb.py 中的
  KnowledgeBase / QdrantKnowledgeBase / KnowledgeBaseFactory 里。
"""

import asyncio
import logging
import os
from typing import Optional

from knowledge.kb import KnowledgeBase, KnowledgeBaseFactory

logger = logging.getLogger(__name__)

DEFAULT_SHARE_CONFIG = {"access_level": "global", "department_ids": [], "user_uids": []}

class KBNotFoundError(Exception):
    """知识库未找到异常"""

class KnowledgeBaseManager:
    """
    知识库管理器

    使用方式：
        mgr = KnowledgeBaseManager(work_dir="data/kb")
        kb  = mgr.get_kb("qdrant", kb_id="tax")
        await kb.ingest_file("a.pdf")
        results = await kb.search("什么是进项税")
    """

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        # kb_type -> {kb_id -> KnowledgeBase}
        self.kb_instances: dict[str, dict[str, KnowledgeBase]] = {}
        self.metadata_lock = asyncio.Lock()

    # ---------- 实例管理 ---------- #
    def get_kb(self, kb_type: str, kb_id: Optional[str] = None) -> KnowledgeBase:
        """
        获取（或创建）一个知识库实例
        :param kb_type:  例如 "qdrant"
        :param kb_id:    具体知识库 ID；为空时每次新建
        """
        kb_id = kb_id or f"{kb_type}_default"
        bucket = self.kb_instances.setdefault(kb_type, {})
        if kb_id in bucket:
            return bucket[kb_id]

        kb_work_dir = os.path.join(self.work_dir, kb_type, kb_id)
        kb_instance = KnowledgeBaseFactory.create(kb_type, kb_work_dir, kb_id=kb_id)
        bucket[kb_id] = kb_instance
        logger.info(f"Created {kb_type} knowledge base instance: {kb_id}")
        return kb_instance

    def _get_kb_for_database(self, kb_type: str, kb_id: str) -> KnowledgeBase:
        """根据 kb_type + kb_id 获取知识库实例"""
        if not KnowledgeBaseFactory.is_type_supported(kb_type):
            raise KBNotFoundError(f"不支持的知识库类型: {kb_type}")
        return self.get_kb(kb_type, kb_id)

    # ---------- 权限判断 ---------- #
    @staticmethod
    def _database_info_accessible(user: dict, db_info: dict) -> bool:
        """
        权限判断逻辑

        为什么是 staticmethod？
        - 该方法不依赖 self（不需要访问管理器状态），也不修改类本身；
        - 只是一个纯函数逻辑，用 @staticmethod 可以明确表达这一点，
          同时可以直接作为独立函数使用（便于单元测试）。
        """
        if user.get("role") == "superadmin":
            return True

        user_uid = str(user.get("uid") or "")
        if user_uid and db_info.get("created_by") == user_uid:
            return True

        share_config = db_info.get("share_config") or DEFAULT_SHARE_CONFIG.copy()
        access_level = share_config.get("access_level")

        if access_level == "global":
            return True

        if access_level == "department":
            user_department_id = user.get("department_id")
            if user_department_id is None:
                return False
            try:
                department_ids = [int(d) for d in share_config.get("department_ids") or []]
                return int(user_department_id) in department_ids
            except (ValueError, TypeError):
                return False

        if access_level == "user":
            return bool(user_uid and user_uid in (share_config.get("user_uids") or []))

        return False

# 新增的修改内容
    # ---------- 列表查询 ---------- #
    async def get_databases(self) -> dict:
        """
        获取所有知识库信息。
        从数据库读取元数据，再从对应 KnowledgeBase 补充运行期信息。
        """
        import json
        from database.repositories import KnowledgeBaseRepository

        kb_repo = KnowledgeBaseRepository()
        rows = kb_repo.get_all()

        all_databases: list[dict] = []
        for row in rows:
            kb_type = row.kb_type or "qdrant"
            if not KnowledgeBaseFactory.is_type_supported(kb_type):
                logger.warning(f"跳过不支持的数据库: id={row.id}, type={kb_type}")
                continue

            try:
                kb_instance = self._get_kb_for_database(kb_type, kb_id=row.name)
                db_info = kb_instance.get_database_info(row.name, include_files=False) or {}
            except KBNotFoundError as e:
                logger.warning(str(e))
                continue

            # 解析 share_config（数据库存的是 JSON 字符串）
            share_config = DEFAULT_SHARE_CONFIG.copy()
            raw_share = row.share_config
            if raw_share:
                try:
                    parsed = json.loads(raw_share)
                    share_config.update(parsed)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"知识库 {row.name} 的 share_config 解析失败: {raw_share}")

            # 补充展示字段 + 权限字段
            db_info["kb_id"] = row.name
            db_info["kb_type"] = kb_type
            db_info["name"] = row.name
            db_info["created_at"] = str(row.create_at) if row.create_at else ""
            db_info["created_by"] = row.created_by or ""
            db_info["share_config"] = share_config
            all_databases.append(db_info)

        return {"databases": all_databases}

    async def get_databases_by_user(self, user) -> dict:
        """
        根据用户权限过滤知识库列表
        :param user: dict 或具备 uid/role/department_id 属性的对象
        """
        if isinstance(user, dict):
            user_info = user
        else:
            user_info = {
                "uid": getattr(user, "uid", None),
                "role": getattr(user, "role", "user"),
                "department_id": getattr(user, "department_id", None),
            }

        logger.info(
            f"获取用户知识库列表: role={user_info.get('role')}, "
            f"department={user_info.get('department_id')}"
        )

        all_databases = (await self.get_databases()).get("databases", [])

        if user_info.get("role") == "superadmin":
            return {"databases": all_databases}

        filtered = [
            db for db in all_databases if self._database_info_accessible(user_info, db)
        ]
        return {"databases": filtered}

    async def get_databases_by_uid(self, uid: str) -> dict:
        """根据 uid 获取知识库列表"""
        from database.repositories import UserRepository

        user_repo = UserRepository()
        user = user_repo.get_by_uid(uid)
        if not user:
            logger.warning(f"用户不存在: {uid}")
            return {"databases": []}

        user_info = {
            "uid": user.username,
            "role": "user",
            "department_id": None,
        }
        return await self.get_databases_by_user(user_info)

    async def check_accessible(self, user: dict, kb_id: str) -> bool:
        """检查用户是否有权限访问指定知识库"""
        if user.get("role") == "superadmin":
            return True

        from database.repositories import KnowledgeBaseRepository

        kb_repo = KnowledgeBaseRepository()
        kb = kb_repo.get_by_kb_id(kb_id)
        if kb is None:
            return False

        return self._database_info_accessible(
            user,
            {
                "created_by": getattr(kb, "created_by", ""),
                "share_config": getattr(kb, "share_config", None) or DEFAULT_SHARE_CONFIG.copy(),
            },
        )