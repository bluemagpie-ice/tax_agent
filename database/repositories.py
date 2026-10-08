"""
数据访问层（Repository）
---------------------------------------
封装对 SQLAlchemy 模型的常用 CRUD，使业务代码不需要直接操作 Session。
"""

import logging
from typing import Optional

from database.models import (
    SessionLocal,
    User,
    KnowledgeBases,
    KnowledgeFiles,
    KnowledgeChunks,
)

logger = logging.getLogger(__name__)

class UserRepository:
    """用户数据访问"""

    def get_by_uid(self, uid: str) -> Optional[User]:
        """
        根据用户名 / uid 查询用户。
        （当前 User 模型用 username 作为唯一标识，这里用 uid 匹配 username）
        """
        session = SessionLocal()
        try:
            return session.query(User).filter(User.username == uid).first()
        finally:
            session.close()

class KnowledgeBaseRepository:
    """知识库元数据访问"""

    def get_all(self) -> list[KnowledgeBases]:
        session = SessionLocal()
        try:
            return session.query(KnowledgeBases).all()
        finally:
            session.close()

    def get_by_kb_id(self, kb_id: str) -> Optional[KnowledgeBases]:
        """
        当前 KnowledgeBases 使用 name 作为业务标识；
        这里通过 name 匹配（kb_id 视为 name）。
        """
        session = SessionLocal()
        try:
            return session.query(KnowledgeBases).filter(KnowledgeBases.name == kb_id).first()
        finally:
            session.close()

# 新增的修改内容
    def create(self, name: str, created_by: str = "", kb_type: str = "qdrant", share_config: str | None = None) -> KnowledgeBases:
        session = SessionLocal()
        try:
            kb = KnowledgeBases(
                name=name,
                created_by=created_by,
                kb_type=kb_type,
                share_config=share_config,
            )
            session.add(kb)
            session.commit()
            session.refresh(kb)
            return kb
        finally:
            session.close()

    def delete(self, kb_id: str) -> bool:
        session = SessionLocal()
        try:
            kb = session.query(KnowledgeBases).filter(KnowledgeBases.name == kb_id).first()
            if kb is None:
                return False
            session.delete(kb)
            session.commit()
            return True
        finally:
            session.close()

class KnowledgeFileRepository:
    """知识库文件数据访问"""

    def create(self, path: str, business_id: str | None = None, file_name: str | None = None, tags: str = "", chunk_count: int = 0) -> KnowledgeFiles:
        session = SessionLocal()
        try:
            f = KnowledgeFiles(
                path=path,
                business_id=business_id,
                file_name=file_name,
                tags=tags,
                chunk_count=chunk_count,
            )
            session.add(f)
            session.commit()
            session.refresh(f)
            return f
        finally:
            session.close()

    def get_by_id(self, file_id: int) -> Optional[KnowledgeFiles]:
        session = SessionLocal()
        try:
            return session.query(KnowledgeFiles).filter(KnowledgeFiles.id == file_id).first()
        finally:
            session.close()

    def get_by_business_id(self, business_id: str) -> Optional[KnowledgeFiles]:
        """通过业务主键（file_id 如 file_1234567890）查询文件记录"""
        session = SessionLocal()
        try:
            return session.query(KnowledgeFiles).filter(KnowledgeFiles.business_id == business_id).first()
        finally:
            session.close()

    def delete_by_business_id(self, business_id: str) -> bool:
        """通过业务主键删除文件记录，同时级联删除关联的 chunks（需手动处理）"""
        session = SessionLocal()
        try:
            record = session.query(KnowledgeFiles).filter(KnowledgeFiles.business_id == business_id).first()
            if record is None:
                return False
            # 先删除关联 chunks
            session.query(KnowledgeChunks).filter(KnowledgeChunks.file_id == record.id).delete()
            session.delete(record)
            session.commit()
            return True
        finally:
            session.close()

    def get_all(self) -> list[KnowledgeFiles]:
        """获取所有文件记录（按创建时间倒序）"""
        session = SessionLocal()
        try:
            return session.query(KnowledgeFiles).order_by(KnowledgeFiles.create_at.desc()).all()
        finally:
            session.close()


class KnowledgeChunkRepository:
    """知识库分块数据访问"""

    def create_batch(self, sql_file_id: int, chunks: list[dict]) -> list[KnowledgeChunks]:
        """
        批量创建分块记录。

        Args:
            sql_file_id: KnowledgeFiles 表的自增主键 id
            chunks: 分块数据列表，每项含 {"id": str, "content": str}
        """
        session = SessionLocal()
        try:
            records = []
            for chunk in chunks:
                records.append(KnowledgeChunks(
                    file_id=sql_file_id,
                    content=chunk.get("content", ""),
                ))
            session.add_all(records)
            session.commit()
            return records
        finally:
            session.close()

    def count_by_file_id(self, sql_file_id: int) -> int:
        """统计指定文件的分块数"""
        session = SessionLocal()
        try:
            return session.query(KnowledgeChunks).filter(KnowledgeChunks.file_id == sql_file_id).count()
        finally:
            session.close()