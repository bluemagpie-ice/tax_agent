import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import logging

# ========数据库=========

# todo:表的映射 relationship() 方法

load_dotenv()

# 创建基础类
Base = declarative_base()
logger = logging.getLogger(__name__)

class User(Base):
    # 似乎可以不需要。
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, comment="用户ID,主键")
    username = Column(String(30), unique=True, nullable=False, comment="用户名")
    create_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")
    update_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    # 打印内容
    def __repr__(self):
        return f"User(id = {self.id}, username = {self.username})"

class KnowledgeFiles(Base):
    """
    上传的文件信息
    """
    __tablename__ = "knowledge_files"

    id = Column(Integer, primary_key=True, comment="文件ID,主键")
    business_id = Column(String(100), unique=True, nullable=True, comment="业务主键，对应知识库系统的 file_id（如 file_1234567890）")
    file_name = Column(String(255), nullable=True, comment="原始文件名")
    path = Column(Text, nullable=False, comment="文件路径")
    create_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")
    tags = Column(String(200),comment="所属知识库，逗号分割。")
    chunk_count = Column(Integer, default=0, comment="分块数量")
    chunks = relationship("KnowledgeChunks", back_populates="file")

    def __repr__(self):
        return f"KnowledgeFiles(id={self.id},tags={self.tags})"

class KnowledgeChunks(Base):
    """
    知识库：文件产生的Chunk
    """
    __tablename__ = "knowledge_chunks"
    id = Column(Integer, primary_key=True, comment="ChunkID,主键")
    file_id = Column(Integer, ForeignKey("knowledge_files.id"),nullable=False, comment="所属文件ID")
    content = Column(Text, nullable=False, comment="Chunk内容")
    create_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")
    file = relationship("KnowledgeFiles", back_populates="chunks")

    def __repr__(self):
        return f"KnowledgeChunks(id={self.id}, file_id={self.file_id})"

class UserMemories(Base):
    """
    知识库：用户存储的记忆
    """
    __tablename__ = "user_memories"
    id = Column(Integer, primary_key=True, comment="记忆ID,主键")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID")
    content = Column(Text, nullable=False, comment="记忆内容")
    create_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")

    def __repr__(self):
        return f"UserMemories(id={self.id}, user_id={self.user_id})"

# 新增的修改内容
class KnowledgeBases(Base):
    """
    知识库总表
    """
    __tablename__ = "knowledge_bases"
    id = Column(Integer, primary_key=True, comment="知识库ID,主键")
    name = Column(String(30), nullable=False, comment="知识库名称")
    kb_type = Column(String(30), nullable=False, default="qdrant", comment="知识库类型: qdrant / faiss / milvus 等")
    created_by = Column(String(100), nullable=False, default="", comment="创建者 uid")
    # SQLite 没有原生 JSON 类型，用 Text 存 JSON 字符串，读写时自行 json.dumps / json.loads
    share_config = Column(Text, nullable=True, comment="分享配置 JSON: {access_level, department_ids, user_uids}")
    create_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")

    def __repr__(self):
        return f"KnowledgeBases(id={self.id}, name={self.name}, created_by={self.created_by})"

class Conversation(Base):
    """
    对话表：一次对话会话
    """
    __tablename__ = "conversations"

    id= Column(Integer, primary_key=True, comment = "对话ID,主键")
    user_id = Column(Integer, ForeignKey("users.id"), comment="用户id，关联用户表")
    create_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")

    def __repr__(self):
        return f"Conversation(id={self.id}, user_id={self.user_id})"

class Messages(Base):
    """
    对话表：一次对话会话中的消息
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, comment="消息ID,主键")
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, comment="所属对话ID")
    role = Column(String(30), nullable=False, comment="角色:user/agent")
    content = Column(Text, nullable=False, comment="对话内容")
    create_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")

    def __repr__(self):
        return f"Messages(id={self.id}, role={self.role})"

# ========数据库连接与操作配置=========
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./tax_agent.db"

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)

# ========创建表格========
def create_tables():
    """
    创建所有数据库
    """
    Base.metadata.create_all(bind=engine)
    # create_all 不会更新旧表；仅补齐文档 CRUD 已使用的三个可空/默认字段。
    if engine.dialect.name == "sqlite":
        from sqlalchemy import inspect, text
        existing = {c["name"] for c in inspect(engine).get_columns("knowledge_files")}
        with engine.begin() as connection:
            for name, definition in (("business_id", "VARCHAR(100)"), ("file_name", "VARCHAR(255)"),
                                     ("chunk_count", "INTEGER DEFAULT 0")):
                if name not in existing:
                    connection.execute(text(f"ALTER TABLE knowledge_files ADD COLUMN {name} {definition}"))
                    logger.info("旧文档表补齐字段：%s", name)
            connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_knowledge_files_business_id ON knowledge_files (business_id)"))
    logger.info("数据库表格创建成功")

def get_db_session():
    """
    获取数据库会话对象。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        logger.debug("数据库会话已关闭")


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    logger.info("表创建成功")
