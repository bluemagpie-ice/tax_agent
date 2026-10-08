"""
数据库初始化
功能：
1. 创建SQL数据库
2. 创建业务表
3. 插入初始数据
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from database.models import (
    Base, SessionLocal,
    UserMemories, KnowledgeBases, Conversation, Messages,
    create_tables as create_all_tables, User, engine
)

logger = logging.getLogger(__name__)

def load_env():
    """加载环境变量"""
    load_dotenv()
    logger.info("环境变量加载成功")

def create_tables():
    """
    创建数据库表
    若表已经存在，则不创建
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("所有业务表创建成功")
        return True
    except Exception as e:
        logger.error(f"创建数据库表失败: {e}")
        return False

def insert_initial_data():
    """
    插入初始数据

    添加示例数据方便：
    1、示例用户（text_user）
    2、示例知识库（税务知识库）
    3、示例对话和消息
    4、示例用户记忆
    """
    session = SessionLocal()
    try:
        # 插入示例用户（text_user）
        if session.query(User).filter_by(username="text_user").first() is None:
            text_user = User(username="text_user",create_at=datetime.now())
            session.add(text_user)
            session.commit()
            logger.info("示例用户(text_user)插入成功")
        else:
            logger.info("示例用户(text_user)已存在，跳过")
        # 插入示例知识库（税务知识库）
        # 新增的修改内容
        if session.query(KnowledgeBases).filter_by(name="税务知识库").first() is None:
            knowledge_database = KnowledgeBases(
                name="税务知识库",
                kb_type="qdrant",
                created_by="text_user",
                share_config='{"access_level": "global", "department_ids": [], "user_uids": []}',
                create_at=datetime.now(),
            )
            session.add(knowledge_database)
            session.commit()
            logger.info("示例知识库(税务知识库)插入成功")
        else:
            logger.info("示例知识库已存在，跳过")
        # 插入示例对话和消息
        if session.query(Conversation).filter_by(user_id=1).first() is None:
            text_con = Conversation(user_id=1, create_at=datetime.now())
            session.add(text_con)
            session.commit()
            # 创建示例消息
            msg_user = Messages(
                conversation_id=text_con.id,
                role="user",
                content="你好，我想咨询一下税额的计算公式",
                create_at=datetime.now())
            msg_agent = Messages(
                conversation_id=text_con.id,
                role="agent",
                content="您好。税额的计算公式为：税额 = 不含税价格 × 税率",
                create_at=datetime.now()
            )
            session.add_all([msg_user, msg_agent])
            session.commit()
            logger.info("示例对话插入成功")
        else:
            logger.info("示例对话已存在，跳过")

        # 插入用户记忆
        if session.query(UserMemories).first() is None:
            text_memory = UserMemories(
                user_id=1,
                content="用户咨询过税额的计算公式，对税额计算比较感兴趣",
                create_at=datetime.now()
            )
            session.add(text_memory)
            session.commit()
            logger.info("示例用户记忆创建成功")
        else:
            logger.info("示例用户记忆已存在，跳过")

    except Exception as e:
        logger.error(f"插入初始化数据失败：{e}")
        return False
    finally:
        session.close()

def init_database():
    """
    数据库初始化流程
    1、加载环境变量
    2、创建业务表
    3、插入初始数据
    """
    logger.info("开始数据库初始化")
    load_env()

    if not create_tables():
        return f"创建数据库表失败"
    if not insert_initial_data():
        return f"插入初始数据失败"

    logger.info(f"数据库初始化成功，路径: {os.path.abspath('./tax_agent.db')}")


if __name__ == '__main__':
    init_database()

