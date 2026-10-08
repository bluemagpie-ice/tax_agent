"""checkpointer 接线冒烟测试。

验证：_ensure_checkpointer 会真正创建 AsyncSqliteSaver 并建表（此前从未被调用，
导致 LangGraph 多轮历史从未真正持久化），且 delete_thread 可正常执行。
"""

import asyncio
from unittest.mock import patch

from agents.base_agent import BaseAgent


class _DummyAgent(BaseAgent):
    """最小子类，仅用于触发 checkpointer 初始化，不真正构建图。"""

    name = "dummy"

    async def get_graph(self, **kwargs):
        class _G:
            checkpointer = self.checkpointer

        return _G()


def test_ensure_checkpointer_and_delete(tmp_path):
    async def _run():
        agent = _DummyAgent()
        await agent._ensure_checkpointer()
        assert agent._checkpointer is not None, "checkpointer 应被初始化"

        # 删除不存在的线程：表已 setup，DELETE 0 行也不应抛异常
        assert await agent.delete_thread("no_such_thread") is True
        assert await agent.delete_thread("another_thread") is True

        # 清理：关闭底层 aiosqlite 连接，避免 asyncio.run 结束时后台线程报错
        await agent._checkpointer.conn.close()

    with patch("agents.base_agent._CHECKPOINT_DB_PATH", tmp_path / "cp.sqlite3"):
        asyncio.run(_run())
