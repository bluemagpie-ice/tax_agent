"""
T12 — FastAPI 应用启动 & 基础路由测试

测试点：
  1. app 能正常导入（不 import 错误）
  2. GET / 返回 200 + HTML
  3. GET /api/skills 返回 skills JSON
  4. POST /api/chat 不带 Authorization → 422
  5. POST /api/chat 带错误 Token → 401
  6. POST /api/chat 带正确 Token → 200 + TaxQueryResponse
"""

import sys
from unittest.mock import AsyncMock, patch, MagicMock
import pytest


# -------- 依赖探测：没装 python-multipart 就跳过整个模块 --------
# web/main.py 的 /api/knowledge/upload 路由需要它
try:
    import multipart  # noqa: F401
except ImportError:
    pytest.skip(
        "缺少依赖 python-multipart，请先执行：pip install python-multipart",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient


# ================================================================== 0. 准备：mock 全局 knowledge_agent
# ==================================================================
# web/main.py 在模块顶层实例化了 KnowledgeAgent()（依赖 LLM/RAG），
# 测试时我们用 Mock 替换它，避免真正调用外部服务。

_MOCKED_AGENT_PATH = "web.main.knowledge_agent"

# 模拟 invoke_messages 的返回值：LangGraph state 格式
_FAKE_AGENT_RESPONSE = {
    "messages": [
        # 最后一条是 AI 回复（倒序遍历时先碰到）
        type("AIMsg", (), {"content": "增值税税率一般为 13%、9%、6%、3%", "__class__": type("AIMessage", (), {})})(),
        # 前面还有一条占位（会被倒序跳过）
    ],
    "retrieved_context": "增值税税率相关政策...",
}

@pytest.fixture(autouse=True)
def _mock_agents():
    """每个测试自动 mock knowledge_agent / tax_agent / conversation_memory，避免依赖外部服务"""
    fake_agent = MagicMock()
    fake_agent.name = "knowledge_agent"
    fake_agent.description = "基于知识库检索的税务问答智能体"
    fake_agent.capabilities = ["税务问答", "知识检索", "政策解读", "术语解释"]
    fake_agent.invoke_messages = AsyncMock(return_value=_FAKE_AGENT_RESPONSE)

    # /api/chat 已改为走编排器 tax_agent.run()，同样 mock，避免真实 LLM/Qdrant
    fake_tax_agent = MagicMock()
    fake_tax_agent.run = AsyncMock(return_value="增值税税率一般为 13%、9%、6%、3%")

    # /api/memory 依赖 conversation_memory（lifespan 里才初始化），这里 mock 掉
    fake_memory = MagicMock()
    fake_memory.get_session = MagicMock(return_value=None)
    fake_memory.context_for = MagicMock(
        return_value={"memory_layers": {}, "context_budget": {}}
    )
    fake_memory.list_sessions = MagicMock(return_value=[])
    fake_memory.stats = MagicMock(
        return_value={"sessions": 0, "turns": 0, "working_messages": 0, "episodes": 0}
    )
    fake_memory.delete_session = MagicMock(return_value=False)

    with patch(_MOCKED_AGENT_PATH, fake_agent), \
         patch("web.main.tax_agent", fake_tax_agent), \
         patch("web.main.conversation_memory", fake_memory):
        # 重新导入 app，确保 patch 生效
        if "web.main" in sys.modules:
            del sys.modules["web.main"]
        # 把 mock 注入到 web.main 命名空间
        import web.main as main_module
        main_module.knowledge_agent = fake_agent
        main_module.tax_agent = fake_tax_agent
        main_module.conversation_memory = fake_memory
        yield

@pytest.fixture
def client():
    from web.main import app
    return TestClient(app)

# ================================================================== 1. 应用启动
class TestAppStartup:
    def test_app_importable(self):
        """web.main.app 能正常导入，不抛出 import 错误"""
        from web.main import app
        assert app.title == "税务Agent"
        assert app.version == "1.0"

    def test_app_has_expected_routes(self, client):
        """app 注册了预期的路由"""
        routes = {r.path for r in client.app.routes}
        assert "/" in routes
        assert "/api/skills" in routes
        assert "/api/chat" in routes
        assert "/api/chat/stream" in routes

# ================================================================== 2. GET /
class TestHomePage:
    def test_get_home_returns_200_and_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_home_html_not_empty(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert len(resp.text) > 100

# ================================================================== 3. GET /api/skills
class TestSkillsEndpoint:
    def test_skills_returns_success(self, client):
        resp = client.get("/api/skills")
        assert resp.status_code == 200

        data = resp.json()
        assert data["success"] is True
        assert "data" in data

        agent_info = data["data"]
        assert agent_info["name"] == "knowledge_agent"
        assert "capabilities" in agent_info
        assert isinstance(agent_info["capabilities"], list)
        assert len(agent_info["capabilities"]) > 0
        assert "sub_agents" in agent_info

# ================================================================== 4 & 5. POST /api/chat 认证错误
class TestChatAuth:
    def test_chat_without_auth_header_returns_422(self, client):
        """不带 Authorization 头 → FastAPI Header(...) 必填校验 → 422"""
        resp = client.post(
            "/api/chat",
            json={"user_id": "u1", "query": "什么是增值税"},
            # 故意不带 headers
        )
        assert resp.status_code == 422  # 422 Unprocessable Entity（参数校验失败）

    def test_chat_without_bearer_prefix_returns_401(self, client):
        """带 Authorization 但不是 Bearer 格式 → 401"""
        resp = client.post(
            "/api/chat",
            json={"user_id": "u1", "query": "什么是增值税"},
            headers={"Authorization": "Basic test_token_123"},
        )
        assert resp.status_code == 401

    def test_chat_with_wrong_token_returns_401(self, client):
        """Bearer 格式但 token 错误 → 401"""
        resp = client.post(
            "/api/chat",
            json={"user_id": "u1", "query": "什么是增值税"},
            headers={"Authorization": "Bearer wrong_token_xxx"},
        )
        assert resp.status_code == 401
        assert "无效" in resp.json()["detail"] or "凭证" in resp.json()["detail"]

# ================================================================== 6. POST /api/chat 正常流程
class TestChatSuccess:
    VALID_TOKEN = "Bearer test_token_123"

    def test_chat_with_valid_token_returns_200(self, client):
        resp = client.post(
            "/api/chat",
            json={
                "user_id": "u1",
                "query": "增值税税率是多少",
                "session_id": "session_test_001",
            },
            headers={"Authorization": self.VALID_TOKEN},
        )
        assert resp.status_code == 200

        body = resp.json()
        # TaxQueryResponse 结构
        assert body["session_id"] == "session_test_001"
        assert "answer" in body
        assert "references" in body
        assert body["status"] == "success"

    def test_chat_generates_session_id_if_omitted(self, client):
        """不传 session_id 时，服务端自动生成"""
        resp = client.post(
            "/api/chat",
            json={"user_id": "u1", "query": "你好"},
            headers={"Authorization": self.VALID_TOKEN},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"].startswith("session_")

    def test_chat_with_admin_token_also_works(self, client):
        resp = client.post(
            "/api/chat",
            json={"user_id": "u1", "query": "管理员测试"},
            headers={"Authorization": "Bearer admin_token_456"},
        )
        assert resp.status_code == 200

    def test_chat_invalid_body_returns_422(self, client):
        """请求体缺少必填字段 query → 422"""
        resp = client.post(
            "/api/chat",
            json={"user_id": "u1"},
            headers={"Authorization": self.VALID_TOKEN},
        )
        assert resp.status_code == 422

# ================================================================== 额外：辅助路由
class TestOtherEndpoints:
    def test_global_search_works(self, client):
        """/api/search 已对接真实检索，这里 mock 掉底层避免真实 embedding/Qdrant 调用"""
        fake_result = [{"id": "c1", "title": "增值税暂行条例", "content": "第一条...", "score": 0.9}]
        with patch("services.tools.kb_tools.search_tax_policy", return_value=fake_result) as mock_search:
            resp = client.get("/api/search", params={"q": "增值税", "limit": 10},
                              headers={"Authorization": "Bearer test_token_123"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total"] == 1
        assert body["results"][0]["title"] == "增值税暂行条例"
        mock_search.assert_called_once()

    def test_memory_endpoint(self, client):
        resp = client.get(
            "/api/memory",
            params={"user_id": "u1"},
            headers={"Authorization": "Bearer test_token_123"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_memory_endpoint_requires_auth(self, client):
        """未带 Authorization 头访问 /api/memory → 应被拒绝（Header 必填 → 422）"""
        resp = client.get("/api/memory", params={"user_id": "u1"})
        assert resp.status_code == 422

    def test_conversations_endpoint_requires_auth(self, client):
        """未带 Authorization 头访问 /api/conversations → 应被拒绝"""
        resp = client.get("/api/conversations", params={"user_id": "u1"})
        assert resp.status_code == 422

    def test_knowledge_upload_no_file_returns_422(self, client):
        resp = client.post("/api/knowledge/upload")
        assert resp.status_code == 422  # 缺少 file 参数

    def test_settings_requires_auth(self, client):
        """未带 Authorization 头访问 /api/settings → 应被拒绝"""
        resp = client.get("/api/settings")
        assert resp.status_code == 422

    def test_policy_status_requires_auth(self, client):
        """未带 Authorization 头访问 /api/policy/status → 应被拒绝"""
        resp = client.get("/api/policy/status")
        assert resp.status_code == 422

    def test_me_returns_user_role(self, client):
        resp = client.get("/api/me", headers={"Authorization": "Bearer test_token_123"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "user"
        assert body["is_developer"] is False

    def test_me_returns_developer_role(self, client):
        resp = client.get("/api/me", headers={"Authorization": "Bearer admin_token_456"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "admin"
        assert body["is_developer"] is True

    def test_me_requires_auth(self, client):
        resp = client.get("/api/me")
        assert resp.status_code == 422

    def test_knowledge_databases_requires_auth(self, client):
        resp = client.get("/api/knowledge/databases")
        assert resp.status_code == 422

    def test_create_knowledge_database_requires_auth(self, client):
        resp = client.post("/api/knowledge/databases", json={"name": "测试库"})
        assert resp.status_code == 422

    def test_list_knowledge_databases(self, client):
        fake_mgr = MagicMock()
        fake_mgr.get_databases_by_user = AsyncMock(return_value={"databases": []})
        with patch("web.main.kb_manager", fake_mgr):
            resp = client.get("/api/knowledge/databases", headers={"Authorization": "Bearer test_token_123"})
        assert resp.status_code == 200
        assert resp.json()["databases"] == []

    def test_rename_conversation_requires_auth(self, client):
        resp = client.patch("/api/conversations/sess_1", json={"title": "新标题"})
        assert resp.status_code == 422

    def test_rename_conversation_works(self, client):
        resp = client.patch(
            "/api/conversations/sess_1",
            json={"title": "新标题"},
            headers={"Authorization": "Bearer test_token_123"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert resp.json()["title"] == "新标题"
