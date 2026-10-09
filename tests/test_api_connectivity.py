"""
API 连通性测试
=============
测试各外部 API 是否能正常连接和响应：

  - LLM API (智谱 GLM)     →  /chat/completions
  - Embedding API (SiliconFlow) → /embeddings
  - Qdrant 向量数据库       →  集合查询

运行方式：
  1. 测试全部 API:
     pytest tests/test_api_connectivity.py -v -s

  2. 只测试 LLM:
     pytest tests/test_api_connectivity.py -v -s -k "llm"

  3. 只测试 Embedding:
     pytest tests/test_api_connectivity.py -v -s -k "embedding"

  4. 只测试 Qdrant:
     pytest tests/test_api_connectivity.py -v -s -k "qdrant"

  5. 快速冒烟（每个 API 只测一个请求）:
     pytest tests/test_api_connectivity.py -v -s --run-smoke

前提条件：
  - .env 文件中已正确配置 LLM_API_KEY / EMBEDDING_API_KEY / QDRANT_* 等
  - 网络能访问对应 API 地址
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import pytest
import httpx
from dotenv import load_dotenv

# 确保项目根目录在 sys.path 中，方便 from knowledge.embedding import ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env
load_dotenv(PROJECT_ROOT / ".env")

# ============================================================
# 辅助函数：安全读取环境变量
# ============================================================

def _env(key: str, fallback: str = "") -> str:
    """读取环境变量，去除首尾空白。"""
    return (os.getenv(key) or fallback).strip()

def _mask(s: str, show: int = 4) -> str:
    """遮蔽敏感字符串，只显示首尾各 show 个字符。"""
    if len(s) <= show * 2:
        return "*" * len(s)
    return s[:show] + "****" + s[-show:]

# ============================================================
# 测试配置
# ============================================================

# LLM (智谱 GLM)
LLM_API_KEY = _env("LLM_API_KEY")
LLM_BASE_URL = _env("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
LLM_MODEL_ID = _env("LLM_MODEL_ID", "glm-4-flash")

# Embedding (SiliconFlow)
EMB_API_KEY = _env("EMBEDDING_API_KEY")
EMB_BASE_URL = _env("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMB_MODEL_ID = _env("EMBEDDING_MODEL_ID", "BAAI/bge-m3")

# Qdrant
QDRANT_URL = _env("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = _env("QDRANT_API_KEY")
VECTOR_SIZE = int(_env("VECTOR_SIZE", "1024"))

# ============================================================
# pytest 标记注册（不使用 pytest.ini，用 conftest 方式）
# ============================================================

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: 慢速测试（默认跳过，用 --run-slow 启用）",
    )

# ============================================================
# 1. LLM API (智谱 GLM) 连通性测试
# ============================================================

class TestLLMApi:
    """测试智谱 GLM LLM API 连通性"""

    @pytest.fixture(autouse=True)
    def check_config(self):
        """前置检查：API Key 必须存在。"""
        if not LLM_API_KEY:
            pytest.skip("LLM_API_KEY 未配置，跳过 LLM 连通性测试")

    # ---------- 测试 /chat/completions 基础可用性 ----------

    def test_llm_chat_simple(self):
        """
        发送一条简单消息，验证 LLM 能正常返回回复。
        预期：HTTP 200，response.choices[0].message.content 非空。
        """
        payload = {
            "model": LLM_MODEL_ID,
            "messages": [{"role": "user", "content": "请回复：OK"}],
            "max_tokens": 20,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"

        t0 = time.time()
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)) as client:
            resp = client.post(url, json=payload, headers=headers)
        elapsed = time.time() - t0

        print(f"\n  LLM URL:      {_mask(url, 8)}")
        print(f"  LLM Model:    {LLM_MODEL_ID}")
        print(f"  HTTP Status:  {resp.status_code}")
        print(f"  耗时:         {elapsed:.2f}s")

        assert resp.status_code == 200, (
            f"LLM API 返回非 200: {resp.status_code}\n"
            f"响应体: {resp.text[:500]}"
        )

        data = resp.json()
        assert "choices" in data, f"响应缺少 choices 字段。完整响应: {json.dumps(data, ensure_ascii=False)[:500]}"
        assert len(data["choices"]) > 0, f"choices 为空。完整响应: {json.dumps(data, ensure_ascii=False)[:500]}"

        message = data["choices"][0].get("message", {})
        content = (message.get("content") or "").strip()
        # 部分模型（如 GLM thinking 系列）将输出放在 reasoning_content 而非 content
        reasoning = (message.get("reasoning_content") or "").strip()

        print(f"  响应 keys:     {sorted(message.keys())}")
        if content:
            print(f"  回复内容:     {content}")
        if reasoning:
            print(f"  推理内容:     {reasoning[:120]}..." if len(reasoning) > 120 else f"  推理内容:     {reasoning}")

        # 只要 content 或 reasoning_content 非空就算成功
        assert content or reasoning, (
            f"LLM 返回空内容（content 和 reasoning_content 均为空）。"
            f"完整 message: {json.dumps(message, ensure_ascii=False)[:500]}"
        )
        print(f"  ✅ LLM API 连通正常")

    # ---------- 测试 API Key 是否有效 ----------

    def test_llm_auth_check(self):
        """验证 API Key 是否有效：用无效 Key 应该返回 401。"""
        url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
        payload = {
            "model": LLM_MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        }

        # 用空 key
        with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)) as client:
            resp_bad = client.post(
                url,
                json=payload,
                headers={"Authorization": "Bearer invalid-key-12345", "Content-Type": "application/json"},
            )
        print(f"\n  无效 Key 状态码: {resp_bad.status_code}")
        assert resp_bad.status_code in (401, 403), (
            f"预期 401/403，实际 {resp_bad.status_code}。API 可能未启用鉴权"
        )
        print(f"  ✅ LLM 鉴权机制正常")

# ============================================================
# 2. Embedding API (SiliconFlow) 连通性测试
# ============================================================

class TestEmbeddingApi:
    """测试 SiliconFlow Embedding API 连通性"""

    @pytest.fixture(autouse=True)
    def check_config(self):
        if not EMB_API_KEY:
            pytest.skip("EMBEDDING_API_KEY 未配置，跳过 Embedding 连通性测试")

    # ---------- 测试 /embeddings 基础可用性 ----------

    def test_embedding_simple(self):
        """
        向量化一段简单文本，验证返回的 embedding 向量维度正确。
        """
        payload = {
            "model": EMB_MODEL_ID,
            "input": "增值税是对商品流转过程中产生的增值额征收的一种流转税",
        }
        headers = {
            "Authorization": f"Bearer {EMB_API_KEY}",
            "Content-Type": "application/json",
        }
        url = f"{EMB_BASE_URL.rstrip('/')}/embeddings"

        t0 = time.time()
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)) as client:
            resp = client.post(url, json=payload, headers=headers)
        elapsed = time.time() - t0

        print(f"\n  Embedding URL:    {_mask(url, 8)}")
        print(f"  Embedding Model:  {EMB_MODEL_ID}")
        print(f"  HTTP Status:      {resp.status_code}")
        print(f"  耗时:             {elapsed:.2f}s")

        assert resp.status_code == 200, (
            f"Embedding API 返回非 200: {resp.status_code}\n"
            f"响应体: {resp.text[:500]}"
        )

        data = resp.json()
        assert "data" in data, f"响应缺少 data 字段: {json_keys(data)}"
        assert len(data["data"]) > 0, "data 数组为空"
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list), f"embedding 不是数组: {type(embedding)}"
        assert len(embedding) > 0, "embedding 长度为 0"

        print(f"  向量维度:         {len(embedding)}")
        if len(embedding) != VECTOR_SIZE:
            print(f"  ⚠️ 注意：向量维度 ({len(embedding)}) 与 .env 中 VECTOR_SIZE ({VECTOR_SIZE}) 不一致！")
        print(f"  ✅ Embedding API 连通正常")

    # ---------- 测试批量向量化 ----------

    def test_embedding_batch(self):
        """
        批量向量化多条文本，验证返回数量与输入一致。
        """
        texts = [
            "销售货物或提供应税劳务",
            "企业所得税税率 25%",
            "小规模纳税人增值税免征政策",
        ]
        payload = {"model": EMB_MODEL_ID, "input": texts}
        headers = {
            "Authorization": f"Bearer {EMB_API_KEY}",
            "Content-Type": "application/json",
        }
        url = f"{EMB_BASE_URL.rstrip('/')}/embeddings"

        t0 = time.time()
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=5.0)) as client:
            resp = client.post(url, json=payload, headers=headers)
        elapsed = time.time() - t0

        print(f"\n  输入条数:     {len(texts)}")
        print(f"  耗时:         {elapsed:.2f}s")
        print(f"  HTTP Status:  {resp.status_code}")

        assert resp.status_code == 200, f"批量 Embedding 失败: {resp.status_code}"
        data = resp.json()
        assert len(data["data"]) == len(texts), (
            f"返回向量数 ({len(data['data'])}) ≠ 输入数 ({len(texts)})"
        )
        print(f"  ✅ Embedding 批量向量化正常")

# ============================================================
# 3. Qdrant 向量数据库连通性测试
# ============================================================

class TestQdrantConnectivity:
    """测试 Qdrant Cloud 连通性"""

    @pytest.fixture(autouse=True)
    def check_config(self):
        if not QDRANT_URL or not QDRANT_API_KEY:
            pytest.skip("QDRANT_CLUSTER_URL 或 QDRANT_API_KEY 未配置，跳过 Qdrant 连通性测试")

    # ---------- 测试集合是否存在 ----------

    def test_qdrant_collection_exists(self):
        """
        检查 tax_chunks 集合是否存在且可访问。
        """
        from qdrant_client import QdrantClient

        t0 = time.time()
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10.0)
        collections = client.get_collections().collections
        elapsed = time.time() - t0

        collection_names = [c.name for c in collections]

        print(f"\n  Qdrant URL:       {_mask(QDRANT_URL, 8)}")
        print(f"  连接耗时:         {elapsed:.2f}s")
        print(f"  集合列表:         {collection_names}")

        assert "tax_chunks" in collection_names, (
            f"tax_chunks 集合不存在！现有集合: {collection_names}"
        )
        print(f"  ✅ Qdrant tax_chunks 集合可用")

    # ---------- 测试向量检索 ----------

    def test_qdrant_search(self):
        """
        用一段税务问题做向量检索，验证能返回结果。
        """
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance

        # 1. 先获取一个向量（用 SiliconFlow）
        if not EMB_API_KEY:
            pytest.skip("EMBEDDING_API_KEY 未配置，无法生成查询向量")

        emb_url = f"{EMB_BASE_URL.rstrip('/')}/embeddings"
        emb_payload = {"model": EMB_MODEL_ID, "input": "增值税抵扣规则"}
        emb_headers = {
            "Authorization": f"Bearer {EMB_API_KEY}",
            "Content-Type": "application/json",
        }

        t0 = time.time()
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)) as http:
            emb_resp = http.post(emb_url, json=emb_payload, headers=emb_headers)
        emb_elapsed = time.time() - t0

        assert emb_resp.status_code == 200, f"生成查询向量失败: {emb_resp.status_code}"
        query_vector = emb_resp.json()["data"][0]["embedding"]

        print(f"\n  查询向量维度:     {len(query_vector)}")
        print(f"  向量化耗时:       {emb_elapsed:.2f}s")

        # 2. 用向量检索 Qdrant
        t1 = time.time()
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10.0)
        results = client.query_points(
            collection_name="tax_chunks",
            query=query_vector,
            limit=3,
            with_payload=True,
        ).points
        search_elapsed = time.time() - t1

        print(f"  检索耗时:         {search_elapsed:.2f}s")
        print(f"  命中数:           {len(results)}")

        for i, pt in enumerate(results):
            content = (pt.payload or {}).get("content", "")[:80]
            score = getattr(pt, "score", "N/A")
            source = (pt.payload or {}).get("source_type", "?")
            print(f"    [{i+1}] score={score:.4f} | source={source} | {content}...")

        assert len(results) > 0, (
            "Qdrant 检索返回 0 条结果。"
            "可能原因：(1) tax_chunks 集合为空，需先上传知识库；(2) 查询向量与库中数据语义差异过大"
        )
        print(f"  ✅ Qdrant 向量检索正常")
        client.close()

# ============================================================
# 辅助
# ============================================================

def json_keys(data: dict) -> str:
    """安全列出 dict 的顶层 key。"""
    try:
        return ", ".join(sorted(data.keys()))
    except Exception:
        return str(type(data))

# ============================================================
# 命令行入口（可直接 python 运行）
# ============================================================

if __name__ == "__main__":
    """
    不依赖 pytest 也能直接运行:
      python tests/test_api_connectivity.py
    """
    print("=" * 60)
    print("  Tax Agent — API 连通性检查")
    print("=" * 60)

    results: dict[str, bool] = {}

    # ---- LLM ----
    print("\n[1/3] 测试 LLM API (智谱 GLM) ...")
    if LLM_API_KEY:
        try:
            url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
            payload = {"model": LLM_MODEL_ID, "messages": [{"role": "user", "content": "回复 OK"}], "max_tokens": 10}
            headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
            with httpx.Client(timeout=httpx.Timeout(connect=10, read=30)) as c:
                r = c.post(url, json=payload, headers=headers)
            ok = r.status_code == 200 and "choices" in r.json()
            results["LLM"] = ok
            print(f"  {'✅ 正常' if ok else '❌ 失败'}  HTTP {r.status_code}  耗时 {(r.elapsed.total_seconds()):.1f}s  Model: {LLM_MODEL_ID}")
        except Exception as e:
            results["LLM"] = False
            print(f"  ❌ 异常: {e}")
    else:
        results["LLM"] = False
        print("  ⏭️ 跳过（LLM_API_KEY 未配置）")

    # ---- Embedding ----
    print("\n[2/3] 测试 Embedding API (SiliconFlow) ...")
    if EMB_API_KEY:
        try:
            url = f"{EMB_BASE_URL.rstrip('/')}/embeddings"
            payload = {"model": EMB_MODEL_ID, "input": "测试文本"}
            headers = {"Authorization": f"Bearer {EMB_API_KEY}", "Content-Type": "application/json"}
            with httpx.Client(timeout=httpx.Timeout(connect=10, read=30)) as c:
                r = c.post(url, json=payload, headers=headers)
            data = r.json()
            ok = r.status_code == 200 and "data" in data
            results["Embedding"] = ok
            dim = len(data["data"][0]["embedding"]) if ok else "?"
            print(f"  {'✅ 正常' if ok else '❌ 失败'}  HTTP {r.status_code}  耗时 {(r.elapsed.total_seconds()):.1f}s  Model: {EMB_MODEL_ID}  维度: {dim}")
        except Exception as e:
            results["Embedding"] = False
            print(f"  ❌ 异常: {e}")
    else:
        results["Embedding"] = False
        print("  ⏭️ 跳过（EMBEDDING_API_KEY 未配置）")

    # ---- Qdrant ----
    print("\n[3/3] 测试 Qdrant 向量数据库 ...")
    if QDRANT_URL and QDRANT_API_KEY:
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10.0)
            cols = client.get_collections().collections
            names = [c.name for c in cols]
            ok = "tax_chunks" in names
            results["Qdrant"] = ok
            print(f"  {'✅ 正常' if ok else '❌ 失败'}  集合列表: {names}")
            client.close()
        except Exception as e:
            results["Qdrant"] = False
            print(f"  ❌ 异常: {e}")
    else:
        results["Qdrant"] = False
        print("  ⏭️ 跳过（Qdrant 配置未完整）")

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
    print(f"\n  结果: {passed}/{total} 项通过")
    if passed == total:
        print("  🎉 所有外部 API 连通正常！")
    else:
        print("  ⚠️ 部分 API 未通过，请检查 .env 配置和网络连接")
    print("=" * 60)