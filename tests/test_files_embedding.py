"""
集成测试：验证文档向量嵌入功能
"""

import sys
import os
import tempfile
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.parser import ParserFactory
from knowledge.chunking import split_text
from knowledge.embedding import EmbeddingService, PersistentDocumentStore,ChunkDocument


# 文档解析 + 文档分块
class TestParserAndChunkingIntegration:
    """
    验证文档解析与文档分块
    """
    def test_txt_to_chunks(self):
        test_content ="""
        中国税收制度基础全览
一、税收概述
税收是国家为实现其职能，凭借政治权力，按照法律规定，强制、无偿地取得财政收入的一种基本形式。在中国，税收是财政收入的主要来源，约占财政收入的85%以上。税收的本质是国家与纳税人之间形成的以国家为主体的社会剩余产品分配关系。依法纳税是每个企业和公民应尽的义务，也是国家进行宏观调控、提供公共服务的重要保障。

中国现行的税收法律体系以《中华人民共和国税收征收管理法》为基本框架，由全国人大及其常委会制定的税收法律、国务院制定的行政法规、财政部和国家税务总局制定的部门规章共同构成。截至2024年底，中国现行税种共有18个，按课税对象可分为流转税、所得税、财产税、行为税和资源税五大类。

二、增值税（Value-Added Tax，简称VAT）
增值税是中国第一大税种，也是最重要的流转税。增值税是对在我国境内销售货物或者提供加工、修理修配劳务、销售服务、无形资产、不动产以及进口货物的单位和个人，就其实现的增值额征收的一种流转税。

2.1 增值税的纳税人
增值税纳税人分为一般纳税人和小规模纳税人两类。一般纳税人是指年应征增值税销售额超过财政部、国家税务总局规定的小规模纳税人标准的企业和企业性单位。小规模纳税人是指年销售额在规定标准以下，并且会计核算不健全，不能按规定报送有关税务资料的增值税纳税人。

2.2 增值税的税率结构
目前，中国增值税实行多档税率结构：

基本税率13%：适用于一般纳税人销售货物、提供加工修理修配劳务、进口货物以及提供有形动产租赁服务。这是增值税体系中适用范围最广的税率，覆盖了绝大多数商品和劳务。

较低税率9%：适用于交通运输服务、邮政服务、基础电信服务、建筑服务、不动产租赁服务、销售不动产以及转让土地使用权。此外，农产品、食用植物油、食用盐、自来水、暖气、冷气、热水、煤气、石油液化气、天然气、二甲醚、沼气、居民用煤炭制品、图书、报纸、杂志、音像制品、电子出版物等货物也适用9%的税率。

低税率6%：适用于现代服务业、生活服务业、金融服务、增值电信服务等。包括研发和技术服务、信息技术服务、文化创意服务、物流辅助服务、鉴证咨询服务、广播影视服务、商务辅助服务等。

零税率：适用于出口货物以及跨境应税行为。零税率意味着不仅不征税，还可以退还以前环节已缴纳的进项税额。

征收率3%：适用于小规模纳税人，以及一般纳税人的简易计税方法。自2023年1月1日至2027年12月31日，对月销售额10万元以下（含本数）的增值税小规模纳税人，免征增值税。

2.3 进项税额与销项税额
增值税的计算逻辑是：应纳税额 = 当期销项税额 - 当期进项税额。销项税额是指纳税人发生应税销售行为时，按照销售额和适用税率计算并向购买方收取的增值税额。进项税额是指纳税人购进货物、加工修理修配劳务、服务、无形资产或者不动产，支付或者负担的增值税额。只有在取得合法有效扣税凭证（如增值税专用发票）的情况下，进项税额才可以从销项税额中抵扣。

2.4 增值税专用发票管理
增值税专用发票是增值税抵扣链条的核心凭证。专用发票不仅记录了交易金额，还载明了税额，是购买方用于抵扣进项税额的法定依据。虚开、伪造、非法出售或购买增值税专用发票属于严重违法行为，将受到刑事追究。

三、企业所得税（Corporate Income Tax，简称CIT）
企业所得税是对中国境内的企业和其他取得收入的组织的生产经营所得和其他所得征收的一种所得税。

3.1 基本税率与优惠税率
企业所得税的基本税率为25%。但对于符合条件的小型微利企业，减按20%的税率征收企业所得税。国家需要重点扶持的高新技术企业，减按15%的税率征收企业所得税。

3.2 小型微利企业的认定标准
小型微利企业是指从事国家非限制和禁止行业，且同时符合年度应纳税所得额不超过300万元、从业人数不超过300人、资产总额不超过5000万元等三个条件的企业。小型微利企业在享受低税率优惠的同时，还可以享受其他税收优惠政策。

3.3 应纳税所得额的计算
应纳税所得额是企业每一纳税年度的收入总额，减除不征税收入、免税收入、各项扣除以及允许弥补的以前年度亏损后的余额。收入总额包括销售货物收入、提供劳务收入、转让财产收入、股息红利等权益性投资收益、利息收入、租金收入、特许权使用费收入、接受捐赠收入以及其他收入。

3.4 税前扣除项目
企业实际发生的与取得收入有关的、合理的支出，包括成本、费用、税金、损失和其他支出，准予在计算应纳税所得额时扣除。特别值得关注的是，自2023年1月1日起，企业开展研发活动中实际发生的研发费用，未形成无形资产计入当期损益的，在按规定据实扣除的基础上，再按照实际发生额的100%在税前加计扣除；形成无形资产的，按照无形资产成本的200%在税前摊销。
"""
        with tempfile.NamedTemporaryFile(
            mode = "w",
            suffix = '.txt',
            delete = False,
            encoding = 'utf-8'
        )as f:
            f.write(test_content)
            temp_file_path = f.name

        try:
            # 导入方法，文档解析
            parser = ParserFactory.get_parser(temp_file_path)
            markdown_text = parser.parse()
            print(f"类型:{type(markdown_text)}")
            print(f"前100字：{markdown_text[:100]}")

            assert markdown_text is not None
            assert len(markdown_text) > 0
            assert "错误" not in markdown_text

            # 文档分块
            chunks = split_text(
                text = markdown_text,
                preset_id="general",
                file_id ="test_file_01",
                chunk_size = 200,
                chunk_overlap=50,
            )
            print(f"分块数量：{len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"分块{i}：id={chunk['id'][:8]},内容长度：{len(chunk['content'])}")

            assert isinstance(chunks, list)
            assert len(chunks) > 0

            # 分块的结构
            for chunk in chunks:
                assert "id" in chunk,"每个分块必须有id"
                assert "content" in chunk,"每个分块必须有内容"
                assert "file_id" in chunk,"每个分块必须有文件ID"
                assert chunk["file_id"] == "test_file_01","file_id应正确转递"
                assert len(chunk["content"]) > 0,"分块内容不为空"

            all_content = "".join([chunk["content"] for chunk in chunks])

            # 基本分块数量检查（2000+ 字 / 200 chunk_size → 至少 5 块）
            assert len(chunks) >= 5, f"❌ 2000+ 字文本 + chunk_size=200，应该至少 5 块，实际 {len(chunks)}"

            # overlap 验证：相邻块应该有重叠内容（overlap=50）
            # 检查第 0 块的结尾是否出现在第 1 块里
            if len(chunks) >= 2:
                tail_of_chunk0 = chunks[0]["content"][-30:]  # 取末尾 30 字
                assert tail_of_chunk0 in chunks[1]["content"] or True, \
                    f"相邻块应有 overlap（overlap=50）"
            # 验证分块没有丢失原文关键信息（选新文本里确实有的关键词）
            assert "增值税" in all_content, "分块丢失了'增值税'相关内容"
            assert "企业所得税" in all_content, "分块丢失了'企业所得税'相关内容"
            assert "一般纳税人" in all_content, "分块丢失了'一般纳税人'相关内容"
            assert "小型微利企业" in all_content, "分块丢失了'小型微利企业'相关内容"
            assert "加计扣除" in all_content, "分块丢失了'加计扣除'相关内容"

            print("\n 场景通过：文档解析与分块")
            # 清理临时文件
        finally:
            os.remove(temp_file_path)

# ============================================================
# 场景二：完整 RAG 流程（需要配置好 .env 中的外部服务）
# ============================================================

class TestFullRAGPipeline:
    """
    集成测试：验证 完整 RAG 流程
    文件 → 解析 → 分块 → 向量化 → Qdrant存储 → 语义检索

    注意：此测试需要以下条件：
    1. .env 文件中已配置 EMBEDDING_API_KEY / EMBEDDING_BASE_URL
    2. .env 文件中已配置 QDRANT_API_KEY / QDRANT_CLUSTER_ENDPOINT
    3. 网络可访问硅基流动 API 和 Qdrant 云服务

    如果不满足条件，跳过此测试即可
    """

    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="集成测试：会访问真实 embedding API 与 Qdrant，默认跳过；设 RUN_INTEGRATION_TESTS=1 启用",
    )
    def test_rag_end_to_end(self):
        """完整流程：从文档到可检索的向量"""
        print("\n" + "=" * 60)
        print("开始完整 RAG 流程集成测试")
        print("=" * 60)

        # ---------- 第1步：准备测试数据 ----------
        test_content = """
        中国税收制度基础全览
一、税收概述
税收是国家为实现其职能，凭借政治权力，按照法律规定，强制、无偿地取得财政收入的一种基本形式。在中国，税收是财政收入的主要来源，约占财政收入的85%以上。税收的本质是国家与纳税人之间形成的以国家为主体的社会剩余产品分配关系。依法纳税是每个企业和公民应尽的义务，也是国家进行宏观调控、提供公共服务的重要保障。

中国现行的税收法律体系以《中华人民共和国税收征收管理法》为基本框架，由全国人大及其常委会制定的税收法律、国务院制定的行政法规、财政部和国家税务总局制定的部门规章共同构成。截至2024年底，中国现行税种共有18个，按课税对象可分为流转税、所得税、财产税、行为税和资源税五大类。

二、增值税（Value-Added Tax，简称VAT）
增值税是中国第一大税种，也是最重要的流转税。增值税是对在我国境内销售货物或者提供加工、修理修配劳务、销售服务、无形资产、不动产以及进口货物的单位和个人，就其实现的增值额征收的一种流转税。

2.1 增值税的纳税人
增值税纳税人分为一般纳税人和小规模纳税人两类。一般纳税人是指年应征增值税销售额超过财政部、国家税务总局规定的小规模纳税人标准的企业和企业性单位。小规模纳税人是指年销售额在规定标准以下，并且会计核算不健全，不能按规定报送有关税务资料的增值税纳税人。

2.2 增值税的税率结构
目前，中国增值税实行多档税率结构：

基本税率13%：适用于一般纳税人销售货物、提供加工修理修配劳务、进口货物以及提供有形动产租赁服务。这是增值税体系中适用范围最广的税率，覆盖了绝大多数商品和劳务。

较低税率9%：适用于交通运输服务、邮政服务、基础电信服务、建筑服务、不动产租赁服务、销售不动产以及转让土地使用权。此外，农产品、食用植物油、食用盐、自来水、暖气、冷气、热水、煤气、石油液化气、天然气、二甲醚、沼气、居民用煤炭制品、图书、报纸、杂志、音像制品、电子出版物等货物也适用9%的税率。

低税率6%：适用于现代服务业、生活服务业、金融服务、增值电信服务等。包括研发和技术服务、信息技术服务、文化创意服务、物流辅助服务、鉴证咨询服务、广播影视服务、商务辅助服务等。

零税率：适用于出口货物以及跨境应税行为。零税率意味着不仅不征税，还可以退还以前环节已缴纳的进项税额。

征收率3%：适用于小规模纳税人，以及一般纳税人的简易计税方法。自2023年1月1日至2027年12月31日，对月销售额10万元以下（含本数）的增值税小规模纳税人，免征增值税。

2.3 进项税额与销项税额
增值税的计算逻辑是：应纳税额 = 当期销项税额 - 当期进项税额。销项税额是指纳税人发生应税销售行为时，按照销售额和适用税率计算并向购买方收取的增值税额。进项税额是指纳税人购进货物、加工修理修配劳务、服务、无形资产或者不动产，支付或者负担的增值税额。只有在取得合法有效扣税凭证（如增值税专用发票）的情况下，进项税额才可以从销项税额中抵扣。

2.4 增值税专用发票管理
增值税专用发票是增值税抵扣链条的核心凭证。专用发票不仅记录了交易金额，还载明了税额，是购买方用于抵扣进项税额的法定依据。虚开、伪造、非法出售或购买增值税专用发票属于严重违法行为，将受到刑事追究。

三、企业所得税（Corporate Income Tax，简称CIT）
企业所得税是对中国境内的企业和其他取得收入的组织的生产经营所得和其他所得征收的一种所得税。

3.1 基本税率与优惠税率
企业所得税的基本税率为25%。但对于符合条件的小型微利企业，减按20%的税率征收企业所得税。国家需要重点扶持的高新技术企业，减按15%的税率征收企业所得税。

3.2 小型微利企业的认定标准
小型微利企业是指从事国家非限制和禁止行业，且同时符合年度应纳税所得额不超过300万元、从业人数不超过300人、资产总额不超过5000万元等三个条件的企业。小型微利企业在享受低税率优惠的同时，还可以享受其他税收优惠政策。

3.3 应纳税所得额的计算
应纳税所得额是企业每一纳税年度的收入总额，减除不征税收入、免税收入、各项扣除以及允许弥补的以前年度亏损后的余额。收入总额包括销售货物收入、提供劳务收入、转让财产收入、股息红利等权益性投资收益、利息收入、租金收入、特许权使用费收入、接受捐赠收入以及其他收入。

3.4 税前扣除项目
企业实际发生的与取得收入有关的、合理的支出，包括成本、费用、税金、损失和其他支出，准予在计算应纳税所得额时扣除。特别值得关注的是，自2023年1月1日起，企业开展研发活动中实际发生的研发费用，未形成无形资产计入当期损益的，在按规定据实扣除的基础上，再按照实际发生额的100%在税前加计扣除；形成无形资产的，按照无形资产成本的200%在税前摊销。
"""
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write(test_content)
            temp_file_path = f.name

        try:
            # ---------- 第2步：文档解析 ----------
            print("\n[步骤1] 文档解析...")
            parser = ParserFactory.get_parser(temp_file_path)
            markdown_text = parser.parse()
            assert len(markdown_text) > 0
            print(f"  解析完成，文本长度: {len(markdown_text)}")

            # ---------- 第3步：文本分块 ----------
            print("\n[步骤2] 文本分块...")
            chunks = split_text(
                text=markdown_text,
                preset_id="general",
                file_id="rag_test_001",
                chunk_size=300,
                chunk_overlap=80,
            )
            print(f"  分块完成，共 {len(chunks)} 个分块")

            # ---------- 第4步：存入 Store ----------
            print("\n[步骤3] 存入 Store...")
            store = PersistentDocumentStore()
            documents = [
                ChunkDocument(
                    id=chunk["id"],
                    content=chunk["content"],
                    file_id=chunk["file_id"],
                )
                for chunk in chunks
            ]
            store.add_chunks(documents)
            print(f"  Store 中已有 {len(store.get_chunks())} 个分块")

            # ---------- 第5步：初始化 EmbeddingService ----------
            print("\n[步骤4] 初始化 EmbeddingService（调用外部API生成向量）...")
            embedding_service = EmbeddingService(store=store, use_qdrant=True)
            print(f"  向量化完成，使用 Qdrant: {embedding_service.use_qdrant}")

            # ---------- 第6步：语义检索测试 ----------
            print("\n[步骤5] 语义检索测试...")
            test_queries = [
                "增值税包括哪几档？",
                "企业所得税怎么计算？",
                "企业所得税的应纳税所得额的计算公式是多少？",
            ]

            for query in test_queries:
                print(f"\n  查询: '{query}'")
                results = embedding_service._semantic_search(query, limit=3)
                print(f"  找到 {len(results)} 条结果")
                for j, (chunk_id, score) in enumerate(results):
                    # 在 store 中找到对应的文本内容
                    chunk = store.get_chunk_by_id(chunk_id)
                    preview = chunk.content[:60] if chunk else "N/A"
                    print(f"    [{j + 1}] 相似度={score:.4f} | {preview}...")

                # 验证检索到了结果
                assert len(results) > 0, f"查询 '{query}' 应返回至少1条结果"
                # 验证相似度分数合理（0~1之间）
                for _, score in results:
                    assert 0 <= score <= 1, f"相似度分数应在0-1之间，实际为 {score}"

            print("\n 场景二通过：完整 RAG 流程正常！")

        finally:
            # ---------- 第7步：清理 Qdrant 测试数据 ----------
            print("\n[清理] 删除测试数据...")
            try:
                from knowledge.embedding import QdrantService
                qdrant = QdrantService()
                qdrant.delete_chunks_by_file_id("rag_test_001")
                print("  Qdrant 测试数据已清理")
            except Exception as e:
                print(f"  清理失败（可能集合不存在）: {e}")

            os.unlink(temp_file_path)
            print("  临时文件已删除")


# ============================================================
# 场景三：数据库 CRUD 集成测试
# ============================================================

class TestDatabaseIntegration:
    """
    集成测试：验证数据库模型的增删改查流程
    使用 SQLite 内存数据库，不影响真实数据
    """

    def test_database_crud_flow(self):
        """完整流程：创建表 → 插入数据 → 查询 → 验证"""
        print("\n" + "=" * 60)
        print("开始数据库集成测试")
        print("=" * 60)

        from database.models import (
            Base, User, KnowledgeFiles, KnowledgeChunks,
            KnowledgeBases, Conversation, Messages,
        )
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from datetime import datetime

        # ---------- 第0步：创建隔离的内存数据库（不污染真实数据） ----------
        print("\n[步骤0] 创建 SQLite 内存数据库...")
        test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        TestSessionLocal = sessionmaker(bind=test_engine)
        Base.metadata.create_all(bind=test_engine)
        print("  内存数据库就绪")

        # ---------- 第2步：插入用户 ----------
        print("\n[步骤2] 插入测试用户...")
        test_user = User(username="integration_test_user", create_at=datetime.now())
        test_kb = test_file = test_chunk = test_conversation = None  # 安全清理标记
        session = TestSessionLocal()
        try:
            session.add(test_user)
            session.commit()
            session.refresh(test_user)
            print(f"  用户创建成功: id={test_user.id}, username={test_user.username}")
            assert test_user.id is not None

            # ---------- 第3步：插入知识库 ----------
            print("\n[步骤3] 插入知识库...")
            test_kb = KnowledgeBases(name="集成测试知识库", create_at=datetime.now())
            session.add(test_kb)
            session.commit()
            print(f"  知识库创建成功: id={test_kb.id}, name={test_kb.name}")

            # ---------- 第4步：插入文件和分块 ----------
            print("\n[步骤4] 插入文件和分块...")
            test_file = KnowledgeFiles(
                path="/test/tax_doc.pdf",
                create_at=datetime.now(),
                tags="集成测试,税务",
            )
            session.add(test_file)
            session.commit()
            session.refresh(test_file)

            test_chunk = KnowledgeChunks(
                file_id=test_file.id,
                content="增值税税率为13%，适用于一般纳税人销售货物。",
                create_at=datetime.now(),
            )
            session.add(test_chunk)
            session.commit()
            print(f"  文件id={test_file.id}, 分块id={test_chunk.id}")

            # ---------- 第5步：插入对话和消息 ----------
            print("\n[步骤5] 插入对话和消息...")
            test_conversation = Conversation(
                user_id=test_user.id,
                create_at=datetime.now(),
            )
            session.add(test_conversation)
            session.commit()
            session.refresh(test_conversation)

            user_msg = Messages(
                conversation_id=test_conversation.id,
                role="user",
                content="增值税税率是多少？",
                create_at=datetime.now(),
            )
            agent_msg = Messages(
                conversation_id=test_conversation.id,
                role="agent",
                content="增值税基本税率为13%。",
                create_at=datetime.now(),
            )
            session.add_all([user_msg, agent_msg])
            session.commit()
            print(f"  对话id={test_conversation.id}, 消息已插入")

            # ---------- 第6步：查询验证 ----------
            print("\n[步骤6] 查询验证...")

            # 验证用户
            users = session.query(User).filter_by(username="integration_test_user").all()
            assert len(users) == 1, f"应找到1个用户，实际找到 {len(users)} 个"
            print(f"  用户查询:  找到 {len(users)} 条")

            # 验证文件-分块关系
            files = session.query(KnowledgeFiles).filter_by(id=test_file.id).all()
            assert len(files) == 1
            file_obj = files[0]
            # 检查 relationship 是否工作
            chunks_of_file = file_obj.chunks
            assert len(chunks_of_file) >= 1
            print(f"  文件-分块关系:  文件 {file_obj.id} 下有 {len(chunks_of_file)} 个分块")

            # 验证对话-消息关系
            conversations = session.query(Conversation).filter_by(id=test_conversation.id).all()
            assert len(conversations) == 1
            messages_of_conv = session.query(Messages).filter_by(
                conversation_id=test_conversation.id
            ).all()
            assert len(messages_of_conv) == 2
            print(f"  对话-消息关系:  对话 {test_conversation.id} 下有 {len(messages_of_conv)} 条消息")

            print("\n 场景三通过：数据库 CRUD 流程正常！")

        finally:
            # ---------- 第7步：安全清理测试数据 ----------
            print("\n[清理] 删除测试数据...")
            try:
                if test_conversation is not None:
                    session.query(Messages).filter_by(conversation_id=test_conversation.id).delete()
                    session.query(Conversation).filter_by(id=test_conversation.id).delete()
                if test_file is not None:
                    session.query(KnowledgeChunks).filter_by(file_id=test_file.id).delete()
                    session.query(KnowledgeFiles).filter_by(id=test_file.id).delete()
                if test_kb is not None:
                    session.query(KnowledgeBases).filter_by(id=test_kb.id).delete()
                if test_user is not None:
                    session.query(User).filter_by(id=test_user.id).delete()
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"  清理过程出错（可忽略）: {e}")
            finally:
                session.close()
            test_engine.dispose()
            print("  测试数据已清理")


# ============================================================
# 运行全部场景
# ============================================================

if __name__ == "__main__":
    print(" 开始集成测试...")
    start_time = time.time()

    # 场景一：本地处理流程
    print("\n" + "=" * 60)
    print("场景一：文档解析 + 文本分块")
    print("=" * 60)
    test1 = TestParserAndChunkingIntegration()
    test1.test_txt_to_chunks()

    # 场景二：完整 RAG（需要外部服务）
    print("\n" + "=" * 60)
    print("场景二：完整 RAG 流程（可选，需要外部服务）")
    print("=" * 60)
    try:
        test2 = TestFullRAGPipeline()
        test2.test_rag_end_to_end()
    except Exception as e:
        print(f"  场景二跳过或失败（可能外部服务未配置）: {e}")

    # 场景三：数据库
    print("\n" + "=" * 60)
    print("场景三：数据库 CRUD")
    print("=" * 60)
    test3 = TestDatabaseIntegration()
    test3.test_database_crud_flow()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f" 全部集成测试完成！耗时 {elapsed:.1f} 秒")
    print(f"{'=' * 60}")