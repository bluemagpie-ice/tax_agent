"""Local-only UI fixture. Scripted model and frozen sources; no production lifespan.

python -m tests.research_ui_server
This server is not a deployment or a tax-answer demonstration.
"""
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import create_engine
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / '.test_runs' / ('research-ui-fixture-' + uuid.uuid4().hex)
TEMP.mkdir(parents=True)
os.environ.update(
    PYTHON_DOTENV_DISABLED="1",
    DATABASE_URL=f"sqlite:///{(TEMP / 'business.sqlite').as_posix()}",
    TAX_AGENT_DATA_DIR=str(TEMP / "data"),
    TAX_AGENT_UPLOAD_DIR=str(TEMP / "uploads"),
    AUTH_USER_TOKEN="test_token_123",
    DEVELOPER_TOKEN="admin_token_456",
    AUTH_SESSION_SECRET="ui-fixture-signing-key",
    TAX_AGENT_MODEL_ALLOWED_ORIGINS="http://127.0.0.1:8767",
)

from agents.research_agent import ResearchAgent
from database.models import create_tables
from eval.research_eval import SmokeModel, SnapshotTools
from services.research_store import ResearchStore
from web.main import (
    get_current_user, get_personal_settings, login, me, test_personal_connection, update_personal_settings,
)
from web.research_routes import build_router



class UIModel(SmokeModel):
    def invoke(self, messages, **kwargs):
        import time
        time.sleep(0.35)
        schema = messages[0].content
        payload = json.loads(messages[-1].content)
        result = json.loads(super().invoke(messages, **kwargs).content)
        if 'facts' in result:
            query = payload['query']
            result['facts'] = ['模拟资料'] if '模拟资料' in query else []
        elif '"title": "Assessment"' in schema and '无证据' not in str(payload['questions']):
            evidence = next((json.loads(m.content)['evidence'] for m in messages[1:-1]
                             if m.content.startswith('{"evidence":')), None)
            if evidence:
                eid = evidence['evidence_id']
                result['coverage'] = [{'question_id':'q1', 'evidence_ids':[eid], 'sufficient':True,
                                       'excerpts':{eid:evidence['content'][:24]}}]
        elif '"title": "ResearchReport"' in schema:
            match = next(item for item in payload['coverage'] if item['sufficient'])
            result['analysis'] = [{'question':match['question'], 'evidence_ids':match['evidence_ids'],
                                   'finding':'这是界面流程的模拟分析。'}]
        return AIMessage(content=json.dumps(result, ensure_ascii=False))


@asynccontextmanager
async def lifespan(app):
    create_tables()
    engine = create_engine(f"sqlite:///{TEMP / 'business.sqlite'}", connect_args={'check_same_thread':False})
    tools = SnapshotTools(json.loads((ROOT / 'eval/research/policy_snapshot.json').read_text(encoding='utf-8'))['records'])
    async with AsyncSqliteSaver.from_conn_string(str(TEMP / 'checkpoints.sqlite')) as saver:
        app.state.agent = ResearchAgent(ResearchStore(engine),llm=UIModel(),search=tools.search,
            web_search=tools.web_search,read_page=tools.page,workdir_root=TEMP)
        app.state.agent._checkpointer = saver
        yield
    engine.dispose()


app = FastAPI(lifespan=lifespan)
app.mount('/static',StaticFiles(directory=ROOT / 'web/static'))
app.include_router(build_router(get_current_user,lambda:app.state.agent))
app.add_api_route('/api/auth/login', login, methods=['POST'])
app.add_api_route('/api/me', me, methods=['GET'])
app.add_api_route('/api/settings/personal', get_personal_settings, methods=['GET'])
app.add_api_route('/api/settings/personal', update_personal_settings, methods=['PUT'])
app.add_api_route('/api/settings/personal/connection', test_personal_connection, methods=['POST'])


@app.get('/')
async def home():
    return FileResponse(ROOT / 'web/templates/index.html')


@app.get('/api/conversations')
async def conversations(user=Depends(get_current_user)):
    return {'conversations': []}


@app.post('/fixture/v1/chat/completions')
async def provider(request: Request):
    if request.headers.get('Authorization') != 'Bearer ui-model-key':
        raise HTTPException(401, 'UI provider rejected test key')
    body = await request.json()
    return {'model': body['model'], 'choices': [
        {'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'OK'}},
    ]}


@app.get('/api/settings')
async def settings(user=Depends(get_current_user)):
    from services.config_manager import MODEL_PROVIDERS

    if user.role != 'admin':
        raise HTTPException(403, '需要管理员权限')
    return {'config': {'llm': {'provider': '自定义', 'model': 'scripted-ui-fixture',
                              'base_url': 'https://provider.example/v1/', 'api_key_set': False},
                       'qdrant': {'url': '', 'api_key_set': False, 'vector_limit': 0}},
            'providers': MODEL_PROVIDERS}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8767)
