"""Local-only UI fixture. Scripted model and frozen sources; no production lifespan.

python -m tests.research_ui_server
This server is not a deployment or a tax-answer demonstration.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import create_engine
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from agents.research_agent import ResearchAgent
from eval.research_eval import SmokeModel, SnapshotTools
from services.research_store import ResearchStore
from services.security import tenant_scope
from web.research_routes import build_router

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / '.test_runs' / 'research-ui-fixture'
TEMP.mkdir(parents=True, exist_ok=True)


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


async def auth(authorization: str = Header(...)):
    identities = {'Bearer test_token_123':'ui-alice','Bearer admin_token_456':'ui-bob'}
    if authorization not in identities:
        raise HTTPException(401,'UI fixture credential required')
    with tenant_scope(identities[authorization]):
        yield


@asynccontextmanager
async def lifespan(app):
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
app.include_router(build_router(auth,lambda:app.state.agent))


@app.get('/')
async def home():
    return FileResponse(ROOT / 'web/templates/index.html')


@app.get('/api/conversations')
async def conversations():
    return {'sessions':[]}


@app.get('/api/me')
async def me():
    return {'id':'ui-alice','user_name':'隔离界面测试','role':'user'}


@app.get('/api/settings')
async def settings():
    return {'llm':{'model':'scripted-ui-fixture','api_key_set':False},'model_providers':[]}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8767)
