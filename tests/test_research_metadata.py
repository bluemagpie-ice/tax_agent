from types import SimpleNamespace

from knowledge.policy_updater import PolicyItem, PolicyUpdater
from services.research_evidence import normalize_evidence


def test_policy_ingestion_preserves_official_provenance_without_inventing_effective_date(tmp_path):
    path=tmp_path / 'public.md'
    path.write_text('公开模拟资料',encoding='utf-8')
    captured={}
    async def ingest(file_path,**kwargs):
        captured.update(kwargs)
        return {'status':'success','chunk_count':1}
    updater=object.__new__(PolicyUpdater)
    updater.tracker=SimpleNamespace(file_id_for=lambda key: "")
    updater.chunk_preset_id='general'
    item=PolicyItem(source_id='public',item_id='p1',title='公开资料',file_path=str(path),
                    url='https://www.chinatax.gov.cn/public',published_date='2024-01-01',
                    metadata={'issuer':'模拟机关','region':'全国'})
    updater._ingest_file(item,SimpleNamespace(ingest_file=ingest))
    assert captured['source_url'] == item.url and captured['publish_date'] == '2024-01-01'
    assert captured['issuer'] == '模拟机关' and captured['region'] == '全国'
    assert 'effective_date' not in captured and 'is_current' not in captured
    evidence=normalize_evidence({**captured,'content':'实际读取节选'})
    assert evidence['source_check'] == 'official_domain'
    assert evidence['effective_date'] is None and evidence['is_current'] is None
    assert evidence['validity_check'] == evidence['applicability_check'] == 'unknown'
