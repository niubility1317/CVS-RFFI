"""Persist observed source-only result, evidence and explicit project mirrors."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WS=Path('E:/type10-7')
RUN='20260928-diagnostic-d92-scv-source-s2026092701-r01'
RELEASE_COMMIT='8d2697752c84af4763372bb6e4c4ba6a621bceef'


def main():
    folder=Path('automation_reports/CV-SincNet')/RUN
    src=WS/folder
    record=json.loads((src/'experiment.json').read_text(encoding='utf-8'))
    record['code']['release_commit']=RELEASE_COMMIT
    record['execution']['pid']=2331888
    record['execution']['runtime_command']=record['rows'][0]['command'].replace(record['code']['commit'],RELEASE_COMMIT)
    record['rows'][0]['status']='ANALYZED'
    record['rows'][0]['pid']=2331888
    record['rows'][0]['command']=record['execution']['runtime_command']
    (src/'experiment.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=(src/'report.md').read_text(encoding='utf-8').replace('当前登记状态：PLANNED','当前登记状态：ANALYZED')
    (src/'report.md').write_text(report,encoding='utf-8')
    for name in ['experiment.json','report.md','events.jsonl','results/summary.json','results/compact.jsonl','results/per_cell.csv',
                 'evidence/readback_1790597216.json','evidence/readback_1790597406.json']:
        to=ROOT/folder/name;to.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src/name,to)
    shutil.copyfile(WS/'docs/D92_SUPPORT_UPGRADE_20260928.md',ROOT/'docs/D92_SUPPORT_UPGRADE_20260928.md')
    inventory=WS/'local_artifacts/d92_upgrade_20260928/confirmation_inventory.json'
    shutil.copyfile(inventory,ROOT/folder/'results/confirmation_inventory.json')
    for name in ['tools/read_d92_source_development.py','tools/inspect_d92_confirmation_inventory.py',
                 'tools/predict_d92_support_cv.py','tools/summarize_d92_source_development.py',
                 'tests/test_predict_d92_support_cv.py']:
        target=WS/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    index=ROOT/'experiment_registry/d92_upgrade_20260928.jsonl'
    d=json.loads(index.read_text(encoding='utf-8'));d['status']='ANALYZED';d['goal_complete']=False
    index.write_text(json.dumps(d)+'\n',encoding='utf-8')
    paths=[ROOT/folder/'report.md',ROOT/folder/'experiment.json',ROOT/'docs/D92_SUPPORT_UPGRADE_20260928.md']
    for p in paths:
        text=p.read_text(encoding='utf-8')
        if text.startswith('\ufeff') or '\ufffd' in text:raise ValueError('Encoding corruption')
    print('DELIVERY_READBACK_AND_UTF8_VERIFIED')


if __name__=='__main__':main()
