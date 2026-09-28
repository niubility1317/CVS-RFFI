"""Mirror explicit per-run records without touching unrelated workspace edits."""
import argparse
import json
import re
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WS=Path('E:/type10-7')


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--note',required=True)
    a=p.parse_args();spec=json.loads(a.spec.read_text(encoding='utf-8'))
    rel=Path('automation_reports/CV-SincNet')/spec['run_id'];folder=WS/rel
    record=json.loads((folder/'experiment.json').read_text(encoding='utf-8'))
    record.update(spec)
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    record['status']=next(e['status'] for e in reversed(events) if not e.get('row_id'))
    evidence=sorted((folder/'evidence').glob('readback_*.json'))
    if evidence:
        observed=json.loads(evidence[-1].read_text(encoding='utf-8')).get('startup.json',{})
        if observed:
            record['code']['release_commit']=observed['commit']
            record['execution'].update(pid=observed['pid'],runtime_command=observed['argv'])
    (folder/'experiment.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=(folder/'report.md').read_text(encoding='utf-8')
    report=re.sub(r'当前登记状态：[A-Z_]+', '当前登记状态：'+record['status'], report)
    (folder/'report.md').write_text(report,encoding='utf-8')
    if a.note not in report:
        with (folder/'report.md').open('a',encoding='utf-8') as f:f.write('\n'+a.note+'\n')
    for src in folder.rglob('*'):
        if src.is_file():
            dst=ROOT/rel/src.relative_to(folder);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    index=ROOT/'experiment_registry/d92_upgrade_runs_20260928.jsonl'
    entries=[json.loads(s) for s in index.read_text(encoding='utf-8').splitlines()] if index.exists() else []
    entries=[r for r in entries if r['run_id']!=spec['run_id']]
    entries.append(dict(run_id=spec['run_id'],group_id=spec['group_id'],status=record['status'],report=str(rel/'report.md'),goal_complete=False))
    index.write_text(''.join(json.dumps(r)+'\n' for r in entries),encoding='utf-8')
    print(json.dumps(dict(run_id=spec['run_id'],status=record['status'],mirror=str(ROOT/rel))))


if __name__=='__main__':main()
