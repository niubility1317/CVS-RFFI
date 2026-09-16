"""Persist independently observed release/worker state and exact commands."""
import json,shutil
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
BASE=Path('E:/type10-7');ROOT=Path(__file__).resolve().parent
readback=json.loads((ROOT/'launch_readback_verified.json').read_text(encoding='utf-8'))
state=readback['state'];assert state['status']=='TRAINING' and not readback['dispatcher'].get('absent')
spec=json.loads((ROOT/'experiment.json').read_text(encoding='utf-8'))
old='adv3b02_controls_6a0feba47f';new=Path(state['release']).name
spec=json.loads(json.dumps(spec,ensure_ascii=False).replace(old,new))
spec['code']['commit']=state['commit'];spec['code']['cwd']=state['release']
spec['execution']['launch_command']=readback['dispatcher']['argv'][:-1]
for row in spec['rows']:
    row['status']=state['rows'][row['row_id']]['status']
    observed=state['rows'][row['row_id']]
    if observed.get('argv'):row['command']=observed['argv']
    if observed.get('pid'):row['pid']=observed['pid'];row['gpu']=observed['gpu']
    worker=readback['workers'].get(row['row_id'])
    if worker:
        assert not worker['errors'] and not worker['process'].get('absent')
        cfg=worker['resolved']
        if cfg:
            if row['row_id'].startswith('PURE_'):
                assert cfg['xuc_row']['pure_game'] and not cfg['joint']['native_dr']
            else:assert cfg['fasttrust_rc4'] and cfg['use_adv3b02_daot_stn'] and cfg['from_scratch'] and not cfg['baseline_ckpt']
(ROOT/'experiment.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
counts=dict(Counter(x['status'] for x in state['rows'].values()))
with (ROOT/'events.jsonl').open('a',encoding='utf-8') as f:
    for row in spec['rows']:f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),status=row['status'],row_id=row['row_id'],note='Independent PID/CWD/argv/GPU/log and resolved-config readback',evidence=['launch_readback_verified.json']))+'\n')
with (ROOT/'report.md').open('a',encoding='utf-8') as f:
    f.write('\n## N607启动状态：VERIFIED\n\n'+str(counts)+'；dispatcher PID='+str(state['pid'])+'。实际逐行PID/GPU/argv、resolved config与日志见launch_readback_verified.json。启动代码commit='+state['commit']+'，release='+state['release']+'。原生三seed来源均为scratch；pure_game行明确关闭DAOT/RC4，已观察到接受步。\n\n独立原问题复审PASS：序列化转换不修改原始args或训练配方；三seed回归均通过。r1失败保留，本轮未重用任何失败权重。\n')
WT=BASE/'code/snapshots/native_dr_eg_prepare_20260914_wt'
dst=WT/ROOT.relative_to(BASE)
for p in ROOT.rglob('*'):
    if p.is_file() and p.suffix not in ('.tar','.gz'):
        target=dst/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
print(json.dumps(counts))
