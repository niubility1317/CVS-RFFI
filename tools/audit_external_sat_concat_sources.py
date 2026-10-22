import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'local_artifacts/external_sat_concat_audit_20261023';sys.path.insert(0,str(ROOT))
from experiments.cric_dura_comparison.publish import ssh
from experiments.phase1_external_dg_aug.concat_audit import FAMILIES,audit_selected
oldroot=ROOT/'local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019'
oldrt=json.loads((oldroot/'runtime.json').read_text(encoding='utf-8'))
scroot=ROOT/'local_artifacts/phase1_dg_sourcecount6_20261021'
scrt=json.loads((scroot/'runtime.json').read_text(encoding='utf-8'))
selections=[]
for p in sorted((oldroot/'verified_rows').glob('*.json')):
    d=json.loads(p.read_text(encoding='utf-8'));c=d['config']
    if c['family'] in FAMILIES:selections.append(dict(suite='native219',run_id=oldrt['run_id'],root=oldrt['run_root'],config=c,cost=d['cost'],selected=audit_selected(c,d['cost'])))
for d in json.loads((scroot/'final_360_complete/audited_rows.json').read_text(encoding='utf-8'))['rows']:
    c=d['config'];selections.append(dict(suite='sourcecount360',run_id=scrt['run_id'],root=scrt['run_root'],config=c,cost=d['cost'],selected=audit_selected(c,d['cost'])))
assert len(selections)==516 and all(r['selected']['status']=='PASS' for r in selections)
(P/'selected_source_rows.json').write_text(json.dumps(selections,indent=2)+'\n',encoding='utf-8')
audit_code=(ROOT/'experiments/phase1_external_dg_aug/concat_audit.py').read_text(encoding='utf-8')
reduced=[dict(suite=r['suite'],run_id=r['run_id'],root=r['root'],row_id=r['config']['row_id'],expected=r['config']) for r in selections]
result=ssh(r'''
import json,time
from pathlib import Path
ns={};exec(AUDIT_CODE,ns);rows=[]
for item in ITEMS:
 folder=Path(item['root'])/item['row_id']/'source'
 freeze=json.loads((folder/'freeze.json').read_text());c=freeze['config']
 stages=[]
 for stage in freeze['stages']:
  records=[json.loads(s) for s in (folder/('stage%d'%stage['stage'])/'epochs.jsonl').read_text().splitlines() if s.strip()]
  stages.append(ns['audit_stage'](c,stage,records))
 rows.append(dict(suite=item['suite'],run_id=item['run_id'],row_id=item['row_id'],family=c['family'],variant=c['variant'],
  regime=c['training_regime'],model_seed=c['model_seed'],source_count=c.get('source_count'),
  label_budget=c.get('label_budget'),protocol_id=c['protocol_id'],stages=stages,
  config_matches_stored_row=all(c.get(k)==v for k,v in item['expected'].items()),checkpoint_exists=Path(freeze['checkpoint']).is_file()))
print(json.dumps(dict(status='SOURCE_LOGS_READ_COMPLETE',observed_at=time.time(),rows=rows,target_reads=0,model_updates=0)))
'''.replace('AUDIT_CODE',repr(audit_code)).replace('ITEMS',repr(reduced)))
bykey={(r['run_id'],r['config']['row_id']):r for r in selections}
for row in result['rows']:
    local=bykey[(row['run_id'],row['row_id'])];row['selected']=local['selected']
    row['status']='PASS' if row['config_matches_stored_row'] and row['checkpoint_exists'] and row['selected']['status']=='PASS' and all(s['status']=='PASS' for s in row['stages']) else 'FAIL'
result['status']='PASS' if all(r['status']=='PASS' for r in result['rows']) else 'FAIL'
(P/'source_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],models=len(result['rows']),stages=sum(len(r['stages']) for r in result['rows']),
    records=sum(s['records'] for r in result['rows'] for s in r['stages']),
    failures=[dict(row=r['row_id'],issues=[s['issues'] for s in r['stages']]) for r in result['rows'] if r['status']!='PASS'],target_reads=0)))
