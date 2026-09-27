from pathlib import Path
import json,gzip,csv,statistics,collections
O=Path(__file__).resolve().parent;R=Path('E:/type10-7')
d=json.loads(gzip.decompress((O/'remote_evidence_20260927.json.gz').read_bytes()))
for n,rd in d['runs'].items():
 for rel,item in rd['files'].items():
  if rel.endswith('score.json'):print('SCORE',rel,item['data'])
 if 'pure_game' in n:
  ss=rd['stdout_scans'];print('STDOUT examples',[(s['path'],s['matches'][:2]) for s in ss if s['matches']][:3])
  for rel,item in rd['files'].items():
   if rel.endswith('metrics_epoch.jsonl'):
    rows=item['records'];keys=[k for k in rows[-1] if any(t in k for t in ['skip','nonfinite','epoch','step','loss'])]
    print('NATIVEKEYS',rel,{k:rows[-1][k] for k in keys if 'skip' in k or 'nonfinite' in k})
for p in (R/'local_artifacts').glob('*response*'):
 print('LOCAL_RESPONSE',p.name)
 for row in ['CF_EG_s392007','R0-11_s392007','ADV0-1_s392007']:
  print(row,[(str(f.relative_to(p)),f.stat().st_size) for f in (p/row).glob('*score*.json')])
for name in ['day0_seed_summary.csv','day0_summary.csv']:
 with (R/'automation_reports/CV-SincNet/response_matrix_eval_20260917/results'/name).open(encoding='utf-8-sig') as f:rs=list(csv.DictReader(f))
 print(name,rs[0])
 if name.startswith('day0_seed'):print([x for x in rs if x['metric']=='leo_mean_accuracy'])
