"""Read complete support optimization traces; do not read target predictions/truth."""
import argparse
import json
from pathlib import Path
import subprocess
from read_d92_run import FLAGS

REMOTE=r'''
import json,math,statistics
from collections import Counter
from pathlib import Path
c=CONFIG
root=Path(c['root'])
assert json.loads((root/'complete.json').read_text())['status']=='SCORED'
models=[]
for name in c['rows']:
 path=root/name/'sfhead/fit_trace.jsonl'
 rows=[json.loads(s) for s in path.read_text().splitlines()]
 assert len(rows)==c['splits'] and len({r['split_id'] for r in rows})==c['splits']
 for r in rows:
  assert r['query_rows_used_for_fit']==0 and r['source_runtime_access'] is False and r['encoder_updated'] is False
  assert r['ground_used_for_fit'] is False and r['new_ground_payload_bytes']==0
  assert len(r['steps'])==r['optimizer_steps'] and r['persistent_state_bytes']==8*r['class_count']*161
  assert math.isfinite(r['final']['loss']) and math.isfinite(r['final']['gradient_inf'])
 groups=[]
 for k in sorted({r['k'] for r in rows}):
  g=[r for r in rows if r['k']==k]
  groups.append(dict(k=k,fits=len(g),converged=sum(r['converged'] for r in g),
   status_counts=dict(Counter(r['optimizer_status'] for r in g)),
   termination_counts=dict(Counter(r['termination_message'] for r in g)),
   mean_steps=statistics.mean(r['optimizer_steps'] for r in g),max_steps=max(r['optimizer_steps'] for r in g),
   mean_fit_seconds=statistics.mean(r['fit_seconds'] for r in g),max_fit_seconds=max(r['fit_seconds'] for r in g),
   max_final_gradient_inf=max(r['final']['gradient_inf'] for r in g),
   mean_initial_loss=statistics.mean(r['initial']['loss'] for r in g),mean_final_loss=statistics.mean(r['final']['loss'] for r in g)))
 models.append(dict(row_id=name,trace_file=str(path),trace_file_bytes=path.stat().st_size,
  fits=len(rows),converged=sum(r['converged'] for r in rows),
  total_fit_seconds=sum(r['fit_seconds'] for r in rows),per_k=groups,
  state_bytes_by_classes={str(r['class_count']):r['persistent_state_bytes'] for r in rows}))
print(json.dumps(dict(status='VERIFIED',scope='complete support-only optimizer traces; no query scores read',
 models=models,total_fits=sum(m['fits'] for m in models),total_converged=sum(m['converged'] for m in models),
 new_ground_statistics_bytes=0)))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    spec=json.loads(a.spec.read_text(encoding='utf-8'))
    if spec['confirmation'].get('candidate_method')!='D92-SFHead-v1':raise ValueError('SFHead only')
    c=dict(root=spec['execution']['remote_run_root'],rows=[r['row_id'] for r in spec['rows']],splits=spec['confirmation']['splits_per_model'])
    script=REMOTE.replace('CONFIG',repr(c));compile(script,'remote','exec')
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode(),capture_output=True,check=True)
    data=json.loads(result.stdout)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(data,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in data.items() if k!='models'}))

if __name__=='__main__':main()
