"""Copy completed frozen predictions/scores; never retrieve source IQ or truth maps."""
import argparse
import json
from pathlib import Path
import subprocess
from experiments.cric_dura_comparison.publish import CONNECTION,ssh
from .confirm import RUN,validate_context


def collect(runtime,output):
    rt=json.loads(Path(runtime).read_text(encoding='utf-8'))
    if rt['run_id']!=RUN:raise ValueError('Only this confirmation run is accepted')
    script=r'''
import json
from pathlib import Path
rt=RUNTIME
root=Path(rt['run_root']);ctx=json.loads(Path(rt['context']).read_text());done=json.loads((root/'complete.json').read_text())
if done['status']!='UHISTORY_CONFIRMATION_COMPLETE':raise ValueError('Wait for fixed prediction and independent score completion')
files=[];resolved={}
def add(path,key):
 if not path.is_file():raise FileNotFoundError(path)
 files.append(dict(remote=str(path),local=key,bytes=path.stat().st_size))
for name in ('complete.json','pipeline_state.json','source_checkpoint_smoke.json'):add(root/name,'candidate/'+name)
add(root/'scores/scores.json','candidate/scores.json')
add(root/'predictions/complete.json','candidate/predictions/complete.json')
for task in ctx['tests']:add(root/'predictions'/task/'predictions.npz','candidate/predictions/'+task+'/predictions.npz')
for method,path in ctx['reference_predictions'].items():
 add(Path(path)/'complete.json','references/'+method+'/predictions/complete.json')
 for task in ctx['tests']:add(Path(path)/task/'predictions.npz','references/'+method+'/predictions/'+task+'/predictions.npz')
 score=Path(ctx['reference_scores'][method])
 if not score.is_file():
  # Two original F0 supervised score entries were recovered historically. Read
  # the already completed jobs; never recompute predictions or scores here.
  jobs=json.loads((Path(rt['project'])/'runs/20261025-phase1-PAMF-REPL-control-r03/jobs.json').read_text());matches=[]
  for job in jobs:
   if job['kind']!='score':continue
   cfg=json.loads(Path(job['config']).read_text());marker=Path(job['completion'])
   if cfg.get('row_id')!='F0-'+method+'-s2026092701' or not marker.is_file():continue
   done=json.loads(marker.read_text())
   if done.get('status')=='SCORED_COMPLETE':matches.append(marker.parent/'scores.json')
  if len(matches)!=1:raise ValueError('Exactly one completed historical reference score required: '+method)
  score=matches[0]
 header=json.loads(score.read_text())
 if header.get('status')!='SCORED' or header.get('truth_last') is not True or header.get('row_id')!='F0-'+method+'-s2026092701' or header.get('model_seed')!=2026092701:
  raise ValueError('Historical reference score identity differs')
 resolved[method]=str(score);add(score,'references/'+method+'/scores.json')
print(json.dumps(dict(files=files,resolved_reference_scores=resolved,status='FROZEN_INDEPENDENT_SCORES_COMPLETE',truth_maps_retrieved=False)))
'''.replace('RUNTIME',repr(rt))
    data=ssh(script);root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    for row in data['files']:
        p=(root/row['local']).resolve()
        if root not in p.parents:raise ValueError('Readback path escapes local output')
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.stat().st_size==row['bytes']:continue
        subprocess.run(['scp',*CONNECTION,'N607:'+row['remote'],str(p)],check=True,timeout=60,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    (root/'manifest.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(data['files']),output=str(root),truth_maps_retrieved=False,new_inference=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();collect(a.runtime,a.output)
