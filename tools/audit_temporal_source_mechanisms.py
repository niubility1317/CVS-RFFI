"""Real temporal initial states, source-only phases and hidden-U feedback."""
import argparse,json
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ssh
REMOTE=r'''
import sys,json,time,torch
from pathlib import Path
c=CONFIG;root=Path(c['project'])/'releases'/c['release'];sys.path[:0]=[str(root),str(root/'code')]
from experiments.cvs_time_factorial50 import design as d
from experiments.cvs_time_factorial50.provenance import source_proof
rows=[r for r in d.rows() if r['arm']!='BASE_REF'];initial={}
for row in rows:
 cfg=d.config(row);path=Path(cfg['output_root'])/'initial_smoke.pt'
 if not path.is_file():raise ValueError('Initial checkpoint missing '+row['row_id'])
 initial[row['arm'],row['model_seed']]=torch.load(path,map_location='cpu',weights_only=False)['model']
matched=[]
for seed in d.SEEDS:
 reference=initial['COMPACT_LINEAR',seed];count=0
 for arm in d.ARMS[1:]:
  state=initial[arm,seed]
  if set(state)!=set(reference):raise ValueError('Initial state schema differs')
  for name,value in state.items():
   if not torch.equal(value,reference[name]):raise ValueError('Actual initial state mismatch '+str(seed)+'/'+arm+'/'+name)
  count=len(state)
 matched.append(dict(seed=seed,arms=4,initial_parameter_buffer_tensors=count,all_tensor_values_equal=True))
result=dict(run_id=d.RUN,status='VERIFIED',read_at=time.time(),execution_commit=(root/'release_commit.txt').read_text().strip(),actual_initial_matches=matched,complete_source_rows=[],historical=[],target_IQ_or_truth_read=False,training_or_head_updates=0)
for row in rows:
 cfg=d.config(row);source=Path(cfg['output_root'])
 if not (source/'completion.json').is_file():continue
 source_proof(cfg);stages=[d.read(source/f'source_stage_epoch{e}.json') for e in (32,33,34,40,50)];feedback=[json.loads(s) for s in (source/'source_mechanism_epochs.jsonl').read_text().splitlines()]
 if len(feedback)!=50:raise ValueError('Source feedback incomplete')
 aggregate={}
 for epoch in feedback:
  for key,value in epoch['groups'].items():
   g=aggregate.setdefault(key,dict(total=0,accepted=0,agreement=0,predicted_class=[0]*6,accepted_class=[0]*6))
   for n in ('total','accepted','agreement'):g[n]+=value[n]
   for n in ('predicted_class','accepted_class'):g[n]=[a+b for a,b in zip(g[n],value[n])]
 if sum(g['total'] for g in aggregate.values())!=963900:raise ValueError('U observations budget differs')
 for g in aggregate.values():g['accept_rate']=g['accepted']/g['total'];g['agreement_rate']=g['agreement']/g['total']
 result['complete_source_rows'].append(dict(**row,stages=stages,feedback_by_epoch=feedback,U_aggregate=aggregate,U_truth_read=False,mask_rule_changed=False,head_fit_updates=0))
for path in sorted((d.BASE/'historical_time_diagnostics').glob('*/summary.json')):
 result['historical'].append(d.read(path))
print(json.dumps(result,allow_nan=False))
'''
def collect(output):
 from experiments.cvs_time_factorial50 import design as d
 payload=REMOTE.replace('CONFIG',repr(dict(project=d.PROJECT,release=d.RELEASE)))
 wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True,env=dict(__import__('os').environ,CUDA_VISIBLE_DEVICES=''))\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
 v=json.loads(ssh(wrapper));p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(status=v['status'],initial_seeds=len(v['actual_initial_matches']),source_complete_rows=len(v['complete_source_rows']),historical_rows=len(v['historical']))));return v
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.output)
