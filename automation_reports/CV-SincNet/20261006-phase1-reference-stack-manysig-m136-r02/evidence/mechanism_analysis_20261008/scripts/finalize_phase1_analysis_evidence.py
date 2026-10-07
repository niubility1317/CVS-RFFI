from pathlib import Path
import json,csv,statistics as st
from collections import defaultdict
from analyze_phase1_transfer_20261008 import OUT,ssh
meta=json.loads((OUT/'metadata_epochs.json').read_text(encoding='utf-8'))
r1paths=[r['source_output'] for r in meta['runs']['20261004-phase1-reference-overlay-r1-manysig-m16-r01']['source_matrix_frozen.json']['rows']]
script='''
from pathlib import Path
import json,re
paths=PATHS
results=[]
for source in paths:
 p=Path(source);root=p.parents[2];log=root.parent.parent/'logs'/root.name/('source-'+p.parent.name+'.log')
 # root is runs/<run>; derive the project without assumptions about cwd.
 project=p.parents[3];log=project/'logs'/p.parents[1].name/('source-'+p.parent.name+'.log')
 item={'source':source,'log':str(log),'log_exists':log.is_file()}
 if log.is_file():
  lines=log.read_text(errors='replace').splitlines();item['lines']=len(lines)
  item['errors']=[{'line':i+1,'text':v[:500]} for i,v in enumerate(lines) if re.search(r'Traceback \\(most recent|CUDA error|out of memory|FloatingPointError|Warning|warning|Killed',v)]
 item['resolved']=json.loads((p/'resolved_config.json').read_text())
 results.append(item)
print(json.dumps(results))
'''.replace('PATHS',repr(r1paths))
r1=json.loads(ssh(script));(OUT/'r1_stdout_readback.json').write_text(json.dumps(r1,indent=2),encoding='utf-8')
audits=[];r1epochs=[];active=[]
with (OUT/'full_steps.jsonl').open(encoding='utf-8') as f:
 for line in f:
  r=json.loads(line);isr1=r['row_id'].startswith('r1-');expected=10000 if isr1 else 44400
  assert r['steps']==expected and len(r['epochs'])==r['epoch_csv_rows']==200
  assert len(r['step_epoch_stats'])==200
  assert all(v['count']==(50 if isr1 else 222) for v in r['step_epoch_stats'].values())
  for k in ['train/w_loss_domain_labeled','train/w_loss_adv_labeled','train/w_loss_cons_labeled','train/w_loss_fishr_labeled','train/w_loss_orth_labeled','train/w_loss_proto_labeled','train/w_loss_source_episode','train/loss_unlabeled']:
   eps=[int(ep) for ep,e in r['step_epoch_stats'].items() if e['metrics'].get(k,{}).get('nonzero',0)>0]
   if eps:active.append(dict(row_id=r['row_id'],metric=k,first_epoch=min(eps),last_epoch=max(eps),active_epochs=len(eps)))
  audits.append(dict(row_id=r['row_id'],epochs=len(r['epochs']),steps=r['steps'],csv_epochs=r['epoch_csv_rows'],nonfinite_observed=r['nonfinite']))
  if isr1:r1epochs.extend([dict(row_id=r['row_id'],**e) for e in r['epochs']])
assert len(audits)==96 and len({r['row_id'] for r in audits})==96
for name,rows in [('full_log_coverage.csv',audits),('loss_activation.csv',active),('r1_epoch_curves.csv',r1epochs)]:
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
summary=dict(status='VERIFIED',completed_models=96,completed_epochs=sum(r['epochs'] for r in audits),completed_steps=sum(r['steps'] for r in audits),live_additional_epochs=sum(len(r['epochs']) for r in meta['rows'] if not r['completion']),
 r1_stdout_complete=sum(r['log_exists'] for r in r1),stack_stdout_complete=sum('stdout' in r for r in meta['rows']),r1_stdout_errors=sum(len(r.get('errors',[])) for r in r1),stack_stdout_errors=sum(len(r.get('stdout',{}).get('matches',{}).get('error',[])) for r in meta['rows']),
 completed_step_scope='all 80 completed stack rows and all 16 R1 rows; complete JSON records parsed; scalar losses/gradient/pseudo statistics aggregated',live_step_scope='live step gzip not read; epoch snapshot only',raw_evidence_root=str(OUT),target_reprediction=False,remote_modifications=False)
(OUT/'log_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary));print('ACTIVATION',[r for r in active if r['row_id'].endswith('2026092701')])
