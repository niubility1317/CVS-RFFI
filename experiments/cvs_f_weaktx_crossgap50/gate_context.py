"""Read-only complete native U-gate context for the three matched BASE rows.

No model, IQ, target, U truth, parameter update, or new training recipe is used.
"""
import argparse,gzip,json,math
from pathlib import Path
REFERENCE_RUN='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01'
REFERENCE_COMMIT='31467086c3ced30237f3ac86e91eae68645425d7'
SEEDS=(2026092701,2026092702,2026092703)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def pass_count(rate,total):
 if not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 0<=rate<=1:raise ValueError('Finite batch gate fraction required')
 count=round(rate*total)
 if abs(count-rate*total)>2e-4:raise ValueError('Gate fraction not consistent with packet count')
 return count

def empty(epoch=None):return dict(epoch=epoch,U_observations=0,U_selected=0,temporal_pass=0,strong_pass=0,domain_pass=0,confidence_sum=0.,steps=0)

def finish(v):
 n=v['U_observations'];return dict(v,mean_teacher_confidence=v['confidence_sum']/n if n else None,acceptance=v['U_selected']/n if n else None,temporal_fraction=v['temporal_pass']/n if n else None,strong_fraction=v['strong_pass']/n if n else None,domain_fraction=v['domain_pass']/n if n else None,selected_over_temporal=v['U_selected']/v['temporal_pass'] if v['temporal_pass'] else None)

def summarize(base):
 base=Path(base)
 if base.name!=REFERENCE_RUN:raise ValueError('Only the fixed matched original F baseline')
 rows=[]
 for seed in SEEDS:
  root=base/('BASE-s'+str(seed))/'source';src=read(root/'local_source_complete.json');c=src['config']
  if src['source_commit']!=REFERENCE_COMMIT or c['model_seed']!=seed or c['treatment']!='BASE' or src['epoch']!=50 or src['optimizer_steps']!=11100 or src['initialization']!='scratch' or src['ancestors'] or src['head_fit_updates'] or not src['default_head_joint'] or src['satellite_training'] or src['target_access'] or not src['U_TX_truth_hidden']:raise ValueError('Matched original source-only F row required')
  whole=empty();epochs={};calls=0;zero_temporal=0
  with gzip.open(root/'step_metrics.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    v=json.loads(line);calls+=1;epoch=(calls-1)//222+1;m=v['metrics'];n=m['train/pseudo_total'];selected=m['train/pseudo_selected']
    if v['step']!=calls or v['epoch']!=epoch or m['train/pseudo_truth_available'] or m['train/pseudo_correct']:raise ValueError('Ordered hidden-truth native source records required')
    expected=0 if epoch<=33 else 124 if calls%222==0 else 256
    if n!=expected or not isinstance(selected,(int,float)) or selected!=int(selected) or not 0<=selected<=n:raise ValueError('Original 33L+17U observation budget differs')
    if not n:continue
    rates={key:pass_count(m['train/'+key],n) for key in ('temporal_pass','strong_pass','domain_pass')};conf=m['train/pseudo_conf']
    if not isinstance(conf,(int,float)) or not math.isfinite(conf) or not 0<=conf<=1 or selected>min(rates.values()):raise ValueError('Native gate intersection or confidence differs')
    zero_temporal+=rates['temporal_pass']==0
    for target in (whole,epochs.setdefault(epoch,empty(epoch))):
     target['steps']+=1;target['U_observations']+=n;target['U_selected']+=int(selected);target['confidence_sum']+=n*conf
     for key,value in rates.items():target[key]+=value
  if calls!=11100 or whole['steps']!=3774 or whole['U_observations']!=963900 or len(epochs)!=17:raise ValueError('Complete original50 U logs required')
  rows.append(dict(row_id='BASE-s'+str(seed),model_seed=seed,all_11100_records_parsed=True,source_commit=src['source_commit'],totals=finish(whole),epochs=[finish(v) for v in epochs.values()],zero_temporal_batches=zero_temporal))
 return dict(status='ANALYZED',reference_run=REFERENCE_RUN,reference_commit=REFERENCE_COMMIT,rows=rows,models=3,logged_steps=33300,U_observations=sum(r['totals']['U_observations'] for r in rows),IQ_reads=0,target_reads=0,U_truth_reads=0,model_forwards=0,optimizer_updates=0,head_fit_updates=0,method='Complete source step logs only; batch gate rates converted back to integer counts then pooled using true U batch sizes, including last124. Confidence weighted by U count. Observations repeat across17epochs, not unique packets.',limitations=['No per-predicted-TX or true-TX acceptance is present in these logs; cannot infer weak-TX exposure or pseudo-label precision.','Temporal agreement is conditioned on a physical neighbor in the same random batch and on predicted class/confidence; it is not a pure structural upper bound.','A high teacher mean confidence does not measure the confidence-threshold pass fraction or correctness.','This read-only context does not change current arms, source promotion, thresholds, seeds or checkpoint selection.'])

def self_check():
 assert pass_count(3/124,124)==3 and pass_count(0,256)==0 and pass_count(1,256)==256
 value=empty();value.update(U_observations=380,U_selected=5,temporal_pass=8,strong_pass=300,domain_pass=380,confidence_sum=342.,steps=2);s=finish(value)
 assert s['acceptance']==5/380 and s['mean_teacher_confidence']==.9 and s['selected_over_temporal']==.625
 for bad in (-.1,1.1,float('nan')):
  try:pass_count(bad,256)
  except ValueError:pass
  else:raise AssertionError('Invalid gate fraction accepted')
 return True

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--reference-base',required=True);p.add_argument('--output',required=True);a=p.parse_args();self_check();v=summarize(a.reference_base)
 with Path(a.output).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status=v['status'],models=3,U_observations=v['U_observations'])))
