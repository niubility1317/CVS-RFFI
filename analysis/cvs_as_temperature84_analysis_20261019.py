"""Read every completed training record and scored artifact for the fixed T1/T6 matrix."""
import argparse,gzip,json,math,csv
from collections import Counter
from pathlib import Path
RUN1='20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r01'
RUN2='20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r02'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def records(p):
 if not p.is_file():return []
 with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]
def mean(x):return sum(x)/len(x) if x else None
def collect(project):
 project=Path(project);roots={'T1':project/'runs'/RUN2/'T1-s2026092701','T6':project/'runs'/RUN1/'T6-s2026092701'};summary={};sequences={}
 for arm,b in roots.items():
  segments={};curves=[]
  for name,offset in [('stage20',0),('source',20)]:
   p=b/name
   if not (p/'completion.json').is_file():continue
   done=read(p/'completion.json');cost=read(p/'result.json');n=done['optimizer_steps'];data={k:records(p/(k+'.jsonl.gz')) for k in ('step_metrics','reference_steps','shape_steps','temperature_steps','unit_steps')};errors=[]
   if any(len(v)!=n for v in data.values()):errors.append('record_count')
   if any(r['step']!=i for v in data.values() for i,r in enumerate(v,1)):errors.append('ordered_steps')
   temp=data['temperature_steps'];expectedT=1 if arm=='T1' else 6
   if any(r['T']!=expectedT or r['As_packets']!=64 or r['As_NN']!=128 or r['As_KL']!=0 or not r['all_As_labels_supervised'] for r in temp):errors.append('actual_temperature')
   if arm=='T1' and any(abs(r['As_CE_compensated']-r['As_native_NLL'])>1e-5 for r in temp):errors.append('T1_CE_equivalence')
   metrics=[r['metrics'] for r in data['step_metrics']]
   if any(r.get('train/skipped_nonfinite_loss',0) or r.get('train/skipped_nonfinite_grad',0) for r in metrics):errors.append('nonfinite_update')
   if any(r.get('train/pseudo_truth_available',0)!=0 for r in metrics):errors.append('U_truth')
   if cost['L_NN']!=n*768 or cost['L_As']!=n*64:errors.append('L_budget')
   for ep in range(done['epoch']):
    cut=slice(ep*222,(ep+1)*222);nn=metrics[cut];tt=temp[cut];rr=data['reference_steps'][cut];ss=data['shape_steps'][cut]
    curves.append(dict(epoch=offset+ep+1,steps=len(nn),native_loss=mean([v['train/loss'] for v in nn]),As_compensated_CE=mean([v['As_CE_compensated'] for v in tt]),As_native_NLL=mean([v['As_native_NLL'] for v in tt]),As_per_example_cosine_gradient_proxy=mean([v['As_mean_logit_gradient_norm'] for v in tt]),ordinary_CE=mean([v['common_CE'] for v in rr]),mixed_CE=mean([v['extra_CE'] for v in rr]),raw_CE=mean([v['shape_CE'] for v in ss]),U_selected=mean([float(v.get('train/pseudo_selected',0)) for v in nn])))
   sequence=records(p/'training_sequence.jsonl.gz');sequences[(arm,name)]=sequence;role_counts={}
   for role in ('L','U'):
    vv=[r for r in sequence if r.get('role')==role];role_counts[role]=dict(batches=len(vv),packets=sum(len(r.get('ids',[])) for r in vv))
   native_epochs=records(p/'metrics_epoch.jsonl');csv_counts={}
   for fn in ('metrics_epoch.csv','epoch_metrics.csv'):
    with (p/fn).open(encoding='utf-8-sig',newline='') as handle:csv_counts[fn]=len(list(csv.DictReader(handle)))
   if len(native_epochs)!=done['epoch'] or any(v!=done['epoch'] for v in csv_counts.values()):errors.append('native_epoch_records')
   epochs=records(p/'epoch_metrics.jsonl');scalars={k:v for k,v in cost.items() if v is None or isinstance(v,(str,int,float,bool))};scalars['U_steps_legacy_raw']=scalars.pop('U_steps',None);scalars['actual_U_student_NN']=cost['student_train_NN']-cost['L_NN'];scalars['actual_U_batches']=role_counts['U']['batches']
   segments[name]=dict(physical_role_counts=role_counts,native_epoch_records=len(native_epochs),csv_records=csv_counts,sameAsKD=cost.get('sameAsKD'),parsed_streams={k:len(v) for k,v in data.items()},epoch_records=len(epochs),expected_epochs=done['epoch'],expected_updates=n,checks='PASS' if not errors else 'FAIL',errors=errors,cost=scalars,extra_temperature_NN=0)
  log=project/'runs'/(RUN2 if arm=='T1' else RUN1)/'logs'/(arm+'-s2026092701.log');stdout=log.read_text(errors='replace') if log.is_file() else '';summary[arm]=dict(segments=segments,epochs=curves,full_stdout=dict(lines=len(stdout.splitlines()),tracebacks=stdout.count('Traceback (most recent call last):'),OOM=stdout.count('CUDA out of memory')),complete=len(segments)==2 and (b/'complete.json').is_file())
 comparable=[name for name in ('stage20','source') if all((a,name) in sequences for a in roots)];sequence_equal={name:all(sequences[(a,name)]==sequences[(next(iter(roots)),name)] for a in roots) for name in comparable}
 return dict(physical_sequences_equal=sequence_equal,status='COMPLETE_TRAINING_LOGS' if all(v['complete'] for v in summary.values()) else 'PARTIAL_TRAINING_LOGS',rows=summary,full_structured_scan=True,sampled=False,model_seed_count=1,new_NN=0,checkpoint_reads=0,IQ_reads=0,truth_reads=0)
def pair(project):
 import numpy as np
 project=Path(project);roots={'T1':project/'runs'/RUN2/'T1-s2026092701','T6':project/'runs'/RUN1/'T6-s2026092701'}
 for b in roots.values():
  c=read(b/'prediction/complete.json');s=read(b/'scoring/complete.json')
  if c['status']!='PREDICTIONS_COMPLETE' or c['views']!=list(VIEWS) or s['status']!='SCORED_COMPLETE':raise ValueError('Require full seven-view predictions before paired truth analysis')
 data={}
 for arm,b in roots.items():
  with np.load(b/'prediction/predictions.npz',allow_pickle=False) as f:data[arm]={k:f[k].copy() for k in f.files}
 if not np.array_equal(data['T1']['ids'],data['T6']['ids']):raise ValueError('Paired formal physical IDs differ')
 from experiments.cvs_noise_shape_factorial43 import design as d
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 y,rx,day=truth_arrays(read(d.TRUTH),data['T1']['ids']);out={}
 for view in VIEWS:
  a=data['T6'][view]==y;b=data['T1'][view]==y;out[view]=dict(count=len(y),repaired=int((a&~b).sum()),new_errors=int((~a&b).sum()),net=int(a.sum()-b.sum()),delta_pp=100*float(a.mean()-b.mean()))
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();x=collect(a.project);Path(a.output).write_text(json.dumps(x,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=x['status'])))
