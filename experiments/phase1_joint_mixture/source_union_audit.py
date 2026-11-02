"""Truth-last union audit of already fixed Phase1 source L_s predictions."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from experiments.pamf_phase1_flow import feature_conditioned_experiment as labels_source
from source_validation import s,core,VIEWS

RUNS={'hierarchical':'20261102-phase1-hierarchical-prior-labeled-source-s2026092701-r01','joint':'20261102-phase1-joint-mixture-labeled-source-s2026092701-r01','free':'20261102-phase1-free-mixture-labeled-source-s2026092701-r01'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable audit output required')
 a.out.mkdir(parents=True);data={}
 for name,run in RUNS.items():
  folder=a.root/'runs'/run;done=core.read(folder/'prediction_complete.json')
  if done.get('status')!='SOURCE_PREDICTIONS_FIXED' or done.get('role')!='L_s' or done.get('TX_truth_reads')!=0 or done.get('count')!=1047:raise ValueError('Fixed source L_s predictions required')
  with np.load(folder/'predictions.npz',allow_pickle=False) as z:data[name]={k:z[k] for k in z.files}
 base=data['hierarchical'];idx=base['indices']
 for name,d in data.items():
  for k in ('indices','physical_ids','rx','scene','part','active'):
   if not np.array_equal(d[k],base[k]):raise ValueError('Identical source records and gate required')
  if not np.array_equal(d['predictions'][:,:4],base['predictions'][:,:4]):raise ValueError('Shared clean, raw, DURA and parent predictions required')
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
 if np.any(roles[idx]!='L_s') or not np.array_equal(physical[idx],base['physical_ids']) or np.intersect1d(idx,training).size or not np.array_equal(idx,np.r_[validation[roles[validation]=='L_s'],heldout[roles[heldout]=='L_s']]):raise ValueError('Complete legal source L_s cohort required')
 teacher,contract=core.teacher(a.root,'cpu');truth=labels_source.source_labels(a.root,cache,clean,roles,contract)[idx]
 if np.any((truth<0)|(truth>=6)):raise ValueError('Valid source labels required')
 pred={'clean':base['predictions'][:,0],'raw':base['predictions'][:,1],'dura':base['predictions'][:,2],'parent':base['predictions'][:,3],'hierarchical':data['hierarchical']['predictions'][:,4],'joint':data['joint']['predictions'][:,4],'free':data['free']['predictions'][:,4]};good={k:v==truth for k,v in pred.items()};names=('parent','hierarchical','joint','free');union=np.logical_or.reduce([good[k] for k in names]);union_dura=union|good['dura'];cells={}
 for part,receivers in ((1,(4,6)),(2,(8,))):
  for view in VIEWS:
   mask=(base['part']==part)&np.isin(base['rx'],receivers)&(base['scene']==view);n=int(mask.sum())
   if n==0:raise ValueError('All source cells required')
   cells[f'{part}:{view}']={'count':n,'correct':{k:int((mask&v).sum()) for k,v in good.items()},'physical_union_oracle':int((mask&union).sum()),'physical_plus_dura_union_oracle':int((mask&union_dura).sum()),'physical_extra_over_dura':int((mask&union&~good['dura']).sum()),'physical_misses_recovered_by_dura':int((mask&~union&good['dura']).sum())}
 result={'status':'SOURCE_LS_UNION_AUDIT_COMPLETE','commit':a.commit,'count':len(idx),'cells':cells,'truth_role':'L_s','source_only':True,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'caveat':'Per-record oracle uses source true labels and is not deployable; applies only to these fixed outputs and explored source IDs; no test generalization or other-model upper bound.'}
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('SOURCE_LS_UNION_AUDIT_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
