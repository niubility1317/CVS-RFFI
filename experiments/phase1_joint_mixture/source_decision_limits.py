"""Source-only cross-receiver diagnostic of information in candidate decisions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def key_for(row,gate,use_gate):return tuple(map(int,row))+((int(gate),) if use_gate else ())
def fit_mapping(candidates,gate,labels,use_gate):
 if candidates.ndim!=2 or candidates.shape[1]!=5 or gate.shape!=(len(candidates),) or labels.shape!=(len(candidates),):raise ValueError('Five source candidates and training labels required')
 counts={};orders={}
 for row,on,y in zip(candidates,gate,labels):
  key=key_for(row,on,use_gate);orders.setdefault(key,list(dict.fromkeys(map(int,row))));counts.setdefault(key,{c:0 for c in orders[key]})
  if int(y) in counts[key]:counts[key][int(y)]+=1
 return {k:max(orders[k],key=lambda c:counts[k][c]) for k in counts}

def apply_mapping(mapping,candidates,gate,use_gate):
 out=[];unseen=[]
 for row,on in zip(candidates,gate):
  key=key_for(row,on,use_gate);unseen.append(key not in mapping);out.append(mapping.get(key,int(row[1])))
 return np.asarray(out,dtype=np.int64),np.asarray(unseen,dtype=bool)

def pattern_upper(candidates,gate,labels,use_gate):
 counts={};orders={}
 for row,on,y in zip(candidates,gate,labels):
  key=key_for(row,on,use_gate);orders.setdefault(key,set(map(int,row)));counts.setdefault(key,{c:0 for c in orders[key]})
  if int(y) in counts[key]:counts[key][int(y)]+=1
 return sum(max(v.values()) for v in counts.values())

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source diagnostic required')
 runs=a.root/'runs';prior=runs/'20261102-phase1-unroll-source-replay-s2026092701-r01';m=json.loads((prior/'prediction_complete.json').read_text());path=prior/'source_predictions.npz';expected='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5'
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('sha256')!=expected or sha(path)!=expected or any(m.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed source candidate predictions required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx=z['rx'];scene=z['scene'];values=z['values']
 if ids.shape!=(11761,) or len(np.unique(ids))!=11761 or values.shape!=(11761,15) or not np.isfinite(values).all():raise ValueError('Complete unique source predictions required')
 cache=runs/'20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False);known=roles[ids]=='L_s';unknown=roles[ids]=='U_s'
 if int(known.sum())!=1047 or not np.all(known|unknown):raise ValueError('Expected source roles required')
 meta=runs/'20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:lsids=z['ids'];labels=z['y']
 if lsids.tolist()!=contract['role_ids']['L_s'] or len(lsids)!=6300 or np.any((labels<0)|(labels>=6)):raise ValueError('Exact legal source labels required')
 lookup={str(k):int(v) for k,v in zip(lsids,labels)}
 if len(lookup)!=6300 or not all(str(x) in lookup for x in physical[ids[known]]) or not all(str(x) not in lookup for x in physical[ids[unknown]]):raise ValueError('Source role label binding failed')
 y=np.full(len(ids),-1,dtype=np.int64);y[known]=[lookup[str(x)] for x in physical[ids[known]]];gate=values[:,3]>.8;raw=values[:,5].astype(np.int64);candidates=np.column_stack((raw,np.where(gate,values[:,7],raw),*[np.where(gate,values[:,i],raw) for i in (9,11,13)])).astype(np.int64)
 if not np.isin(candidates,np.arange(6)).all():raise ValueError('Valid fixed six-class predictions required')
 route=np.full(len(ids),-1,dtype=np.int64);route_gate=route.copy();miss=np.zeros(len(ids),dtype=bool);miss_gate=miss.copy();folds={}
 for name,train_rx,test_rx in [('train4_eval6',(4,),(6,)),('train6_eval4',(6,),(4,)),('train46_eval8',(4,6),(8,))]:
  tr=known&np.isin(rx,train_rx);ev=known&np.isin(rx,test_rx)
  if np.any(tr&ev) or not tr.any() or not ev.any():raise ValueError('Disjoint source receiver folds required')
  plain=fit_mapping(candidates[tr],gate[tr],y[tr],False);guard=fit_mapping(candidates[tr],gate[tr],y[tr],True);route[ev],miss[ev]=apply_mapping(plain,candidates[ev],gate[ev],False);route_gate[ev],miss_gate[ev]=apply_mapping(guard,candidates[ev],gate[ev],True);folds[name]=dict(train_receivers=list(train_rx),eval_receivers=list(test_rx),train_L_s=int(tr.sum()),eval_L_s=int(ev.sum()),patterns=len(plain),gate_patterns=len(guard))
 if np.any(route[known]<0) or np.any(route_gate[known]<0) or not np.all(np.any(candidates[known]==route[known,None],axis=1)) or not np.all(np.any(candidates[known]==route_gate[known,None],axis=1)):raise ValueError('Every heldout decision must select a candidate')
 views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired'];results={}
 for group,receivers in [('RX4_6',(4,6)),('RX8',(8,))]:
  results[group]={}
  for view in views:
   mask=known&np.isin(rx,receivers)&((scene!='clean') if view=='all_impaired' else (scene==view));c=candidates[mask];truth=y[mask];g=gate[mask];n=int(mask.sum());cell=dict(count=n,raw_correct=int(np.count_nonzero(c[:,0]==truth)),physical_correct=int(np.count_nonzero(c[:,1]==truth)),unroll025_correct=int(np.count_nonzero(c[:,2]==truth)),unroll05_correct=int(np.count_nonzero(c[:,3]==truth)),unroll1_correct=int(np.count_nonzero(c[:,4]==truth)),crossfit_pattern_correct=int(np.count_nonzero(route[mask]==truth)),crossfit_gate_correct=int(np.count_nonzero(route_gate[mask]==truth)),unseen_pattern=int(miss[mask].sum()),unseen_gate_pattern=int(miss_gate[mask].sum()),label_oracle=int(np.count_nonzero(np.any(c==truth[:,None],axis=1))),same_cell_pattern_upper=pattern_upper(c,g,truth,False),same_cell_gate_upper=pattern_upper(c,g,truth,True));results[group][view]=cell
 a.out.mkdir(parents=True);out=dict(status='SOURCE_DECISION_LIMITS_COMPLETE',commit=a.commit,prior_sha256=expected,L_s_scored=int(known.sum()),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,DURA_in_candidate=False,folds=folds,results=results,caveat='Posthoc source diagnostic on previously explored source IDs. Crossfit uses source L_s from other receivers; upper bounds use same-cell labels and cannot be deployed. F was trained on source.')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
