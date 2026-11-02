"""Independently recount source-only cross-receiver candidate decisions."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 runs=a.root/'runs';path=runs/'20261102-phase1-unroll-source-replay-s2026092701-r01/source_predictions.npz';res=json.loads((a.source/'result.json').read_text())
 if res.get('status')!='SOURCE_DECISION_LIMITS_COMPLETE' or res.get('commit')!='f2699a55d59c6ad20fbad3cdae0747276bc34830' or res.get('prior_sha256')!=sha(path) or any(res.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact fixed source diagnostic required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx=z['rx'];scene=z['scene'];v=z['values']
 cache=runs/'20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False);meta=runs/'20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:keys=z['ids'].copy();labels=z['y'].copy()
 if keys.tolist()!=contract['role_ids']['L_s'] or len(keys)!=6300:raise ValueError('Exact source identity contract required')
 key=np.asarray([str(x) for x in keys]);order=np.argsort(key);ks=key[order];query=np.asarray([str(x) for x in physical[ids]]);pos=np.searchsorted(ks,query);present=(pos<len(ks))&(ks[np.minimum(pos,len(ks)-1)]==query);known=roles[ids]=='L_s'
 if len(set(key.tolist()))!=6300 or int(known.sum())!=1047 or not np.array_equal(present,known) or not np.all(known|(roles[ids]=='U_s')):raise ValueError('Independent source role join failed')
 y=np.full(len(ids),-1,dtype=np.int64);y[known]=labels[order[pos[known]]];gate=v[:,3]>.8;raw=v[:,5].astype(int);cand=np.column_stack((raw,np.where(gate,v[:,7],raw),*[np.where(gate,v[:,j],raw) for j in (9,11,13)])).astype(int);route=np.full(len(ids),-1);route_gate=route.copy();miss=np.zeros(len(ids),bool);miss_gate=miss.copy()
 def key_for(i,use_gate):return tuple(map(int,cand[i]))+((int(gate[i]),) if use_gate else ())
 for name,train_rx,eval_rx in [('train4_eval6',{4},{6}),('train6_eval4',{6},{4}),('train46_eval8',{4,6},{8})]:
  tr=[i for i in range(len(ids)) if known[i] and int(rx[i]) in train_rx];ev=[i for i in range(len(ids)) if known[i] and int(rx[i]) in eval_rx]
  for use_gate,dest,missing in [(False,route,miss),(True,route_gate,miss_gate)]:
   groups=collections.defaultdict(list)
   for i in tr:groups[key_for(i,use_gate)].append(int(y[i]))
   mapping={}
   for k,truths in groups.items():
    choices=list(dict.fromkeys(k[:5]));counts=collections.Counter(truths);mapping[k]=max(choices,key=lambda c:counts[c])
   for i in ev:
    k=key_for(i,use_gate);missing[i]=k not in mapping;dest[i]=mapping.get(k,int(cand[i,1]))
   fold=res['folds'][name];field='gate_patterns' if use_gate else 'patterns'
   if len(mapping)!=fold[field] or len(tr)!=fold['train_L_s'] or len(ev)!=fold['eval_L_s']:raise ValueError('Fold metadata mismatch')
 def upper(indices,use_gate):
  groups=collections.defaultdict(list)
  for i in indices:groups[key_for(i,use_gate)].append(int(y[i]))
  return sum(max(collections.Counter(truths)[c] for c in set(k[:5])) for k,truths in groups.items())
 fields=0
 for group,receivers in [('RX4_6',{4,6}),('RX8',{8})]:
  for view in ['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']:
   ix=[i for i in range(len(ids)) if known[i] and int(rx[i]) in receivers and (scene[i]!='clean' if view=='all_impaired' else scene[i]==view)];good=lambda j:sum(cand[i,j]==y[i] for i in ix);want=dict(count=len(ix),raw_correct=good(0),physical_correct=good(1),unroll025_correct=good(2),unroll05_correct=good(3),unroll1_correct=good(4),crossfit_pattern_correct=sum(route[i]==y[i] for i in ix),crossfit_gate_correct=sum(route_gate[i]==y[i] for i in ix),unseen_pattern=sum(miss[i] for i in ix),unseen_gate_pattern=sum(miss_gate[i] for i in ix),label_oracle=sum(y[i] in cand[i] for i in ix),same_cell_pattern_upper=upper(ix,False),same_cell_gate_upper=upper(ix,True));cell=res['results'][group][view]
   for k,value in want.items():
    if cell[k]!=int(value):raise ValueError('Independent mismatch '+group+'/'+view+'/'+k)
    fields+=1
 out=dict(status='PASS',records=len(ids),L_s_scored=int(known.sum()),independent_fields=fields,prior_sha256=sha(path),U_s_truth_reads=0,V_reads=0,test_reads=0,method='sorted source physical-ID join, independent Counter maps and set-based recount')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
