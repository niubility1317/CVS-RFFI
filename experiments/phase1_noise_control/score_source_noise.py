"""Post-hoc fixed-bin source diagnostic of blind estimated relative noise."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

EDGES=np.array([-6.,-5.,-4.,-3.,-2.,-1.])
def noise_bins(x):
 x=np.asarray(x)
 if not np.isfinite(x).all():raise ValueError('Finite relative log variance required')
 return np.searchsorted(EDGES,x,side='right')
def pair_counts(pred,base,target):
 good=pred==target;b=base==target
 return int(good.sum()),int((good&~b).sum()),int((~good&b).sum())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable score required')
 m=json.loads((a.source/'prediction_complete.json').read_text());path=a.source/'source_noise.npz';prior=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01/source_predictions.npz';expected='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5';ck=a.root/'runs/20261102-phase1-blind-bayes-action-full-s2026092701-r01/checkpoint_epoch10.pt'
 if m.get('commit')!='6cab9dde4a0b9b66f4462c046a927893de1a8b44' or m.get('DURA_loaded') is not False or m.get('F_loaded') is not False or m.get('status')!='SOURCE_NOISE_FIXED' or m.get('count')!=11761 or m.get('sha256')!=sha(path) or m.get('prior_sha256')!=expected or sha(prior)!=expected or m.get('blind_checkpoint_sha256')!=sha(ck) or any(m.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')):raise ValueError('Exact fixed source noise estimates required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx=z['rx'];scene=z['scene'];nv=z['values']
 with np.load(prior,allow_pickle=False) as z:
  if not np.array_equal(ids,z['indices']) or not np.array_equal(rx,z['rx']) or not np.array_equal(scene,z['scene']):raise ValueError('Same physical source IDs and views required')
  v=z['values']
 if ids.shape!=(11761,) or len(set(ids.tolist()))!=11761 or nv.shape!=(11761,2) or v.shape!=(11761,15) or not np.isfinite(nv).all() or not np.isfinite(v).all():raise ValueError('Complete finite source arrays required')
 cache=a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False);known=roles[ids]=='L_s';unknown=roles[ids]=='U_s'
 if int(known.sum())!=1047 or not np.all(known|unknown):raise ValueError('Authorized source roles required')
 meta=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0';contract=json.loads((meta/'source_contract.json').read_text())
 with np.load(meta/'L_s_metadata.npz',allow_pickle=False) as z:keys=z['ids'];labels=z['y']
 if keys.tolist()!=contract['role_ids']['L_s'] or len(keys)!=6300 or np.any((labels<0)|(labels>=6)):raise ValueError('Exact L_s label contract required')
 mapping={str(k):int(y) for k,y in zip(keys,labels)}
 if len(mapping)!=6300 or not all(str(k) in mapping for k in physical[ids[known]]) or not all(str(k) not in mapping for k in physical[ids[unknown]]):raise ValueError('Source physical ID label join failed')
 truth=np.full(len(ids),-1,dtype=np.int64);truth[known]=[mapping[str(k)] for k in physical[ids[known]]];pseudo=v[:,0].astype(int);trusted=v[:,1]>=.95;bins=noise_bins(nv[:,1]);pred={'raw':v[:,5].astype(int),'physical':v[:,7].astype(int),'unroll1':v[:,13].astype(int)};results={}
 for group,receivers in [('RX4_6',(4,6)),('RX8',(8,))]:
  results[group]=[]
  for b in range(7):
   mask=np.isin(rx,receivers)&(scene!='clean')&(bins==b);cell={'count':int(mask.sum())};ps=mask&trusted;ls=mask&known;cell['pseudo_count']=int(ps.sum());cell['L_s_count']=int(ls.sum())
   for label,selected,target in [('pseudo',ps,pseudo),('L_s',ls,truth)]:
    cell[label+'_oracle_correct']=int(np.count_nonzero(selected&np.any(np.column_stack(tuple(pred.values()))==target[:,None],axis=1)))
    for name,values in pred.items():
     correct,repair,damage=pair_counts(values[selected],pred['physical'][selected],target[selected]);cell.setdefault(name,{})[label]={'correct':correct,'repairs_vs_physical':repair,'damages_vs_physical':damage}
   results[group].append(cell)
  if sum(x['count'] for x in results[group])!=int((np.isin(rx,receivers)&(scene!='clean')).sum()):raise ValueError('Complete fixed bins required')
 out=dict(status='SOURCE_NOISE_SCORE_COMPLETE',noise_sha256=sha(path),prior_sha256=expected,fixed_predictions_before_truth=True,bin_edges=EDGES.tolist(),L_s_scored=int(known.sum()),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,results=results,caveat='Post-hoc exploratory source diagnostics; no target truth; normalized blind model estimate is not a measured SNR. Oracle uses true or pseudo target and is not deployable.')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
