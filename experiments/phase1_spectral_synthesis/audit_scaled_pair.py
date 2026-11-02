"""Audit source pairing using scalar routing and a separate physical-ID truth join."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Immutable audit required')
 runs=a.root/'runs';small=runs/'20261102-phase1-spectral-synthesis-pilot-s2026092701-r01';large=runs/'20261102-phase1-spectral-synthesis-scaled-s2026092701-r01';prior=runs/'20261102-phase1-unroll-source-replay-s2026092701-r01';m=json.loads((a.source/'prediction_complete.json').read_text());score=json.loads((a.source/'source_score.json').read_text());sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();path=a.source/'source_predictions.npz';h=sha(path)
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!=a.commit or m.get('count')!=672 or m.get('sha256')!=h or m.get('small_sha256')!=sha(small/'source_predictions.npz') or m.get('scaled_sha256')!=sha(large/'source_predictions.npz') or m.get('prior_sha256')!=sha(prior/'source_predictions.npz') or m.get('model_sha256')!=[sha(small/'checkpoint.pt'),sha(large/'checkpoint.pt')] or score.get('status')!='SOURCE_PAIRED_SCORE_COMPLETE' or score.get('prediction_sha256')!=h or not score.get('fixed_predictions_before_truth') or any(score.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed source comparison provenance required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];split=z['split'];scene=z['scene'];prob=z['gate_probability'];v=z['values']
 with np.load(small/'source_predictions.npz',allow_pickle=False) as s,np.load(large/'source_predictions.npz',allow_pickle=False) as b,np.load(prior/'source_predictions.npz',allow_pickle=False) as prev:
  if not np.array_equal(ids,s['indices']) or not np.array_equal(ids,b['indices']) or not np.array_equal(scene,s['scene']) or not np.array_equal(scene,b['scene']):raise ValueError('Source prediction IDs must align')
  order=np.argsort(prev['indices']);sorted_ids=prev['indices'][order];pos=np.searchsorted(sorted_ids,ids)
  if not np.array_equal(sorted_ids[pos],ids) or not np.array_equal(prob,prev['values'][order[pos],3]):raise ValueError('Exact fixed prior gate required')
  x=s['values'];y=b['values'];routed=0
  for i in range(len(ids)):
   active=bool(prob[i]>.8)
   for column,expected in [(3,x[i,6] if active else x[i,2]),(4,y[i,6] if active else y[i,2]),(5,x[i,7] if active else x[i,3]),(6,y[i,7] if active else y[i,3])]:assert v[i,column]==expected;routed+=1
   assert v[i,0]==x[i,0] and v[i,1]==x[i,1] and v[i,2]==x[i,2]
 if ids.shape!=(672,) or v.shape!=(672,7) or len(set(ids.tolist()))!=672 or int((prob>.8).sum())!=m['gate_active_count']:raise ValueError('Complete source comparison required')
 cache=runs/'20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache';physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False);meta=runs/'20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:physical_ls=z['ids'];truth_ls=z['y']
 if physical_ls.tolist()!=contract['role_ids']['L_s'] or len(physical_ls)!=6300 or not np.all((truth_ls>=0)&(truth_ls<6)):raise ValueError('Invalid authorized L_s labels')
 lookup={str(k):int(y) for k,y in zip(physical_ls,truth_ls)};known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if len(lookup)!=6300 or not np.all(known|unlabeled) or not all(str(x) in lookup for x in physical[ids[known]]) or not all(str(x) not in lookup for x in physical[ids[unlabeled]]):raise ValueError('Source physical-ID join failed')
 truth={i:lookup[str(physical[ids[i]])] for i in np.flatnonzero(known)};cells=0;fields=0
 if score['L_s_scored']!=len(truth):raise ValueError('Source true label count mismatch')
 for group in ('RX4_6','RX8'):
  for view,cell in score['results'][group].items():
   ix=[i for i in range(len(ids)) if split[i]==group and (scene[i]!='clean' if view=='all_impaired' else scene[i]==view)];ps=[i for i in ix if v[i,0]>=.95];ls=[i for i in ix if i in truth];assert cell['count']==len(ix) and cell['pseudo_count']==len(ps) and cell['L_s_count']==len(ls) and cell['active_count']==sum(prob[i]>.8 for i in ix);fields+=4
   for name,col,iqcol in [('small',3,5),('scaled',4,6)]:
    c=cell[name];assert math.isclose(c['iq_error_mean'],sum(float(v[i,iqcol]) for i in ix)/len(ix),rel_tol=1e-6,abs_tol=1e-8);fields+=1
    for key,subset in [('pseudo',ps),('L_s',ls)]:
     target=lambda i:int(v[i,1]) if key=='pseudo' else truth[i];good={i for i in subset if int(v[i,col])==target(i)};base={i for i in subset if int(v[i,3])==target(i)};assert c[key+'_correct']==len(good) and c[key+'_repairs_vs_small']==len(good-base) and c[key+'_damages_vs_small']==len(base-good);fields+=3
   cells+=1
 if cells!=16:raise ValueError('Expected 16 source cells')
 out=dict(status='PASS',source_only=True,records=len(ids),L_s_scored=len(truth),cells=cells,fields_recomputed=fields,routed_fields_recomputed=routed,prediction_sha256=h,method='independent physical-ID join, scalar gate replay and set counts',U_s_truth_reads=0,V_reads=0,test_reads=0)
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
