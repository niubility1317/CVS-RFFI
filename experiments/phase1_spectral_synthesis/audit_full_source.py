"""Independent source physical-ID join and set-based full-cohort score audit."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--score',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('Immutable audit required')
 m=json.loads((a.source/'prediction_complete.json').read_text());sc=json.loads(a.score.read_text());path=a.source/'source_predictions.npz';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();h=sha(path);old=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01/source_predictions.npz';checkpoint=a.root/'runs/20261102-phase1-spectral-synthesis-pilot-s2026092701-r01/checkpoint.pt'
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!='7219afff23fc38d9c4bf7304118b27b4dce0f1f2' or m.get('count')!=11761 or m.get('sha256')!=h or m.get('prior_sha256')!=sha(old) or m.get('model_sha256')!=sha(checkpoint) or sc.get('prediction_sha256')!=h or sc.get('status')!='SOURCE_SCORE_COMPLETE' or not sc.get('fixed_predictions_before_truth') or any(sc.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact complete source predictions required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx=z['rx'];scene=z['scene'];prior=z['prior_values'];v=z['spectral_values']
 with np.load(old,allow_pickle=False) as z:
  if not np.array_equal(ids,z['indices']) or not np.array_equal(rx,z['rx']) or not np.array_equal(scene,z['scene']) or not np.array_equal(prior,z['values']):raise ValueError('Prior source arrays differ')
 if ids.shape!=(11761,) or len(set(ids.tolist()))!=11761 or prior.shape!=(11761,15) or v.shape!=(11761,2):raise ValueError('Complete unique source cohort required')
 inactive=prior[:,3]<=.8
 if not np.array_equal(v[inactive,0],prior[inactive,5]) or not np.allclose(v[inactive,1],prior[inactive,6],rtol=1e-6,atol=1e-7):raise ValueError('Inactive gate must preserve raw waveform predictions')
 cache=a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache';physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False);meta=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:physical_ls=z['ids'];truth_ls=z['y']
 if physical_ls.tolist()!=contract['role_ids']['L_s'] or len(physical_ls)!=6300 or not np.all((truth_ls>=0)&(truth_ls<6)):raise ValueError('Invalid authorized source labels')
 lookup={str(k):int(y) for k,y in zip(physical_ls,truth_ls)};known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if len(lookup)!=6300 or int(known.sum())!=1047 or not np.all(known|unlabeled) or not all(str(x) in lookup for x in physical[ids[known]]) or not all(str(x) not in lookup for x in physical[ids[unlabeled]]):raise ValueError('Source physical-ID label join failed')
 truth={i:lookup[str(physical[ids[i]])] for i in np.flatnonzero(known)};cells=0;fields=0
 for group,receivers in [('RX4_6',{4,6}),('RX8',{8})]:
  for view,cell in sc['results'][group].items():
   ix=[i for i in range(len(ids)) if int(rx[i]) in receivers and (scene[i]!='clean' if view=='all_impaired' else scene[i]==view)];pseudo=[i for i in ix if prior[i,1]>=.95];ls=[i for i in ix if i in truth]
   assert cell['count']==len(ix) and cell['pseudo_count']==len(pseudo) and cell['L_s_count']==len(ls) and cell['active_count']==sum(prior[i,3]>.8 for i in ix);fields+=4
   def decision(name,i):return int(prior[i,5] if name=='raw' else prior[i,7] if name=='physical' else prior[i,2] if name=='DURA' else v[i,0])
   for name in ('raw','physical','synthesis','DURA'):
    item=cell[name]
    if name!='DURA':
     error=sum(float(prior[i,6] if name=='raw' else prior[i,8] if name=='physical' else v[i,1]) for i in ix)/len(ix);assert math.isclose(item['iq_error_mean'],error,rel_tol=1e-6,abs_tol=1e-8);fields+=1
    for key,subset in [('pseudo',pseudo),('L_s',ls)]:
     target=lambda i:int(prior[i,0]) if key=='pseudo' else truth[i];good={i for i in subset if decision(name,i)==target(i)};base={i for i in subset if decision('physical',i)==target(i)};assert item[key+'_correct']==len(good) and item[key+'_repairs_vs_physical']==len(good-base) and item[key+'_damages_vs_physical']==len(base-good);fields+=3
   cells+=1
 if cells!=16:raise ValueError('Expected 16 source cells')
 out=dict(status='PASS',source_only=True,records=len(ids),L_s_scored=len(truth),cells=cells,fields_recomputed=fields,prediction_sha256=h,method='independent source physical-ID join and set-based counts',U_s_truth_reads=0,V_reads=0,test_reads=0)
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
