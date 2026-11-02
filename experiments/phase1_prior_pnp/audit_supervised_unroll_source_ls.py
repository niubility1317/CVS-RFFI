"""Audit source L_s truth scores using an independent physical-ID join."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--score',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Immutable independent audit required')
 m=json.loads((a.source/'prediction_complete.json').read_text());s=json.loads(a.score.read_text());path=a.source/'source_predictions.npz';sha=hashlib.sha256(path.read_bytes()).hexdigest()
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!='60e5f74f39bcc7539d78f4ea8470065915e66a18' or m.get('count')!=11761 or m.get('sha256')!=sha or s.get('prediction_sha256')!=sha or s.get('status')!='SOURCE_LS_TRUTH_LAST_COMPLETE' or not s.get('fixed_predictions_before_truth') or any(s.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed prediction provenance required')
 with np.load(path,allow_pickle=False) as data:ids=data['indices'];rx=data['rx'];scene=data['scene'];v=data['prior_values'];new=data['candidate_values']
 if ids.shape!=(11761,) or len(set(ids.tolist()))!=11761 or v.shape!=(11761,15) or new.shape!=(11761,4):raise ValueError('Complete unique source cohort required')
 cache=a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache';physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False);meta=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:physical_ls=z['ids'];truth_ls=z['y']
 if physical_ls.tolist()!=contract['role_ids']['L_s'] or len(physical_ls)!=6300 or not np.all((truth_ls>=0)&(truth_ls<6)):raise ValueError('Invalid authorized source labels')
 truth_map={str(k):int(y) for k,y in zip(physical_ls,truth_ls)};known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if len(truth_map)!=6300 or int(known.sum())!=1047 or not np.all(known|unlabeled) or not all(str(k) in truth_map for k in physical[ids[known]]) or not all(str(k) not in truth_map for k in physical[ids[unlabeled]]):raise ValueError('Physical-ID role join failed')
 truth={i:truth_map[str(physical[ids[i]])] for i in np.flatnonzero(known)}
 def full(i):return int(v[i,13] if v[i,3]>.8 else v[i,5])
 def prediction(name,i):return full(i) if name=='full' else int(v[i,2]) if name=='DURA' else int(new[i,0 if name=='label' else 2])
 checks=0;fields=0
 for group,receivers in [('RX4_6',{4,6}),('RX8',{8})]:
  for view,cell in s['results'][group].items():
   ix=[i for i in truth if int(rx[i]) in receivers and (scene[i]!='clean' if view=='all_impaired' else scene[i]==view)];assert cell['count']==len(ix) and cell['active']==sum(v[i,3]>.8 for i in ix);fields+=2;baseline={i for i in ix if full(i)==truth[i]}
   for name in ['full','label','multiview','DURA']:
    good={i for i in ix if prediction(name,i)==truth[i]};c=cell[name];assert c['correct']==len(good) and c['repairs_vs_full']==len(good-baseline) and c['damages_vs_full']==len(baseline-good);fields+=3
   checks+=1
 if checks!=16:raise ValueError('Expected 16 subgroup cells')
 result=dict(status='PASS',source_only=True,L_s_scored=len(truth),groups_and_scenes=checks,fields_recomputed=fields,prediction_sha256=sha,method='independent physical-ID join and set-based paired counts',U_s_truth_reads=0,V_reads=0,test_reads=0)
 with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
