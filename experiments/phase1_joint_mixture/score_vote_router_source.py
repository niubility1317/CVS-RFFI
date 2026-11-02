"""Score fixed cross-fitted vote-router source decisions on legitimate L_s."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source score required')
 src=a.root/'runs/20261102-phase1-vote-router-source-s2026092701-r01';m=json.loads((src/'result.json').read_text());path=src/'source_predictions.npz'
 if m.get('status')!='SOURCE_ROUTER_TRAINING_COMPLETE' or m.get('commit')!='4f065676bc23c5b1adaa9e7eab2c7dfd299e849f' or m.get('count')!=1047 or m.get('U_s_truth_reads')!=0 or m.get('test_reads')!=0 or hashlib.sha256(path.read_bytes()).hexdigest()!=m.get('source_output_sha256'):raise ValueError('Fixed source router predictions required')
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 idx=d['indices'];pred=d['candidate_predictions'];choice=d['choice'];selected=d['prediction'];cache=a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';roles=np.load(cache/'roles.npy',allow_pickle=False);physical=np.load(cache/'physical_ids.npy',allow_pickle=False)
 if idx.shape!=(1047,) or len(set(idx.tolist()))!=1047 or not np.all(roles[idx]=='L_s') or not np.array_equal(physical[idx],d['physical_ids']) or pred.shape!=(1047,9) or np.any((choice<0)|(choice>=9)) or not np.array_equal(pred[np.arange(len(idx)),choice],selected):raise ValueError('Exact source L_s binding and route required')
 # Source router predictions were fixed before evaluation labels are used.
 f0=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0';contract=json.loads((f0/'source_contract.json').read_text())
 with np.load(f0/'L_s_metadata.npz',allow_pickle=False) as z:ids=z['ids'];labels=z['y']
 if list(ids)!=contract['role_ids']['L_s'] or labels.shape!=(6300,) or len(set(ids.tolist()))!=6300:raise ValueError('Exact authorized L_s labels required')
 table={str(k):int(v) for k,v in zip(ids,labels)};truth=np.array([table[str(x)] for x in d['physical_ids']]);results={}
 for group,rxs in [('RX4_6_crossfit',(4,6)),('RX8_holdout',(8,))]:
  results[group]={}
  for view in VIEWS:
   mask=np.isin(d['rx'],rxs)&((d['scene']!='clean') if view=='all_impaired' else d['scene']==view);count=int(mask.sum());
   if count==0:raise ValueError('Each source group and view required')
   free=pred[:,1]==truth;router=selected==truth;raw=pred[:,0]==truth;results[group][view]={'count':count,'raw_correct':int(np.count_nonzero(mask&raw)),'free_correct':int(np.count_nonzero(mask&free)),'router_correct':int(np.count_nonzero(mask&router)),'repairs_vs_free':int(np.count_nonzero(mask&router&~free)),'damages_vs_free':int(np.count_nonzero(mask&~router&free)),'selected_nonfree':int(np.count_nonzero(mask&(choice!=1)))}
 a.out.mkdir(parents=True);out=dict(status='SOURCE_ROUTER_TRUTH_SCORE_COMPLETE',commit=a.commit,source_prediction_sha256=m['source_output_sha256'],source_L_s_count=len(idx),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,results=results,caveat='RX4/6 are cross-fitted receiver folds and RX8 is held out from router fitting; base models saw source IDs, source views explored, ground candidate set chosen after ground results.')
 with (a.out/'source_truth_score.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print('SOURCE_ROUTER_TRUTH_SCORE_COMPLETE',json.dumps({k:v['all_impaired'] for k,v in results.items()}),flush=True)
if __name__=='__main__':main()
