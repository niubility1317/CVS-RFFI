"""Score fixed posterior sample decisions using only authorized source labels."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source score required')
 m=json.loads((a.source/'prediction_complete.json').read_text());path=a.source/'source_predictions.npz';h=sha(path);prior=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01/source_predictions.npz';expected='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5';ck=a.root/'runs/20261102-phase1-blind-bayes-action-full-s2026092701-r01/checkpoint_epoch10.pt';src=a.root/'runs/20261102-phase1-blind-bayes-source-validation-s2026092701-r01/source_prior.pt'
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!='f6072ab0935cd6d4a40cb4e50a230f8e463f466b' or m.get('count')!=672 or m.get('sha256')!=h or m.get('prior_sha256')!=expected or sha(prior)!=expected or m.get('blind_checkpoint_sha256')!=sha(ck) or m.get('source_prior_sha256')!=sha(src) or m.get('DURA_loaded') is not False or any(m.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')):raise ValueError('Fixed source sample predictions required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx=z['rx'];scene=z['scene'];split=z['split'];old=z['prior_values'];pred=z['predictions'];err=z['aligned_errors'];prob=z['probabilities'];choices=z['choices'];waves=z['selected_waveforms'];active=z['active']
 if ids.shape!=(672,) or len(set(ids.tolist()))!=672 or old.shape!=(672,15) or pred.shape!=(672,4) or err.shape!=(672,4) or prob.shape!=(672,6,6) or choices.shape!=(672,2) or waves.shape!=(672,2,2,256) or active.shape!=(672,) or not all(np.isfinite(x).all() for x in (old,pred,err,prob,waves)) or not np.isin(pred,np.arange(6)).all() or np.any(err<0):raise ValueError('Complete finite source pilot required')
 if not np.array_equal(pred[:,0],old[:,5]) or not np.array_equal(pred[:,1],old[:,7]) or not np.array_equal(active,old[:,3]>.8):raise ValueError('Fixed source raw, physical and gate required')
 cache=a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False);known=roles[ids]=='L_s';unknown=roles[ids]=='U_s'
 if not np.all(known|unknown):raise ValueError('Authorized source roles required')
 meta=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0';contract=json.loads((meta/'source_contract.json').read_text())
 with np.load(meta/'L_s_metadata.npz',allow_pickle=False) as z:keys=z['ids'];labels=z['y']
 if keys.tolist()!=contract['role_ids']['L_s'] or len(keys)!=6300 or np.any((labels<0)|(labels>=6)):raise ValueError('Exact source L_s labels required')
 lookup={str(k):int(y) for k,y in zip(keys,labels)}
 if len(lookup)!=6300 or not all(str(k) in lookup for k in physical[ids[known]]) or not all(str(k) not in lookup for k in physical[ids[unknown]]):raise ValueError('Source physical identity binding failed')
 truth=np.full(len(ids),-1,dtype=np.int64);truth[known]=[lookup[str(k)] for k in physical[ids[known]]];pseudo=old[:,0].astype(int);trusted=old[:,1]>=.95;sample_pred=prob.argmax(-1);results={};names=['raw','physical','consensus','max_confidence']
 for group in ('RX4_6','RX8'):
  results[group]={}
  for view in VIEWS:
   mask=(split==group)&((scene!='clean') if view=='all_impaired' else (scene==view));ps=mask&trusted;ls=mask&known;cell={'count':int(mask.sum()),'pseudo_count':int(ps.sum()),'L_s_count':int(ls.sum()),'active_count':int(np.count_nonzero(mask&active))}
   for label,selected,target in [('pseudo',ps,pseudo),('L_s',ls,truth)]:
    cell[label+'_oracle_correct']=int(np.count_nonzero(selected&((pred[:,1]==target)|np.any(sample_pred==target[:,None],axis=1))))
    for j,name in enumerate(names):
     good=pred[:,j]==target;base=pred[:,1]==target;item=cell.setdefault(name,{});item[label+'_correct']=int(np.count_nonzero(selected&good));item[label+'_repairs_vs_physical']=int(np.count_nonzero(selected&good&~base));item[label+'_damages_vs_physical']=int(np.count_nonzero(selected&~good&base))
   for j,name in enumerate(names):cell[name]['iq_error_mean']=float(err[mask,j].mean())
   results[group][view]=cell
 out=dict(status='SOURCE_SAMPLE_SCORE_COMPLETE',prediction_sha256=h,fixed_predictions_before_truth=True,L_s_scored=int(known.sum()),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,results=results,caveat='Previously explored source IDs; pseudo targets are not identity truth; target-aided oracle cannot be deployed; Gaussian posterior conditioned on estimated channel is an approximation.')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':out['status'],'L_s_scored':out['L_s_scored'],'RX4_6':results['RX4_6']['all_impaired'],'RX8':results['RX8']['all_impaired']}),flush=True)
if __name__=='__main__':main()
