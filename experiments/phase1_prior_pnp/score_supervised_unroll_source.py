"""Score fixed source supervised-unroll replay against frozen F pseudo-target."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Immutable source score required')
 m=json.loads((a.input/'prediction_complete.json').read_text());path=a.input/'source_predictions.npz'
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!='60e5f74f39bcc7539d78f4ea8470065915e66a18' or m.get('candidate_names')!=['label','multiview'] or m.get('count')!=11761 or any(m.get(x)!=0 for x in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')) or hashlib.sha256(path.read_bytes()).hexdigest()!=m.get('sha256'):raise ValueError('Complete fixed source predictions required')
 with np.load(path,allow_pickle=False) as x:ids=x['indices'];rx=x['rx'];scene=x['scene'];prior=x['prior_values'];low=x['candidate_values']
 if ids.shape!=(11761,) or len(set(ids.tolist()))!=len(ids) or prior.shape!=(len(ids),15) or low.shape!=(len(ids),4) or not np.isfinite(prior).all() or not np.isfinite(low).all():raise ValueError('Complete finite source arrays required')
 active=prior[:,3]>.8;target=prior[:,0].astype(np.int64);valid=prior[:,1]>=.95;raw=prior[:,5].astype(np.int64);pred={'full':np.where(active,prior[:,13],raw).astype(np.int64),'label':low[:,0].astype(np.int64),'multiview':low[:,2].astype(np.int64),'DURA':prior[:,2].astype(np.int64)};iq={'full':np.where(active,prior[:,14],prior[:,6]),'label':low[:,1],'multiview':low[:,3]};views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired'];results={}
 for group,receivers in [('RX4_6',(4,6)),('RX8',(8,))]:
  results[group]={}
  for view in views:
   mask=np.isin(rx,receivers)&((scene!='clean') if view=='all_impaired' else (scene==view));trusted=mask&valid;count=int(mask.sum());n=int(trusted.sum())
   if not count or not n:raise ValueError('Every source group and scene required')
   reference=pred['full']==target;cell={'count':count,'trusted_count':n,'active_count':int(np.count_nonzero(mask&active))}
   for name in pred:
    correct=pred[name]==target;cell[name]={'pseudo_correct':int(np.count_nonzero(trusted&correct)),'repairs_vs_full':int(np.count_nonzero(trusted&correct&~reference)),'damages_vs_full':int(np.count_nonzero(trusted&~correct&reference))}
    if name in iq:cell[name]['aligned_iq_error_mean']=float(iq[name][mask].mean())
   results[group][view]=cell
 out=dict(status='SOURCE_PSEUDO_SCORE_COMPLETE',prediction_sha256=m['sha256'],records=len(ids),source_only=True,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,results=results,caveat='Post-hoc source pseudo-label agreement; clean F predictions are not identity truth; previously explored source records.');a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':out['status'],'RX4_6':results['RX4_6']['all_impaired'],'RX8':results['RX8']['all_impaired']}),flush=True)
if __name__=='__main__':main()
