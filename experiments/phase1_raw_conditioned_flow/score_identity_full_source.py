"""Score fixed full source predictions; access only authorized L_s truth afterward."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core,labels_source
VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('Immutable source score required')
 m=core.read(a.source/'prediction_complete.json');path=a.source/'source_predictions.npz';sha=hashlib.sha256(path.read_bytes()).hexdigest();checkpoint=a.root/'runs/20261102-phase1-identity-flow-source-s2026092701-r01/checkpoint.pt'
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!='d8f98b6df31495905081a71db97f9d00557ac209' or m.get('count')!=11761 or m.get('original_records_matched')!=672 or m.get('sha256')!=sha or m.get('prior_sha256')!='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5' or m.get('model_sha256')!=hashlib.sha256(checkpoint.read_bytes()).hexdigest() or any(m.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')):raise ValueError('Exact frozen source predictions and model required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx_saved=z['rx'];scene_saved=z['scene'];prior=z['prior_values'];v=z['flow_values']
 if ids.shape!=(11761,) or len(set(ids.tolist()))!=11761 or prior.shape!=(11761,15) or v.shape!=(11761,2) or not np.isfinite(prior).all() or not np.isfinite(v).all() or not np.isin(v[:,0],np.arange(6)).all() or np.any(v[:,1]<0):raise ValueError('Valid complete source arrays required')
 torch.set_num_threads(2);cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);_,scene=s.epoch_data(cache,19);state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if state.get('commit')!='4fa488be513e5a116ee1cb88411af2a82e17ccda' or len(state['fit_ids'])!=4096 or np.intersect1d(ids,state['fit_ids']).size or not np.array_equal(ids,np.r_[validation,heldout]) or not np.array_equal(rx_saved,rx[ids]) or not np.array_equal(scene_saved,scene[ids]):raise ValueError('Exact disjoint source cohort required')
 roles=np.load(cache/'train/roles.npy',allow_pickle=False);known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if int(known.sum())!=1047 or not np.all(known|unlabeled):raise ValueError('Exact source roles required')
 # True source identities are accessed only after all predictions are fixed.
 _,contract=core.teacher(a.root,'cpu');truth=labels_source.source_labels(a.root,cache,clean,roles,contract)[ids]
 if np.any(truth[known]<0) or np.any(truth[unlabeled]!=-1):raise ValueError('Only L_s identity truth allowed')
 pseudo=prior[:,0];trusted=prior[:,1]>=.95;pred={'raw':prior[:,5],'physical':prior[:,7],'flow':v[:,0],'DURA':prior[:,2]};iq={'raw':prior[:,6],'physical':prior[:,8],'flow':v[:,1]};results={}
 for group,receivers in [('RX4_6',(4,6)),('RX8',(8,))]:
  results[group]={}
  for view in VIEWS:
   mask=np.isin(rx_saved,receivers)&((scene_saved!='clean') if view=='all_impaired' else (scene_saved==view));ps=mask&trusted;ls=mask&known;cell={'count':int(mask.sum()),'pseudo_count':int(ps.sum()),'L_s_count':int(ls.sum()),'physical_gate_active_count':int(np.count_nonzero(mask&(prior[:,3]>.8)))}
   for name in pred:
    item={}
    if name in iq:item['iq_error_mean']=float(iq[name][mask].mean())
    for key,select,target in [('pseudo',ps,pseudo),('L_s',ls,truth)]:
     good=pred[name]==target;base=pred['physical']==target;item[key+'_correct']=int(np.count_nonzero(select&good));item[key+'_repairs_vs_physical']=int(np.count_nonzero(select&good&~base));item[key+'_damages_vs_physical']=int(np.count_nonzero(select&~good&base))
    cell[name]=item
   results[group][view]=cell
 out=dict(status='SOURCE_SCORE_COMPLETE',prediction_sha256=sha,fixed_predictions_before_truth=True,L_s_scored=int(known.sum()),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,results=results,caveat='Post-hoc source diagnostics; source F pseudo-targets are not identity truth, and IDs previously explored.')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':out['status'],'RX4_6':results['RX4_6']['all_impaired'],'RX8':results['RX8']['all_impaired']}),flush=True)
if __name__=='__main__':main()
