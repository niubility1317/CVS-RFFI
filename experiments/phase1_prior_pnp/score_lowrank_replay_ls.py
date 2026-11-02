"""Score a fixed source replay on authorized L_s truth only."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core
from experiments.pamf_phase1_flow import feature_conditioned_experiment as labels_source

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source truth score required')
 source=a.root/'runs/20261102-phase1-lowrank-source-replay-s2026092701-r03';m=core.read(source/'prediction_complete.json');path=source/'source_predictions.npz'
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!='3d97d871098e440331371bc8fb000ff1aeebf746' or m.get('count')!=11761 or any(m.get(x)!=0 for x in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')) or hashlib.sha256(path.read_bytes()).hexdigest()!=m.get('sha256'):raise ValueError('Exact fixed source predictions required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx_saved=z['rx'];scene_saved=z['scene'];prior=z['prior_values'];low=z['lowrank_values']
 torch.set_num_threads(2);cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);_,scene=s.epoch_data(cache,19);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
 if not np.array_equal(ids,np.r_[validation,heldout]) or not np.array_equal(rx_saved,rx[ids]) or not np.array_equal(scene_saved,scene[ids]) or np.intersect1d(ids,training).size or prior.shape!=(11761,15) or low.shape!=(11761,4):raise ValueError('Disjoint and aligned source cohort required')
 fit=torch.load(a.root/'runs/20261102-phase1-observation-unroll-s2026092701-r01/checkpoint.pt',map_location='cpu',weights_only=False)['fit_ids']
 if np.intersect1d(ids,fit).size:raise ValueError('Restoration fit IDs must be disjoint')
 known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if int(known.sum())!=1047 or not np.all(known|unlabeled):raise ValueError('Exact source roles required')
 # Read authorized L_s labels only after all predictions and physical IDs are fixed.
 _,contract=core.teacher(a.root,'cpu');truth_all=labels_source.source_labels(a.root,cache,clean,roles,contract);truth=truth_all[ids]
 if np.any(truth[known]<0) or np.any(truth[unlabeled]!=-1):raise ValueError('L_s truth only required')
 active=prior[:,3]>.8;raw=prior[:,5];pred={'full':np.where(active,prior[:,13],raw),'rank128':low[:,0],'rank256':low[:,2],'DURA':prior[:,2]};results={}
 for group,receivers in [('RX4_6',(4,6)),('RX8',(8,))]:
  results[group]={}
  for view in ['all_impaired','clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']:
   select=known&np.isin(rx[ids],receivers)&((scene[ids]!='clean') if view=='all_impaired' else (scene[ids]==view));n=int(select.sum())
   if not n:raise ValueError('All L_s cells required')
   cell={'count':n,'active':int(np.count_nonzero(select&active))};base=pred['full']==truth
   for name,v in pred.items():
    ok=v==truth;cell[name]={'correct':int(np.count_nonzero(select&ok)),'repairs_vs_full':int(np.count_nonzero(select&ok&~base)),'damages_vs_full':int(np.count_nonzero(select&~ok&base))}
   results[group][view]=cell
 a.out.mkdir(parents=True);out=dict(status='SOURCE_LS_TRUTH_LAST_COMPLETE',commit=a.commit,prediction_sha256=m['sha256'],fixed_predictions_before_truth=True,L_s_scored=int(known.sum()),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,results=results,caveat='Post-hoc source L_s identity truth on previously explored IDs; not independent ground performance.');core.write(a.out/'source_ls_truth_score.json',out);print(json.dumps({'status':out['status'],'RX4_6':results['RX4_6']['all_impaired'],'RX8':results['RX8']['all_impaired']}),flush=True)
if __name__=='__main__':main()
