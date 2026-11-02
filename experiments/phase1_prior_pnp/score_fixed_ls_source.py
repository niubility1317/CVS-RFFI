"""Truth-last scoring of fixed Phase1 source predictions on legal L_s only."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core
from experiments.pamf_phase1_flow import feature_conditioned_experiment as labels_source

EXPECTED={
 'unroll':('20261102-phase1-unroll-source-replay-s2026092701-r01','134e34b6489ba3ca7f7a0da06449456558f99923'),
 'learned_action':('20261102-phase1-learned-action-validation-s2026092701-r01','5e5d081a107474d8d1d119564b09f98e2056be4c')}
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--candidate',choices=tuple(EXPECTED),required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable scoring output required')
 run,expected=EXPECTED[a.candidate];src=a.root/'runs'/run;path=src/'source_predictions.npz';m=core.read(src/'prediction_complete.json')
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!=expected or not m.get('source_only') or not m.get('DURA_comparator_only') or any(m.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')) or hashlib.sha256(path.read_bytes()).hexdigest()!=m.get('sha256'):raise ValueError('Exact frozen source predictions before truth required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];part=z['split'];rx_saved=z['rx'];scene_saved=z['scene'];values=z['values']
 if len(ids)!=11761 or len(set(ids.tolist()))!=len(ids) or values.shape!=(len(ids),15) or not np.isfinite(values).all():raise ValueError('Complete frozen predictions required')
 torch.set_num_threads(2);cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
 if not np.array_equal(ids,np.r_[validation,heldout]) or not np.array_equal(part,np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)]) or not np.array_equal(rx_saved,rx[ids]) or not np.array_equal(scene_saved,scene[ids]) or np.intersect1d(ids,training).size:raise ValueError('Fixed disjoint source cohort required')
 if not np.isin(values[:,[0,2,5,7,9,11,13]],np.arange(6)).all() or np.any((values[:,3]<0)|(values[:,3]>1)):raise ValueError('Valid fixed prediction and gate values required')
 fit=torch.load(a.root/'runs/20261102-phase1-observation-unroll-s2026092701-r01/checkpoint.pt',map_location='cpu',weights_only=False)['fit_ids']
 if np.intersect1d(ids,fit).size:raise ValueError('Source scoring records overlap restoration fit')
 known=roles[ids]=='L_s';u=roles[ids]=='U_s'
 if not np.all(known|u) or not np.any(known) or not np.any(u):raise ValueError('Source L_s and U_s roles required')
 # Label access starts only after fixed prediction, split and source-role checks above.
 _,contract=core.teacher(a.root,'cpu');truth_all=labels_source.source_labels(a.root,cache,clean,roles,contract);truth=truth_all[ids]
 if np.any(truth[known]<0) or np.any(truth[known]>=6) or np.any(truth[u]!=-1):raise ValueError('Only legal L_s truth may be materialized')
 active=values[:,3]>.8;pred={'clean':values[:,0],'raw':values[:,5],'dura':values[:,2],'parent':np.where(active,values[:,7],values[:,5]),'candidate':np.where(active,values[:,13],values[:,5])};results={}
 for group,pnum,receivers in (('RX4_6',1,(4,6)),('RX8',2,(8,))):
  results[group]={}
  for view in ('all_impaired',)+VIEWS:
   select=known&(part==pnum)&np.isin(rx[ids],receivers)&((scene[ids]!='clean') if view=='all_impaired' else (scene[ids]==view));count=int(select.sum())
   if not count:raise ValueError('All labeled source cells required')
   actual={k:(v==truth) for k,v in pred.items()};cell={'count':count,'active':int(np.count_nonzero(select&active)),'clean_f_pseudo_wrong':int(np.count_nonzero(select&~actual['clean']))}
   for key,correct in actual.items():
    cell[key]={'correct':int(np.count_nonzero(select&correct)),'accuracy_percent':float(np.mean(correct[select])*100),'repairs_vs_parent':int(np.count_nonzero(select&correct&~actual['parent'])),'damages_vs_parent':int(np.count_nonzero(select&~correct&actual['parent'])),'repairs_vs_dura':int(np.count_nonzero(select&correct&~actual['dura'])),'damages_vs_dura':int(np.count_nonzero(select&~correct&actual['dura']))}
   results[group][view]=cell
 a.out.mkdir(parents=True);out={'status':'SOURCE_LS_TRUTH_LAST_COMPLETE','candidate':a.candidate,'commit':a.commit,'prediction_run':run,'prediction_sha256':m['sha256'],'fixed_predictions_before_truth':True,'truth_role':'L_s only','L_s_count':int(known.sum()),'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'gate':.8,'scale':1.0,'caveat':'Post-hoc source L_s truth scoring; F was trained on source IDs and these source views were previously explored. No target adaptation or ground score.','results':results};core.write(a.out/'source_ls_truth_score.json',out);print('SOURCE_LS_TRUTH_SCORE_COMPLETE',json.dumps({'candidate':a.candidate,'count':int(known.sum()),'RX4_6':results['RX4_6']['all_impaired'],'RX8':results['RX8']['all_impaired']}),flush=True)
if __name__=='__main__':main()
