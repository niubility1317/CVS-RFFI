"""Score fixed spectral-synthesis source predictions, then access only L_s truth."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core,labels_source

VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Immutable score required')
 m=core.read(a.source/'prediction_complete.json');path=a.source/'source_predictions.npz';sha=hashlib.sha256(path.read_bytes()).hexdigest()
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!='059987ed75b568b4ce052d7638ee0262bd5652c3' or m.get('count')!=672 or m.get('prediction_sha256')!=sha or m.get('DURA_loaded') is not False or m.get('F_frozen') is not True or any(m.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads','model_updates_after_prediction')):raise ValueError('Complete fixed source predictions required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];split=z['split'];rx_saved=z['rx'];scene_saved=z['scene'];values=z['values']
 if ids.shape!=(672,) or len(set(ids.tolist()))!=672 or values.shape!=(672,8) or not np.isfinite(values).all() or not np.isin(values[:,[1,2,4,6]],np.arange(6)).all() or np.any(values[:,[3,5,7]]<0):raise ValueError('Valid fixed prediction arrays required')
 torch.set_num_threads(2);cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);_,scene=s.epoch_data(cache,19);rng=np.random.default_rng(2026135057);expected=[];expected_split=[]
 for group,rows in [('RX4_6',validation),('RX8',heldout)]:
  for view in VIEWS:
   chosen=np.sort(rng.choice(rows[scene[rows]==view],size=48,replace=False));expected.extend(chosen.tolist());expected_split.extend([group]*48)
 state=torch.load(a.source/'checkpoint.pt',map_location='cpu',weights_only=False)
 if state.get('commit')!=m['commit'] or len(state['fit_ids'])!=1024 or np.intersect1d(ids,state['fit_ids']).size or not np.array_equal(ids,expected) or not np.array_equal(split,expected_split) or not np.array_equal(rx_saved,rx[ids]) or not np.array_equal(scene_saved,scene[ids]):raise ValueError('Exact disjoint fixed source cohort required')
 roles=np.load(cache/'train/roles.npy',allow_pickle=False);known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if not np.all(known|unlabeled):raise ValueError('Only L_s and U_s source roles allowed')
 # Identity truth is read only after source predictions, cohort and checkpoint are fixed.
 _,contract=core.teacher(a.root,'cpu');truth=labels_source.source_labels(a.root,cache,clean,roles,contract)[ids]
 if np.any(truth[known]<0) or np.any(truth[unlabeled]!=-1):raise ValueError('Only authorized L_s identity truth allowed')
 target=values[:,1];trusted=values[:,0]>=.95;pred={'raw':values[:,2],'physical':values[:,4],'synthesis':values[:,6]};iq={'raw':values[:,3],'physical':values[:,5],'synthesis':values[:,7]};results={}
 for group in ('RX4_6','RX8'):
  results[group]={}
  for view in VIEWS+['all_impaired']:
   mask=(split==group)&((scene_saved!='clean') if view=='all_impaired' else (scene_saved==view));pseudo=mask&trusted;ls=mask&known;cell={'count':int(mask.sum()),'pseudo_count':int(pseudo.sum()),'L_s_count':int(ls.sum())}
   for name in pred:
    cell[name]={'iq_error_mean':float(iq[name][mask].mean())}
    for key,selected,correct_truth in [('pseudo',pseudo,target),('L_s',ls,truth)]:
     good=pred[name]==correct_truth;baseline=pred['physical']==correct_truth;cell[name][key+'_correct']=int(np.count_nonzero(selected&good));cell[name][key+'_repairs_vs_physical']=int(np.count_nonzero(selected&good&~baseline));cell[name][key+'_damages_vs_physical']=int(np.count_nonzero(selected&~good&baseline))
   results[group][view]=cell
 out=dict(status='SOURCE_SCORE_COMPLETE',prediction_sha256=sha,fixed_predictions_before_truth=True,L_s_scored=int(known.sum()),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,results=results,caveat='Source clean F pseudo targets are not identity truth. Evaluated source records have been previously explored and are not ground test or a blind test.')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':out['status'],'L_s_scored':out['L_s_scored'],'RX4_6':results['RX4_6']['all_impaired'],'RX8':results['RX8']['all_impaired']}),flush=True)
if __name__=='__main__':main()
