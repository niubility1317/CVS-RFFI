"""Score fixed cross-fitted source router on pseudo and authorized L_s labels."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
NAMES=['raw','physical','unroll025','unroll05','unroll1'];VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source router score required')
 m=json.loads((a.source/'result.json').read_text());path=a.source/'source_predictions.npz';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();h=sha(path)
 if m.get('status')!='SOURCE_GATE_ROUTER_TRAINING_COMPLETE' or m.get('commit')!='9dac8eaa0a35e352652791bf0080ade59751af09' or m.get('count')!=11761 or m.get('candidate_names')!=NAMES or m.get('source_output_sha256')!=h or m.get('DURA_in_candidate') is not False or any(m.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact fixed source router predictions required')
 for fold,digest in m['fold_model_sha256'].items():
  if sha(a.source/(fold+'.pt'))!=digest:raise ValueError('Fixed receiver-fold model hash required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];physical_saved=z['physical_ids'];rx=z['rx'];scene=z['scene'];prior=z['prior_values'];candidates=z['candidate_predictions'];choice=z['choice'];prediction=z['prediction']
 if ids.shape!=(11761,) or len(set(ids.tolist()))!=11761 or candidates.shape!=(11761,5) or prior.shape!=(11761,15) or choice.shape!=(11761,) or prediction.shape!=(11761,) or not np.isin(candidates,np.arange(6)).all() or not np.isin(choice,np.arange(5)).all() or not np.array_equal(prediction,candidates[np.arange(len(ids)),choice]):raise ValueError('Valid fixed candidate-selection predictions required')
 cache=a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache';physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False);known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if not np.array_equal(physical[ids],physical_saved) or int(known.sum())!=1047 or not np.all(known|unlabeled):raise ValueError('Exact source roles and physical IDs required')
 # All receiver-fold predictions were fixed before these L_s identities are joined.
 meta=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:ls_ids=z['ids'];ls_labels=z['y']
 if ls_ids.tolist()!=contract['role_ids']['L_s'] or len(ls_ids)!=6300 or not np.all((ls_labels>=0)&(ls_labels<6)):raise ValueError('Exact authorized source L_s labels required')
 mapping={str(k):int(v) for k,v in zip(ls_ids,ls_labels)}
 if len(mapping)!=6300 or not all(str(k) in mapping for k in physical[ids[known]]) or not all(str(k) not in mapping for k in physical[ids[unlabeled]]):raise ValueError('Source label role binding failed')
 truth=np.full(len(ids),-1,dtype=np.int64);truth[known]=[mapping[str(k)] for k in physical[ids[known]]];pseudo=prior[:,0].astype(np.int64);trusted=prior[:,1]>=.95;pred={name:candidates[:,i] for i,name in enumerate(NAMES)};pred['router']=prediction;results={}
 for group,receivers in [('RX4_6',(4,6)),('RX8',(8,))]:
  results[group]={}
  for view in VIEWS:
   mask=np.isin(rx,receivers)&((scene!='clean') if view=='all_impaired' else (scene==view));ps=mask&trusted;ls=mask&known;cell={'count':int(mask.sum()),'pseudo_count':int(ps.sum()),'L_s_count':int(ls.sum()),'selected_counts':{name:int(np.count_nonzero(mask&(choice==i))) for i,name in enumerate(NAMES)}}
   for key,sel,target in [('pseudo',ps,pseudo),('L_s',ls,truth)]:cell[key+'_oracle_correct']=int(np.count_nonzero(sel&np.any(candidates==target[:,None],axis=1)))
   for name,decisions in pred.items():
    item={}
    for key,sel,target in [('pseudo',ps,pseudo),('L_s',ls,truth)]:
     good=decisions==target;base=pred['physical']==target;item[key+'_correct']=int(np.count_nonzero(sel&good));item[key+'_repairs_vs_physical']=int(np.count_nonzero(sel&good&~base));item[key+'_damages_vs_physical']=int(np.count_nonzero(sel&~good&base))
    cell[name]=item
   results[group][view]=cell
 out=dict(status='SOURCE_GATE_ROUTER_SCORE_COMPLETE',prediction_sha256=h,fixed_predictions_before_source_scoring=True,L_s_scored=int(known.sum()),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,results=results,caveat='Cross-fitted source diagnostics on previously explored IDs. Clean F pseudo target is not identity truth; oracle uses labels and cannot be deployed. IQ waveforms not recomputed.')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':out['status'],'RX4_6':results['RX4_6']['all_impaired'],'RX8':results['RX8']['all_impaired']}),flush=True)
if __name__=='__main__':main()
