"""Receiver-crossfit source L_s reliability selector with fixed predictions before scoring."""
import argparse,json
from pathlib import Path
import numpy as np
from reliability_router import ReliabilityRouter,choose_by_score
from agreement_source_crossfit import source_data,source_truth,core,VIEWS

MARGINS=(0.,.05,.1)
FEATURE_RUN='20261102-phase1-candidate-reliability-features-s2026092701-r01'

def predict(root,out,commit):
 if out.exists():raise FileExistsError('Immutable source output required')
 out.mkdir(parents=True);cache,clean,roles,ids,base,cand=source_data(root);done=core.read(root/'runs'/FEATURE_RUN/'result.json')
 if done.get('status')!='SOURCE_CANDIDATE_FEATURES_FIXED' or done.get('count')!=len(ids) or done.get('TX_truth_reads')!=0:raise ValueError('Fixed source candidate features required')
 with np.load(root/'runs'/FEATURE_RUN/'features.npz',allow_pickle=False) as z:f={k:z[k] for k in z.files}
 if not np.array_equal(f['indices'],ids) or not np.array_equal(f['physical_ids'],base['physical_ids']) or not np.array_equal(f['rx'],base['rx']) or not np.array_equal(f['scene'],base['scene']) or not np.array_equal(f['predictions'],cand) or f['features'].shape!=(len(ids),5,19) or not np.isfinite(f['features']).all():raise ValueError('Aligned finite source candidate features required')
 truth=source_truth(root,cache,clean,roles,ids);chosen=np.full((len(ids),len(MARGINS)),-1,dtype=np.int64);scores=np.full((len(ids),5),np.nan,dtype=np.float64);folds={}
 for target,train_rx in ((4,(6,)),(6,(4,)),(8,(4,6))):
  train=np.isin(base['rx'],train_rx);test=base['rx']==target
  if not train.any() or not test.any() or np.any(train&test):raise ValueError('Disjoint source receiver folds required')
  model=ReliabilityRouter().fit(f['features'][train],cand[train],truth[train]);scores[test]=model.score(f['features'][test]);folds[str(target)]={'train_receivers':list(train_rx),'train_count':int(train.sum()),'test_count':int(test.sum())}
  for j,margin in enumerate(MARGINS):chosen[test,j]=choose_by_score(scores[test],cand[test],margin)
 if np.any(chosen<0) or not np.isfinite(scores).all():raise ValueError('Complete heldout source receiver decisions required')
 with (out/'predictions.npz').open('xb') as z:np.savez(z,indices=ids,physical_ids=base['physical_ids'],rx=base['rx'],scene=base['scene'],candidates=cand,chosen=chosen,scores=scores,clean_prediction=base['predictions'][:,0])
 done={'status':'SOURCE_CROSSFIT_PREDICTIONS_FIXED','commit':commit,'count':len(ids),'folds':folds,'margins':list(MARGINS),'training_truth':'L_s only, disjoint receiver per fold','holdout_labels_used_for_that_fold_fit':False,'U_s_truth_reads':0,'V_reads':0,'test_reads':0}
 with (out/'prediction_complete.json').open('x',encoding='utf8') as z:json.dump(done,z,indent=2);z.write('\n')
 print('RELIABILITY_CROSSFIT_PREDICTIONS_FIXED',json.dumps(done),flush=True)

def score(root,out,commit):
 done=core.read(out/'prediction_complete.json')
 if done.get('status')!='SOURCE_CROSSFIT_PREDICTIONS_FIXED' or done.get('commit')!=commit or done.get('holdout_labels_used_for_that_fold_fit') is not False:raise ValueError('Fixed source crossfit decisions required')
 with np.load(out/'predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 cache,clean,roles,ids,base,cand=source_data(root)
 if not np.array_equal(ids,d['indices']) or not np.array_equal(base['physical_ids'],d['physical_ids']) or not np.array_equal(cand,d['candidates']) or np.any((d['chosen']<0)|(d['chosen']>=5)):raise ValueError('Matching fixed source predictions required')
 truth=source_truth(root,cache,clean,roles,ids);cells={}
 for receiver in (4,6,8):
  for view in ('all_impaired',*VIEWS):
   mask=(d['rx']==receiver)&((d['scene']!='clean') if view=='all_impaired' else (d['scene']==view));n=int(mask.sum())
   if n==0:raise ValueError('All receiver/scene source cells required')
   baseok=cand[:,0]==truth;cleanok=d['clean_prediction']==truth;oracle=(cand==truth[:,None]).any(1);cell={'count':n,'clean_correct':int((mask&cleanok).sum()),'dura_correct':int((mask&baseok).sum()),'oracle_correct':int((mask&oracle).sum()),'margins':{}}
   for j,margin in enumerate(MARGINS):
    choice=d['chosen'][:,j];prediction=cand[np.arange(len(cand)),choice];ok=prediction==truth;cell['margins'][str(margin)]={'correct':int((mask&ok).sum()),'repairs':int((mask&~baseok&ok).sum()),'damages':int((mask&baseok&~ok).sum()),'switched':int((mask&(choice!=0)).sum())}
   cells[f'{receiver}:{view}']=cell
 result={'status':'SOURCE_CROSSFIT_SCORE_COMPLETE','commit':commit,'count':len(ids),'cells':cells,'source_only':True,'training_truth':'L_s only disjoint receiver per fold','U_s_truth_reads':0,'V_reads':0,'test_reads':0,'caveat':'Source identities and other channel views were previously used by F/development; logistic scores are not guaranteed calibrated on unseen receivers. Oracle uses true labels and is not deployable.'}
 with (out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('RELIABILITY_CROSSFIT_SCORE_COMPLETE',json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.mode=='predict':predict(a.root,a.out,a.commit)
 else:score(a.root,a.out,a.commit)
if __name__=='__main__':main()
