"""Cross-receiver source-L_s-only agreement routing; fixed predictions before scoring."""
import argparse,json
from pathlib import Path
import numpy as np
from agreement_router import AgreementRouter
from source_union_audit import RUNS,s,core,VIEWS,labels_source

MARGINS=(0.,.03,.1)

def source_data(root):
 data={}
 for name,run in RUNS.items():
  folder=root/'runs'/run;done=core.read(folder/'prediction_complete.json')
  if done.get('status')!='SOURCE_PREDICTIONS_FIXED' or done.get('role')!='L_s' or done.get('TX_truth_reads')!=0 or done.get('count')!=1047:raise ValueError('Fixed source L_s predictions required')
  with np.load(folder/'predictions.npz',allow_pickle=False) as z:data[name]={k:z[k] for k in z.files}
 base=data['hierarchical']
 for d in data.values():
  for k in ('indices','physical_ids','rx','scene','part','active'):
   if not np.array_equal(d[k],base[k]):raise ValueError('Aligned source cohorts required')
  if not np.array_equal(d['predictions'][:,:4],base['predictions'][:,:4]):raise ValueError('Common fixed source baseline required')
 cache,clean,rx,training,pv,ph=s.source_contract(root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);ids=base['indices'];physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
 if np.any(roles[ids]!='L_s') or not np.array_equal(physical[ids],base['physical_ids']) or np.intersect1d(ids,training).size or not np.array_equal(ids,np.r_[validation[roles[validation]=='L_s'],heldout[roles[heldout]=='L_s']]):raise ValueError('Complete source L_s cohorts required')
 cand=np.column_stack((base['predictions'][:,2],base['predictions'][:,3],base['predictions'][:,4],data['joint']['predictions'][:,4],data['free']['predictions'][:,4]))
 return cache,clean,roles,ids,base,cand

def source_truth(root,cache,clean,roles,ids):
 _,contract=core.teacher(root,'cpu');truth=labels_source.source_labels(root,cache,clean,roles,contract)[ids]
 if np.any((truth<0)|(truth>=6)):raise ValueError('Valid L_s truth required')
 return truth

def predict(root,out,commit):
 if out.exists():raise FileExistsError('Immutable source output required')
 out.mkdir(parents=True);cache,clean,roles,ids,base,cand=source_data(root);truth=source_truth(root,cache,clean,roles,ids);chosen=np.full((len(ids),len(MARGINS)),-1,dtype=np.int64);folds={}
 for target,train_rx in ((4,(6,)),(6,(4,)),(8,(4,6))):
  train=np.isin(base['rx'],train_rx);test=base['rx']==target
  if not train.any() or not test.any() or np.any(train&test):raise ValueError('Disjoint source receiver folds required')
  folds[str(target)]={'train_receivers':list(train_rx),'train_count':int(train.sum()),'test_count':int(test.sum())}
  for j,margin in enumerate(MARGINS):
   router=AgreementRouter(prior_strength=16.,min_count=8,margin=margin).fit(cand[train],truth[train]);chosen[test,j]=router.choose(cand[test])
 if np.any(chosen<0):raise ValueError('Complete heldout source receiver predictions required')
 with (out/'predictions.npz').open('xb') as f:np.savez(f,indices=ids,physical_ids=base['physical_ids'],rx=base['rx'],scene=base['scene'],candidates=cand,chosen=chosen,clean_prediction=base['predictions'][:,0])
 done={'status':'SOURCE_CROSSFIT_PREDICTIONS_FIXED','commit':commit,'count':len(ids),'folds':folds,'margins':list(MARGINS),'prior_strength':16.,'min_count':8,'training_truth':'L_s only, disjoint receiver per fold','holdout_labels_used_for_that_fold_fit':False,'U_s_truth_reads':0,'V_reads':0,'test_reads':0}
 with (out/'prediction_complete.json').open('x',encoding='utf8') as f:json.dump(done,f,indent=2);f.write('\n')
 print('AGREEMENT_CROSSFIT_PREDICTIONS_FIXED',json.dumps(done),flush=True)

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
 result={'status':'SOURCE_CROSSFIT_SCORE_COMPLETE','commit':commit,'count':len(ids),'cells':cells,'source_only':True,'training_truth':'L_s only disjoint receiver per fold','U_s_truth_reads':0,'V_reads':0,'test_reads':0,'caveat':'All source identities and other channel views were previously used by F/development. Oracle requires true labels and is not deployable; source holdouts are not independent target test.'}
 with (out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('AGREEMENT_CROSSFIT_SCORE_COMPLETE',json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.mode=='predict':predict(a.root,a.out,a.commit)
 else:score(a.root,a.out,a.commit)
if __name__=='__main__':main()
