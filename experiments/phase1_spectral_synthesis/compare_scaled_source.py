"""Compare fixed small and scaled source models with one prior source gate."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

def route_pair(small,large,probability):
 if small.ndim!=2 or small.shape[1]!=8 or large.shape!=small.shape or probability.shape!=(len(small),):raise ValueError('Matched source predictions and gate required')
 on=probability>.8;return np.stack((small[:,0],small[:,1],small[:,2],np.where(on,small[:,6],small[:,2]),np.where(on,large[:,6],large[:,2]),np.where(on,small[:,7],small[:,3]),np.where(on,large[:,7],large[:,3])),axis=1)

def main():
 import torch
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
 from source_pilot import s,core,labels_source
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source comparison required')
 runs=a.root/'runs';small=runs/'20261102-phase1-spectral-synthesis-pilot-s2026092701-r01';large=runs/'20261102-phase1-spectral-synthesis-scaled-s2026092701-r01';prior=runs/'20261102-phase1-unroll-source-replay-s2026092701-r01';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mans=[core.read(p/'prediction_complete.json') for p in (small,large)];paths=[p/'source_predictions.npz' for p in (small,large)];commits=['059987ed75b568b4ce052d7638ee0262bd5652c3','d67348d3312a24917681c66d4e3074e3f3ff86c1']
 for m,path,commit in zip(mans,paths,commits):
  if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!=commit or m.get('count')!=672 or m.get('prediction_sha256')!=sha(path) or m.get('DURA_loaded') is not False or any(m.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads','model_updates_after_prediction')):raise ValueError('Both exact fixed source predictions required')
 states=[torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False) for p in (small,large)];smallfit,largefit=(x['fit_ids'] for x in states)
 if len(smallfit)!=1024 or len(largefit)!=8192 or not np.isin(smallfit,largefit).all() or any(x.get('commit')!=c for x,c in zip(states,commits)):raise ValueError('Exact nested source fit checkpoints required')
 with np.load(paths[0],allow_pickle=False) as x,np.load(paths[1],allow_pickle=False) as y:
  ids=x['indices'].copy();split=x['split'].copy();rx=x['rx'].copy();scene=x['scene'].copy();u=x['values'].copy();v=y['values'].copy()
  if not all(np.array_equal(x[k],y[k]) for k in ('indices','split','rx','scene')) or not np.array_equal(u[:,[1,2,4]],v[:,[1,2,4]]) or not np.allclose(u[:,[0,3,5]],v[:,[0,3,5]],rtol=1e-6,atol=1e-7):raise ValueError('Matched source views and common base predictions required')
 if len(ids)!=672 or len(set(ids.tolist()))!=672 or np.intersect1d(ids,largefit).size or u.shape!=(672,8) or v.shape!=(672,8) or not np.isfinite(u).all() or not np.isfinite(v).all():raise ValueError('Unique disjoint finite source cohort required')
 pm=core.read(prior/'prediction_complete.json');pp=prior/'source_predictions.npz'
 if pm.get('sha256')!='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5' or sha(pp)!=pm['sha256']:raise ValueError('Exact fixed source gate predictions required')
 with np.load(pp,allow_pickle=False) as z:
  index={int(v):i for i,v in enumerate(z['indices'])};rows=np.array([index[int(i)] for i in ids]);old=z['values'][rows].copy()
  if not np.array_equal(rx,z['rx'][rows]) or not np.array_equal(scene,z['scene'][rows]) or not np.array_equal(u[:,1],old[:,0]) or not np.array_equal(u[:,2],old[:,5]) or not np.allclose(u[:,0],old[:,1],rtol=1e-6,atol=1e-7):raise ValueError('Source prior IDs, views and base F must match')
 probability=old[:,3];active=probability>.8
 if not np.array_equal(u[active,4],old[active,7]):raise ValueError('Common active physical posterior decisions required')
 values=route_pair(u,v,probability);a.out.mkdir(parents=True);path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,split=split,rx=rx,scene=scene,gate_probability=probability,values=values)
 m=dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=len(ids),sha256=sha(path),small_sha256=mans[0]['prediction_sha256'],scaled_sha256=mans[1]['prediction_sha256'],prior_sha256=pm['sha256'],model_sha256=[sha(p/'checkpoint.pt') for p in (small,large)],gate_threshold=.8,gate_active_count=int(active.sum()),source_only=True,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0)
 core.write(a.out/'prediction_complete.json',m)
 torch.set_num_threads(2);cache,clean,allrx,training,pv,ph=s.source_contract(a.root);roles=np.load(cache/'train/roles.npy',allow_pickle=False);known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if not np.all(known|unlabeled):raise ValueError('Only source roles required')
 _,contract=core.teacher(a.root,'cpu');truth=labels_source.source_labels(a.root,cache,clean,roles,contract)[ids]
 if np.any(truth[known]<0) or np.any(truth[unlabeled]!=-1):raise ValueError('Only authorized L_s identity truth required')
 trusted=values[:,0]>=.95;pseudo=values[:,1];results={}
 for group in ('RX4_6','RX8'):
  results[group]={}
  for view in ['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']:
   mask=(split==group)&((scene!='clean') if view=='all_impaired' else (scene==view));ps=mask&trusted;ls=mask&known;cell={'count':int(mask.sum()),'pseudo_count':int(ps.sum()),'L_s_count':int(ls.sum()),'active_count':int(np.count_nonzero(mask&active))}
   for name,col,iq in [('small',3,5),('scaled',4,6)]:
    item={'iq_error_mean':float(values[mask,iq].mean())}
    for key,sel,target in [('pseudo',ps,pseudo),('L_s',ls,truth)]:
     good=values[:,col]==target;base=values[:,3]==target;item[key+'_correct']=int(np.count_nonzero(sel&good));item[key+'_repairs_vs_small']=int(np.count_nonzero(sel&good&~base));item[key+'_damages_vs_small']=int(np.count_nonzero(sel&~good&base))
    cell[name]=item
   results[group][view]=cell
 score=dict(status='SOURCE_PAIRED_SCORE_COMPLETE',prediction_sha256=m['sha256'],fixed_predictions_before_truth=True,L_s_scored=int(known.sum()),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,results=results,caveat='Same developmental source records already seen in prior analyses; no ground test or blind claim.')
 core.write(a.out/'source_score.json',score);print(json.dumps({'status':score['status'],'RX4_6':results['RX4_6']['all_impaired'],'RX8':results['RX8']['all_impaired']}),flush=True)
if __name__=='__main__':main()
