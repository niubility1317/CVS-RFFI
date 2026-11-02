"""Cross-fit class-aware waveform router using source L_s and clean-F pseudo labels."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from class_aware_router import ClassAwareRouter,multi_positive_loss
from source_pilot import s,core
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_prior_pnp'))
from train_multiview_unroll import allowed_fit_labels

NAMES=['raw','attention','structured_prior','white_prior','free_mixture'];COMPLEMENT='20261102-phase1-complementary-full-source-s2026092701-r01';FREE='20261102-phase1-free-mixture-full-source-s2026092701-r01';PRIOR_SHA='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5'

def load_predictions(root):
 runs=root/'runs';items=[];hashes={}
 for name,run,commit in [('complementary',COMPLEMENT,'5d39fc51d7b93aca9f846238c26836e10fe0b1a4'),('free',FREE,'a69b3755b72be7fd23ae78e00ff05283b191d6b1')]:
  folder=runs/run;m=core.read(folder/'prediction_complete.json');path=folder/'source_predictions.npz';h=hashlib.sha256(path.read_bytes()).hexdigest()
  if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!=commit or m.get('count')!=11761 or m.get('sha256')!=h or m.get('prior_sha256')!=PRIOR_SHA or m.get('DURA_in_candidate') is not False or any(m.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')):raise ValueError('Exact full source candidate predictions required: '+name)
  audit=core.read(runs/'20261102-phase1-router-full-source-audit-s2026092701-r01'/(name+'.json'))
  if audit.get('status')!='PASS' or audit.get('kind')!=name or audit.get('prediction_sha256')!=h:raise ValueError('Independent full source prediction audit required: '+name)
  with np.load(path,allow_pickle=False) as z:items.append({k:z[k].copy() for k in z.files})
  hashes[name]=h
 comp,free=items
 if any(not np.array_equal(comp[k],free[k]) for k in ('indices','rx','scene','prior_values')) or comp['candidate_predictions'].shape!=(11761,3) or free['candidate_prediction'].shape!=(11761,):raise ValueError('Exact matching source records and common physical reference required')
 v=comp['prior_values'];pred=np.column_stack((v[:,5].astype(np.int64),comp['candidate_predictions'],free['candidate_prediction']))
 if pred.shape!=(11761,5) or not np.isin(pred,np.arange(6)).all() or not np.all(pred[v[:,3]<=.8]==pred[v[:,3]<=.8,:1]):raise ValueError('Valid fixed source predictions and inactive-gate identity required')
 return comp,pred,hashes

def fit_router(pred,target,weights):
 represented=np.any(pred==target[:,None],axis=1);disagree=np.any(pred!=pred[:,:1],axis=1);eligible=(target>=0)&represented&disagree;idx=np.flatnonzero(eligible)
 if len(idx)<100:raise ValueError('At least 100 covered source training disagreements required')
 classes=np.bincount(target[idx],minlength=6)
 if np.any(classes==0):raise ValueError('All six source classes require covered disagreements')
 w=weights[idx]/np.sqrt(classes[target[idx]]);w=w/w.mean();x=torch.tensor(pred[idx].tolist(),dtype=torch.long);y=torch.tensor(target[idx].tolist(),dtype=torch.long);wt=torch.tensor(w.tolist(),dtype=torch.float32);torch.manual_seed(2026142054);model=ClassAwareRouter();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001);losses=[]
 for step in range(400):
  loss=multi_positive_loss(model(x),x,y,wt);opt.zero_grad(set_to_none=True);loss.backward()
  if not torch.isfinite(loss) or any(q.grad is not None and not torch.isfinite(q.grad).all() for q in model.parameters()):raise ValueError('Finite source router training required')
  opt.step()
  if step in (0,399):losses.append(float(loss.detach()))
 model.eval().requires_grad_(False);return model,dict(eligible_disagreements=len(idx),class_counts=classes.tolist(),source_L_s_count=int(np.count_nonzero(weights[idx]>1)),pseudo_count=int(np.count_nonzero(weights[idx]==1)),uncovered=int(np.count_nonzero((target>=0)&~represented)),unanimous=int(np.count_nonzero((target>=0)&~disagree)),loss_start=losses[0],loss_end=losses[-1])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable router output required')
 torch.set_num_threads(2);ref,pred,hashes=load_predictions(a.root);ids=ref['indices'];rx=ref['rx'];scene=ref['scene'];prior=ref['prior_values'];cache,clean,allrx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,allrx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
 if not np.array_equal(ids,np.r_[validation,heldout]) or not np.array_equal(rx,allrx[ids]) or not np.all(np.isin(roles[ids],['L_s','U_s'])) or np.intersect1d(ids,training).size:raise ValueError('Exact source validation and heldout cohort required')
 _,contract=core.teacher(a.root,'cpu');choices=np.full(len(ids),-1,dtype=np.int64);decisions=np.full(len(ids),-1,dtype=np.int64);folds={};models={}
 for name,train_rx,eval_rx in [('train4_eval6',(4,),(6,)),('train6_eval4',(6,),(4,)),('train46_eval8',(4,6),(8,))]:
  train=np.isin(rx,train_rx);evaluate=np.isin(rx,eval_rx);other=~(train|evaluate);actual=allowed_fit_labels(a.root,cache,clean,roles,contract,ids[train],ids[evaluate],ids[other]);target=np.full(len(ids),-1,dtype=np.int64);known=train&(roles[ids]=='L_s');pseudo=train&(roles[ids]=='U_s')&(prior[:,1]>=.95);target[known]=actual[ids[known]];target[pseudo]=prior[pseudo,0].astype(np.int64);weights=np.ones(len(ids),dtype=np.float64);weights[known]=2.
  if np.any(target[~train]!=-1) or np.any(target[known]<0) or np.any(actual[ids[~train]]!=-1):raise ValueError('Only training receiver source labels may affect router fit')
  model,stats=fit_router(pred,target,weights);folds[name]=dict(stats,train_receivers=list(train_rx),eval_receivers=list(eval_rx),training_known=int(known.sum()),training_pseudo=int(pseudo.sum()),evaluation_records=int(evaluate.sum()))
  with torch.no_grad():choice,decision=model.choose(torch.tensor(pred[evaluate].tolist(),dtype=torch.long))
  choices[evaluate]=np.from_dlpack(choice.contiguous());decisions[evaluate]=np.from_dlpack(decision.contiguous());models[name]=model
 if np.any(choices<0) or not np.array_equal(pred[np.arange(len(ids)),choices],decisions):raise ValueError('Every prediction must select an actual candidate')
 a.out.mkdir(parents=True);path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,physical_ids=physical[ids],rx=rx,scene=scene,prior_values=prior,candidate_predictions=pred,choice=choices,prediction=decisions)
 model_hashes={}
 for name,model in models.items():
  ck=a.out/(name+'.pt')
  with ck.open('xb') as f:torch.save(dict(commit=a.commit,model=model.state_dict(),candidate_names=NAMES,source_prediction_sha256=hashes,fold=folds[name]),f)
  model_hashes[name]=hashlib.sha256(ck.read_bytes()).hexdigest()
 result=dict(status='SOURCE_ROUTER_TRAINING_COMPLETE',commit=a.commit,count=len(ids),candidate_names=NAMES,source_prediction_sha256=hashes,source_output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fold_model_sha256=model_hashes,fold_training=folds,L_s_metadata_labels_read=6300,L_s_metadata_loads=3,pseudo_source='frozen clean F confidence >=0.95 on U_s',U_s_truth_reads=0,V_reads=0,test_reads=0,DURA_in_candidate=False,caveat='Previously explored source IDs; clean F pseudo labels are not identity truth; no blind ground claim.')
 core.write(a.out/'result.json',result);print('CLASS_AWARE_SOURCE_ROUTER_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
