"""Cross-fit gate-aware five-candidate selector using source labels and pseudo labels."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from gate_aware_router import GateAwareRouter
from class_aware_router import multi_positive_loss
from source_pilot import s,core
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_prior_pnp'))
from train_multiview_unroll import allowed_fit_labels

NAMES=['raw','physical','unroll025','unroll05','unroll1'];PRIOR='20261102-phase1-unroll-source-replay-s2026092701-r01';PRIOR_SHA='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def fit_router(pred,gate,target,weights,steps):
 represented=np.any(pred==target[:,None],axis=1);disagree=np.any(pred!=pred[:,:1],axis=1);eligible=(target>=0)&represented&disagree;idx=np.flatnonzero(eligible)
 if len(idx)<100:raise ValueError('At least 100 covered source training disagreements required')
 classes=np.bincount(target[idx],minlength=6)
 if np.any(classes==0):raise ValueError('All source classes require covered disagreements')
 w=weights[idx]/np.sqrt(classes[target[idx]]);w=w/w.mean();x=torch.tensor(pred[idx].tolist(),dtype=torch.long);g=torch.tensor(gate[idx].tolist(),dtype=torch.float32);y=torch.tensor(target[idx].tolist(),dtype=torch.long);wt=torch.tensor(w.tolist(),dtype=torch.float32);torch.manual_seed(2026154054);model=GateAwareRouter();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001);losses=[]
 for step in range(steps):
  loss=multi_positive_loss(model(x,g),x,y,wt);opt.zero_grad(set_to_none=True);loss.backward()
  if not torch.isfinite(loss) or any(q.grad is not None and not torch.isfinite(q.grad).all() for q in model.parameters()):raise ValueError('Finite source router training required')
  opt.step()
  if step in (0,steps-1):losses.append(float(loss.detach()))
 model.eval().requires_grad_(False);return model,dict(eligible_disagreements=len(idx),class_counts=classes.tolist(),source_L_s_count=int(np.count_nonzero(weights[idx]>1)),pseudo_count=int(np.count_nonzero(weights[idx]==1)),uncovered=int(np.count_nonzero((target>=0)&~represented)),unanimous=int(np.count_nonzero((target>=0)&~disagree)),loss_start=losses[0],loss_end=losses[-1])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source output required')
 torch.set_num_threads(1);folder=a.root/'runs'/PRIOR;m=core.read(folder/'prediction_complete.json');path=folder/'source_predictions.npz'
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('sha256')!=PRIOR_SHA or sha(path)!=PRIOR_SHA or any(m.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact fixed source candidate predictions required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'].copy();rx=z['rx'].copy();scene=z['scene'].copy();prior=z['values'].copy()
 if ids.shape!=(11761,) or len(np.unique(ids))!=11761 or prior.shape!=(11761,15) or not np.isfinite(prior).all():raise ValueError('Complete fixed source cohort required')
 gate=prior[:,3];active=gate>.8;raw=prior[:,5].astype(np.int64);pred=np.column_stack((raw,np.where(active,prior[:,7],raw),*[np.where(active,prior[:,i],raw) for i in (9,11,13)])).astype(np.int64)
 if not np.isin(pred,np.arange(6)).all() or np.any((gate<0)|(gate>1)) or not np.all(pred[~active]==pred[~active,:1]):raise ValueError('Valid fixed candidates and gate required')
 cache,clean,allrx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,allrx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
 if not np.array_equal(ids,np.r_[validation,heldout]) or not np.array_equal(rx,allrx[ids]) or not np.all(np.isin(roles[ids],['L_s','U_s'])) or np.intersect1d(ids,training).size:raise ValueError('Exact source cohort and roles required')
 _,contract=core.teacher(a.root,'cpu');choices=np.full(len(ids),-1,dtype=np.int64);decisions=np.full(len(ids),-1,dtype=np.int64);folds={};models={}
 for name,train_rx,eval_rx in [('train4_eval6',(4,),(6,)),('train6_eval4',(6,),(4,)),('train46_eval8',(4,6),(8,))]:
  train=np.isin(rx,train_rx);evaluate=np.isin(rx,eval_rx);other=~(train|evaluate);actual=allowed_fit_labels(a.root,cache,clean,roles,contract,ids[train],ids[evaluate],ids[other]);target=np.full(len(ids),-1,dtype=np.int64);known=train&(roles[ids]=='L_s');pseudo=train&(roles[ids]=='U_s')&(prior[:,1]>=.95);target[known]=actual[ids[known]];target[pseudo]=prior[pseudo,0].astype(np.int64);weights=np.ones(len(ids),dtype=np.float64);weights[known]=2.
  if np.any(target[~train]!=-1) or np.any(target[known]<0) or np.any(actual[ids[~train]]!=-1):raise ValueError('Only training receiver source labels may affect router')
  model,stats=fit_router(pred,gate,target,weights,5 if a.smoke else 400);folds[name]=dict(stats,train_receivers=list(train_rx),eval_receivers=list(eval_rx),training_known=int(known.sum()),training_pseudo=int(pseudo.sum()),evaluation_records=int(evaluate.sum()))
  with torch.no_grad():choice,decision=model.choose(torch.tensor(pred[evaluate].tolist(),dtype=torch.long),torch.tensor(gate[evaluate].tolist(),dtype=torch.float32))
  choices[evaluate]=np.from_dlpack(choice.contiguous());decisions[evaluate]=np.from_dlpack(decision.contiguous());models[name]=model
 if np.any(choices<0) or not np.array_equal(pred[np.arange(len(ids)),choices],decisions):raise ValueError('Every output must select an actual candidate')
 a.out.mkdir(parents=True);out=a.out/'source_predictions.npz'
 with out.open('xb') as f:np.savez(f,indices=ids,physical_ids=physical[ids],rx=rx,scene=scene,prior_values=prior,candidate_predictions=pred,choice=choices,prediction=decisions)
 model_hashes={}
 for name,model in models.items():
  ck=a.out/(name+'.pt')
  with ck.open('xb') as f:torch.save(dict(commit=a.commit,model=model.state_dict(),candidate_names=NAMES,prior_sha256=PRIOR_SHA,fold=folds[name]),f)
  model_hashes[name]=sha(ck)
 result=dict(status='SOURCE_GATE_ROUTER_SMOKE_PASS' if a.smoke else 'SOURCE_GATE_ROUTER_TRAINING_COMPLETE',commit=a.commit,count=len(ids),candidate_names=NAMES,prior_sha256=PRIOR_SHA,source_output_sha256=sha(out),fold_model_sha256=model_hashes,fold_training=folds,L_s_metadata_labels_read=6300,L_s_metadata_loads=3,pseudo_source='frozen clean F confidence >=0.95 on U_s',U_s_truth_reads=0,V_reads=0,test_reads=0,DURA_in_candidate=False,caveat='Source IDs and F previously explored; gate is a model score, not measured SNR; no blind or real satellite claim.')
 core.write(a.out/'result.json',result);print('GATE_ROUTER_SOURCE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
