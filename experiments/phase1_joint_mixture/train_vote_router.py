"""Fit a class-equivariant vote router using only legitimate L_s source truth."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from vote_router import VoteRouter

SOURCE_RUNS=[('free_mixture','20261102-phase1-free-mixture-labeled-source-s2026092701-r01'),('full_hard','20261102-phase1-free-full-hard-labeled-source-s2026092701-r01'),('hierarchical','20261102-phase1-hierarchical-prior-labeled-source-s2026092701-r01'),('joint_mixture','20261102-phase1-joint-mixture-labeled-source-s2026092701-r01'),('confidence_router','20261102-phase1-confidence-router-labeled-source-s2026092701-r01')]
NEW='20261102-phase1-complementary-source-replay-s2026092701-r02'
NAMES=['raw']+[x[0] for x in SOURCE_RUNS]+['attention','structured_prior','white_prior']

def load_fixed(root):
 ref=None;pred=[];hashes={}
 for name,run in SOURCE_RUNS:
  folder=root/'runs'/run;m=json.loads((folder/'prediction_complete.json').read_text());path=folder/'predictions.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('role')!='L_s' or m.get('count')!=1047 or m.get('TX_truth_reads')!=0 or digest!=m.get('predictions_sha256'):raise ValueError('Fixed authorized source predictions required: '+name)
  with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in ('indices','physical_ids','rx','scene','part','predictions','active')}
  if ref is None:ref=data
  elif any(not np.array_equal(data[k],ref[k]) for k in ('indices','physical_ids','rx','scene','part','active')) or not np.array_equal(data['predictions'][:,1],ref['predictions'][:,1]):raise ValueError('Source candidate alignment mismatch')
  pred.append(data['predictions'][:,4]);hashes[name]=digest
 folder=root/'runs'/NEW;m=json.loads((folder/'prediction_complete.json').read_text());path=folder/'source_predictions.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('count')!=1047 or m.get('candidate_names')!=NAMES[-3:] or m.get('TX_truth_reads')!=0 or m.get('U_s_truth_reads')!=0 or m.get('test_reads')!=0 or digest!=m.get('sha256'):raise ValueError('Exact complementary source predictions required')
 with np.load(path,allow_pickle=False) as z:
  if any(not np.array_equal(z[k],ref[k]) for k in ('indices','physical_ids','rx','scene','part','active')) or not np.array_equal(z['raw'],ref['predictions'][:,1]):raise ValueError('Complementary source alignment mismatch')
  new=z['candidate_predictions'].copy()
 if new.shape!=(1047,3):raise ValueError('Three complementary predictions required')
 hashes['complementary']=digest;allpred=np.column_stack([ref['predictions'][:,1],*pred,new]);
 if allpred.shape!=(1047,9) or np.any((allpred<0)|(allpred>=6)):raise ValueError('Valid candidate predictions required')
 return ref,allpred,hashes

def fit(predictions,truth,train_mask):
 represented=np.any(predictions==truth[:,None],axis=1);disagreement=np.any(predictions!=predictions[:,[0]],axis=1);index=np.flatnonzero(train_mask&represented&disagreement)
 if len(index)<20:raise ValueError('At least 20 covered source disagreements required')
 x=torch.tensor(predictions[index].tolist(),dtype=torch.long);y=torch.tensor(truth[index].tolist(),dtype=torch.long);counts=torch.bincount(y,minlength=6).float()
 if (counts==0).any():raise ValueError('All six source classes require covered disagreements')
 weights=(1/counts)[y];weights=weights/weights.mean();torch.manual_seed(2026130054);model=VoteRouter();opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001);losses=[]
 for step in range(400):
  logits=model(x);loss=(F.cross_entropy(logits,y,reduction='none')*weights).mean();opt.zero_grad(set_to_none=True);loss.backward()
  if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Finite source router optimization required')
  opt.step()
  if step in (0,399):losses.append(float(loss.detach()))
 model.eval().requires_grad_(False);return model,dict(training_disagreements=len(index),train_counts=counts.int().tolist(),loss_start=losses[0],loss_end=losses[-1],uncovered=int(np.count_nonzero(train_mask&~represented)),unanimous=int(np.count_nonzero(train_mask&~disagreement)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable router output required')
 torch.set_num_threads(2);ref,pred,hashes=load_fixed(a.root);cache=a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';roles=np.load(cache/'roles.npy',allow_pickle=False);physical=np.load(cache/'physical_ids.npy',allow_pickle=False);ids=ref['indices']
 if not np.all(roles[ids]=='L_s') or not np.array_equal(physical[ids],ref['physical_ids']) or len(set(ids.tolist()))!=1047:raise ValueError('Exact L_s source role and IDs required')
 f0=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0';contract=json.loads((f0/'source_contract.json').read_text())
 # Labels are read only after all source candidate outputs are fixed and bound.
 with np.load(f0/'L_s_metadata.npz',allow_pickle=False) as z:source_ids=z['ids'];labels=z['y']
 if list(source_ids)!=contract['role_ids']['L_s'] or labels.shape!=(6300,) or len(set(source_ids.tolist()))!=6300:raise ValueError('Authorized L_s label metadata required')
 lookup={str(k):int(v) for k,v in zip(source_ids,labels)};truth=np.array([lookup[str(k)] for k in ref['physical_ids']]);rx=ref['rx'];choice=np.full(len(ids),-1,dtype=np.int64);prediction=np.full(len(ids),-1,dtype=np.int64);foldstats={};final=None
 for name,train_rx,eval_rx in [('train4_eval6',(4,),(6,)),('train6_eval4',(6,),(4,)),('train46_eval8',(4,6),(8,))]:
  train=np.isin(rx,train_rx);evaluate=np.isin(rx,eval_rx);model,stats=fit(pred,truth,train)
  with torch.no_grad():c,v=model.choose(torch.tensor(pred[evaluate].tolist(),dtype=torch.long))
  choice[evaluate]=np.from_dlpack(c.contiguous());prediction[evaluate]=np.from_dlpack(v.contiguous());foldstats[name]=stats
  if name=='train46_eval8':final=model
 if np.any(choice<0) or np.any(prediction<0) or not np.array_equal(pred[np.arange(len(ids)),choice],prediction):raise ValueError('Every source prediction must select an existing candidate')
 a.out.mkdir(parents=True);path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,physical_ids=ref['physical_ids'],rx=rx,scene=ref['scene'],candidate_predictions=pred,choice=choice,prediction=prediction)
 cp=a.out/'checkpoint.pt'
 with cp.open('xb') as f:torch.save(dict(commit=a.commit,model=final.state_dict(),candidate_names=NAMES,source_prediction_sha256=hashes,fit_receivers=[4,6],training_steps=400,seed=2026130054),f)
 result=dict(status='SOURCE_ROUTER_TRAINING_COMPLETE',commit=a.commit,count=len(ids),candidate_names=NAMES,source_prediction_sha256=hashes,source_output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),fold_training=foldstats,L_s_metadata_labels_read=6300,L_s_role='fit RX4 or RX6 and heldout other receiver; final fit RX4/6, source holdout RX8',U_s_truth_reads=0,V_reads=0,test_reads=0,DURA_in_candidate=False,caveat='Source-only fit; base F and candidate models previously saw source IDs; ground development test has been observed.')
 (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('SOURCE_VOTE_ROUTER_TRAINING_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
