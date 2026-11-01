"""Phase1 L_s source truth-last diagnostic of a fixed blind Bayesian restorer."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from labeled_counts import summarize

DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-blind-bayes-source-validation-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='bcc0c56686ad57d0c101d7b0a7fc7822dd6e5ccb':raise ValueError('Exact source validation dependency required')
sys.path.insert(0,str(DEP/'experiments/phase1_blind_bayes_validation'))
from blind_bayes_source_validation import BlindActionNet,action_matrix,s,core,posterior,source_prior,SOURCE,SHA,VIEWS
from experiments.pamf_phase1_flow import feature_conditioned_experiment as labels_source

PRIOR_RUN='20261102-phase1-blind-bayes-source-validation-s2026092701-r01'
PRIOR_SHA='5e35516d54d3e3d1d1749933861f290c1f87c8ffdb4b863d2f9819914135ceb9'

def _load_source(root,device):
 ckpt=root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_path=root/'runs'/PRIOR_RUN/'source_prior.pt'
 if hashlib.sha256(ckpt.read_bytes()).hexdigest()!=SHA or hashlib.sha256(prior_path.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact fixed source model and prior required')
 state=torch.load(ckpt,map_location='cpu',weights_only=False);prior=torch.load(prior_path,map_location='cpu',weights_only=False)
 if state['epoch']!=10 or state['commit']!='94693a342bb2daa684c0f21415d7b949f9fb2107' or prior['source_ids']!=8192:raise ValueError('Fixed source provenance required')
 model=BlindActionNet().to(device);model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False);_,gate=s.initial_models(root,device);gate.eval().requires_grad_(False);teacher,contract=core.teacher(root,device);dura=source_prior.frozen_dura(root,device)
 return model,gate,teacher,contract,dura,prior['means'].to(device),prior['covariances'].to(device),prior['priors'].to(device)

@torch.no_grad()
def blind(model,gate,x,means,cov,prior,eye):
 active=gate(x).sigmoid()>.99;out=x.clone()
 if active.any():
  v=x[active];h,lv=model(v);A=action_matrix(h);N=eye[None,:,:]*lv.exp()[:,None,None];states,lp=posterior(v.reshape(len(v),512),A,N,means,cov,prior);out[active]=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(v)
 return out,active

def predict(root,out,commit,device,smoke):
 if out.exists():raise FileExistsError('Immutable output required')
 out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;cache,clean,rx,training,pv,ph=s.source_contract(root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);received,scene=s.epoch_data(cache,19)
 if smoke:ids=training[:32];parts=np.zeros(len(ids),dtype=np.int64)
 else:
  val=validation[roles[validation]=='L_s'];hold=heldout[roles[heldout]=='L_s'];ids=np.r_[val,hold];parts=np.r_[np.ones(len(val),dtype=np.int64),np.full(len(hold),2,dtype=np.int64)]
  if not len(val) or not len(hold) or np.any(roles[ids]!='L_s') or np.intersect1d(ids,training).size:raise ValueError('Source L_s receiver holdout required')
 model,gate,teacher,contract,dura,means,covs,priors=_load_source(root,device);eye=torch.eye(512,device=device);chunks=[];triggers=[];tick=time.monotonic()
 for start in range(0,len(ids),8):
  idx=ids[start:start+8];x=core.tensor(received[idx],device);clean_x=core.tensor(clean[idx],device);z,on=blind(model,gate,x,means,covs,priors,eye)
  variants=(clean_x,x,dura(x),z);row=torch.stack([teacher.classify_features(teacher.features(t)).argmax(1) for t in variants],1);chunks.append(row.cpu());triggers.append(on.cpu())
  if (start+len(idx))%256==0:print('LABELED_SOURCE_PREDICT',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
 predictions=np.from_dlpack(torch.cat(chunks).contiguous());active=np.from_dlpack(torch.cat(triggers).to(torch.uint8).contiguous());path=out/'predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,physical_ids=physical[ids],rx=rx[ids],scene=scene[ids],part=parts,predictions=predictions,active=active)
 manifest=dict(status='SOURCE_PREDICTIONS_FIXED',commit=commit,count=len(ids),source_only=True,role='smoke' if smoke else 'L_s',threshold=.99,checkpoint_sha256=SHA,prior_sha256=PRIOR_SHA,predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0)
 with (out/'prediction_complete.json').open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2);f.write('\n')
 print('LABELED_SOURCE_PREDICTIONS_FIXED',json.dumps(manifest),flush=True)

def score(root,out,commit,device):
 done=core.read(out/'prediction_complete.json');path=out/'predictions.npz'
 if done.get('status')!='SOURCE_PREDICTIONS_FIXED' or done.get('role')!='L_s' or done.get('commit')!=commit or done.get('TX_truth_reads')!=0 or hashlib.sha256(path.read_bytes()).hexdigest()!=done.get('predictions_sha256'):raise ValueError('Fixed source predictions required before source truth')
 with np.load(path,allow_pickle=False) as x:data={name:x[name] for name in x.files}
 cache,clean,rx,training,pv,ph=s.source_contract(root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);idx=data['indices']
 if len(idx)!=done['count'] or not np.array_equal(data['physical_ids'],physical[idx]) or np.any(roles[idx]!='L_s') or np.intersect1d(idx,training).size or not np.array_equal(data['rx'],rx[idx]):raise ValueError('Source L_s pairing mismatch')
 if not np.array_equal(idx,np.r_[validation[roles[validation]=='L_s'],heldout[roles[heldout]=='L_s']]):raise ValueError('Complete predeclared source cohorts required')
 teacher,contract=core.teacher(root,device);truth=labels_source.source_labels(root,cache,clean,roles,contract)[idx]
 if np.any(truth<0) or np.any(truth>=6):raise ValueError('Source L_s truth required')
 result={}
 for part,receivers in ((1,(4,6)),(2,(8,))):
  for view in VIEWS:
   mask=(data['part']==part)&np.isin(data['rx'],receivers)&(data['scene']==view);result[f'{part}:{view}']=summarize(data['predictions'],truth,mask)
 result=dict(status='SOURCE_LS_TRUTH_LAST_COMPLETE',commit=commit,predictions_sha256=done['predictions_sha256'],source_counts=result,source_only=True,threshold=.99,truth_role='L_s',U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='The frozen ground F previously trained on source IDs; the same source physical IDs have been explored in other channel views; not independent blind or target test.')
 with (out/'source_truth_score.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('LABELED_SOURCE_TRUTH_LAST_COMPLETE',json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.mode=='predict':predict(a.root,a.out,a.commit,a.device,a.smoke)
 else:
  if a.smoke:raise ValueError('No truth access in smoke')
  score(a.root,a.out,a.commit,a.device)
if __name__=='__main__':main()
