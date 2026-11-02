"""Fixed Phase1 seven-view predictions and truth-last score for learned mixture."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from source_validation import s,core,BlindActionNet,physical,SOURCE,PRIOR_RUN,MixtureController
LS='20261102-phase1-joint-mixture-labeled-source-s2026092701-r01'

def load_fixed(root,device):
 sel=core.read(root/'runs/20261102-phase1-joint-mixture-validation-s2026092701-r01/source_validation.json')['selection']['1.0']['primary_parent'];audit=core.read(root/'runs'/LS/'independent_audit.json')
 if sel!=dict(kind='residual',threshold=.8,validation_net=27,reference='parent') or audit.get('status')!='PASS':raise ValueError('Fixed source selection and source label audit required')
 bs=torch.load(root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p=torch.load(root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False);state=torch.load(root/'runs/20261102-phase1-joint-mixture-pilot-s2026092701-r01/checkpoint.pt',map_location='cpu',weights_only=False)
 if bs.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or bs.get('epoch')!=10 or p.get('source_ids')!=8192 or state.get('commit')!='c67dc6d7b6a3880ffe2fe4e49357ee4da18eb4ef':raise ValueError('Matching frozen Phase1 source models required')
 blind=BlindActionNet().to(device);blind.load_state_dict(bs['model']);blind.eval().requires_grad_(False);prior=(p['means'].to(device),p['covariances'].to(device),p['priors'].to(device));net=MixtureController().to(device);net.load_state_dict(state['model']);net.eval().requires_grad_(False);_,gate=s.initial_models(root,device);gate.eval().requires_grad_(False);teacher,_=core.teacher(root,device);teacher.eval().requires_grad_(False)
 return blind,gate,teacher,prior,net,torch.eye(512,device=device)

@torch.no_grad()
def predict_batch(x,models):
 blind,gate,teacher,prior,net,eye=models;raw_logits=teacher.classify_features(teacher.features(x));raw=raw_logits.argmax(1);active=gate(x).sigmoid()>.8;parent=raw.clone();routed=raw.clone();choice=torch.zeros(len(x),device=x.device,dtype=torch.long)
 if active.any():
  st,lp,v,z=physical(blind,teacher,x[active],prior,eye);candidate=net(v,lp,st,scale=1.)[0];pl=teacher.classify_features(teacher.features(z.reshape(-1,2,256)));cl=teacher.classify_features(teacher.features(candidate));parent[active]=pl.argmax(1);routed[active]=cl.argmax(1);choice[active]=1
  if not all(torch.isfinite(t).all() for t in (z,candidate,pl,cl)):raise ValueError('Finite fixed output required')
 if not torch.isfinite(raw_logits).all():raise ValueError('Finite input logits required')
 return raw,parent,routed,active,choice

def smoke(root,out,commit,device):
 if out.exists():raise FileExistsError('Immutable experiment output required')
 out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;models=load_fixed(root,device)
 with np.load(root/'runs'/LS/'predictions.npz',allow_pickle=False) as p:data={k:p[k] for k in p.files}
 cache,clean,rx,training,pv,ph=s.source_contract(root);received,scene=s.epoch_data(cache,19);ids=data['indices'];physical_ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
 if len(ids)!=1047 or not np.array_equal(physical_ids[ids],data['physical_ids']):raise ValueError('Fixed source audit cohort required')
 tested=[];chosen=None
 for batchsize in (32,8):
  mismatches=0;tick=time.monotonic()
  for start in range(0,len(ids),batchsize):
   ix=ids[start:start+batchsize];raw,parent,routed,active,choice=predict_batch(core.tensor(received[ix],device),models);actual=np.from_dlpack(torch.stack((raw,parent,routed,active.long(),choice),1).cpu().contiguous());expected=np.column_stack((data['predictions'][start:start+len(ix),1],data['predictions'][start:start+len(ix),3],data['predictions'][start:start+len(ix),4],data['active'][start:start+len(ix)],data['choice'][start:start+len(ix)]));mismatches+=int(np.count_nonzero(np.any(actual!=expected,axis=1)))
  tested.append(dict(batch=batchsize,records=len(ids),record_mismatches=mismatches,seconds=round(time.monotonic()-tick,3)))
  if mismatches==0:chosen=batchsize;break
 if chosen is None:raise ValueError('Fixed source audit predictions not reproduced')
 core.write(out/'prediction_smoke.json',dict(status='PASS',source_only=True,records=len(ids),tests=tested,batch=chosen,F_frozen=True,DURA_inference=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_IQ_reads=0,commit=commit));print('JOINT_MIXTURE_PHASE1_SMOKE PASS',json.dumps(tested),flush=True)

@torch.no_grad()
def predict(root,out,commit,device):
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;check=core.read(out/'prediction_smoke.json')
 if check.get('status')!='PASS' or check.get('commit')!=commit or check.get('batch') not in (8,32) or check.get('test_IQ_reads')!=0:raise ValueError('Source-only real-model smoke required')
 models=load_fixed(root,device);folder,ids=core.reference(root);dst=out/'flow'/'test_prediction';dst.mkdir(parents=True,exist_ok=False);np.save(dst/'physical_ids.npy',ids,allow_pickle=False);accepted={};batch=check['batch'];tick=time.monotonic()
 for view in core.VIEWS:
  values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
  if values.shape!=(len(ids),2,256) or values.dtype!=np.float32:raise ValueError('Complete Phase1 ground test IQ view required')
  raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));count=0
  for start in range(0,len(ids),batch):
   x=core.tensor(values[start:start+batch],device);a,_,b,on,_=predict_batch(x,models);raw[start:start+len(x)]=np.from_dlpack(a.cpu().contiguous());restored[start:start+len(x)]=np.from_dlpack(b.cpu().contiguous());count+=int(on.sum().item())
   if (start+len(x))%16384==0:print('JOINT_MIXTURE_PHASE1_PROGRESS',view,start+len(x),'seconds',round(time.monotonic()-tick,1),flush=True)
  raw.flush();restored.flush();del raw,restored;accepted[view]=count;print('JOINT_MIXTURE_PHASE1_VIEW_COMPLETE',view,len(ids),'active',count,'seconds',round(time.monotonic()-tick,1),flush=True)
 core.write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,commit=commit,TX_truth_reads=0,model_updates=0,accepted_by_view=accepted));print('JOINT_MIXTURE_PHASE1_PREDICTIONS_COMPLETE',len(ids),flush=True)

def score(root,out,commit):
 dst=out/'flow'/'test_prediction';done=core.read(dst/'complete.json');core.validate_prediction_manifest(done,count=168000,commit=commit)
 if done.get('TX_truth_reads')!=0 or done.get('model_updates')!=0:raise ValueError('Frozen complete predictions required before truth')
 for view in core.VIEWS:
  for arm in ('raw','restored'):
   x=np.load(dst/(view+'.'+arm+'.npy'),mmap_mode='r',allow_pickle=False)
   if x.shape!=(168000,) or x.dtype!=np.int64:raise ValueError('Complete fixed prediction arrays required')
 core.score(root,out,commit);print('JOINT_MIXTURE_PHASE1_TRUTH_SCORE_COMPLETE',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
 if a.mode=='smoke':smoke(a.root,a.out,a.commit,a.device)
 elif a.mode=='predict':predict(a.root,a.out,a.commit,a.device)
 else:score(a.root,a.out,a.commit)
if __name__=='__main__':main()
