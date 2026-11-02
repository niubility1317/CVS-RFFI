"""CPU Phase1 test of fixed learned-action recovery with raw-F consistency protection."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from pipeline_learned_action import load_fixed as load_recovery,evaluate_wave,s,core
LS='20261102-phase1-free-full-hard-labeled-source-s2026092701-r01'

def require_raw_match(actual,expected,view,start):
 if actual.shape!=expected.shape:raise ValueError('Raw F reference shape mismatch')
 different=np.flatnonzero(actual!=expected)
 if len(different):raise ValueError(f'CPU raw F differs from fixed baseline: view={view}, index={start+int(different[0])}, batch_mismatches={len(different)}')

def load_fixed(root,device):
 recovery=load_recovery(root,device);teacher,_=core.teacher(root,device);teacher.eval().requires_grad_(False);return recovery,teacher

@torch.no_grad()
def predict_batch(x,models):
 recovery,teacher=models;raw_logits=teacher.classify_features(teacher.features(x));raw=raw_logits.argmax(1);parent,wave,active=evaluate_wave(x,recovery);out=teacher.classify_features(teacher.features(wave));pred=out.argmax(1)
 if not all(torch.isfinite(z).all() for z in (raw_logits,parent,wave,out)):raise ValueError('Finite fixed predictions required')
 return raw,pred,active

def smoke(root,out,commit,device):
 if device!='cpu':raise ValueError('CPU-only runner required')
 if out.exists():raise FileExistsError('Immutable output required')
 out.mkdir(parents=True);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;models=load_fixed(root,device)
 with np.load(root/'runs'/LS/'predictions.npz',allow_pickle=False) as p:data={k:p[k] for k in p.files}
 cache,clean,rx,training,pv,ph=s.source_contract(root);received,scene=s.epoch_data(cache,19);ids=data['indices'];physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
 if len(ids)!=1047 or not np.array_equal(physical[ids],data['physical_ids']):raise ValueError('Fixed source replay cohort required')
 batches=[]
 for size in (8,32):
  chunks=[]
  for start in range(0,len(ids),size):
   ix=ids[start:start+size];raw,pred,active=predict_batch(core.tensor(received[ix],device),models);chunks.append(torch.stack((raw,pred,active.long()),1).cpu())
  batches.append(np.from_dlpack(torch.cat(chunks).contiguous()))
 common=np.column_stack((data['predictions'][:,1],data['active']));mismatch=int(np.count_nonzero(np.any(batches[0][:,[0,2]]!=common,axis=1)));batch_diff=int(np.count_nonzero(np.any(batches[0]!=batches[1],axis=1)))
 if mismatch or batch_diff:raise ValueError(f'Source replay mismatches: common={mismatch}, batch={batch_diff}')
 core.write(out/'prediction_smoke.json',dict(status='PASS',source_only=True,records=len(ids),batch=32,common_mismatches=mismatch,batch_mismatches=batch_diff,commit=commit,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_IQ_reads=0));print('LEARNED_ACTION_CPU_PHASE1_SMOKE PASS',len(ids),flush=True)

@torch.no_grad()
def predict(root,out,commit,device):
 if device!='cpu':raise ValueError('CPU-only runner required')
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;check=core.read(out/'prediction_smoke.json')
 if check.get('status')!='PASS' or check.get('commit')!=commit or check.get('test_IQ_reads')!=0:raise ValueError('Source-only smoke required')
 models=load_fixed(root,device);folder,ids=core.reference(root);reference=root/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction'
 if not np.array_equal(ids,np.load(reference/'physical_ids.npy',allow_pickle=False)):raise ValueError('Fixed baseline physical IDs required')
 dst=out/'flow'/'test_prediction';dst.mkdir(parents=True,exist_ok=False);np.save(dst/'physical_ids.npy',ids,allow_pickle=False);accepted={};tick=time.monotonic()
 for view in core.VIEWS:
  values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False);reference_raw=np.load(reference/(view+'.raw.npy'),mmap_mode='r',allow_pickle=False)
  if reference_raw.shape!=(len(ids),) or reference_raw.dtype!=np.int64:raise ValueError('Complete raw F baseline required')
  if values.shape!=(len(ids),2,256) or values.dtype!=np.float32:raise ValueError('Complete Phase1 view required')
  raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));count=0
  for start in range(0,len(ids),32):
   x=core.tensor(values[start:start+32],device);a,b,on=predict_batch(x,models);a_np=np.from_dlpack(a.cpu().contiguous());require_raw_match(a_np,reference_raw[start:start+len(x)],view,start);raw[start:start+len(x)]=a_np;restored[start:start+len(x)]=np.from_dlpack(b.cpu().contiguous());count+=int(on.sum().item())
   if (start+len(x))%16384==0:print('LEARNED_ACTION_CPU_PHASE1_PROGRESS',view,start+len(x),'seconds',round(time.monotonic()-tick,1),flush=True)
  raw.flush();restored.flush();del raw,restored;accepted[view]=count;print('LEARNED_ACTION_CPU_PHASE1_VIEW_COMPLETE',view,len(ids),'active',count,'seconds',round(time.monotonic()-tick,1),flush=True)
 core.write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,commit=commit,TX_truth_reads=0,model_updates=0,accepted_by_view=accepted));print('LEARNED_ACTION_CPU_PHASE1_PREDICTIONS_COMPLETE',len(ids),flush=True)

def score(root,out,commit):
 dst=out/'flow'/'test_prediction';done=core.read(dst/'complete.json');core.validate_prediction_manifest(done,count=168000,commit=commit)
 if done.get('TX_truth_reads')!=0 or done.get('model_updates')!=0:raise ValueError('Frozen complete predictions required before truth')
 for view in core.VIEWS:
  for arm in ('raw','restored'):
   x=np.load(dst/(view+'.'+arm+'.npy'),mmap_mode='r',allow_pickle=False)
   if x.shape!=(168000,) or x.dtype!=np.int64:raise ValueError('Complete fixed prediction arrays required')
 core.score(root,out,commit);print('LEARNED_ACTION_CPU_PHASE1_TRUTH_SCORE_COMPLETE',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
 if a.mode=='smoke':smoke(a.root,a.out,a.commit,a.device)
 elif a.mode=='predict':predict(a.root,a.out,a.commit,a.device)
 else:score(a.root,a.out,a.commit)
if __name__=='__main__':main()
