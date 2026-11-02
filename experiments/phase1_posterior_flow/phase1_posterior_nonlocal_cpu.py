"""Fixed posterior nonlocal synthesis: source replay and Phase1 truth-last test."""
import argparse,hashlib,sys,time
from pathlib import Path
import numpy as np
import torch
from posterior_nonlocal_synthesis import PosteriorNonlocalSynthesis
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core,BlindActionNet,action_matrix,posterior,SOURCE,PRIOR_RUN
TRAIN='20261102-phase1-posterior-nonlocal-source-s2026092701-r02'
TRAIN_COMMIT='b1170e6ad3620c2c821bbcbcdca2f5a3482e3285'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_fixed(root):
 source=root/'runs'/TRAIN;m=core.read(source/'prediction_complete.json');checkpoint=source/'checkpoint.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!=TRAIN_COMMIT or state.get('commit')!=TRAIN_COMMIT or digest(checkpoint)!=m.get('checkpoint_sha256') or len(state['fit_ids'])!=8192 or m.get('source_fit_ids')!=8192 or m.get('DURA_loaded') is not False or m.get('F_frozen') is not True or any(m.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads','model_updates_after_prediction')):raise ValueError('Exact fixed source-only synthesis model required')
 net=PosteriorNonlocalSynthesis();net.load_state_dict(state['model'],strict=True);net.eval().requires_grad_(False)
 b=torch.load(root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p=torch.load(root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False)
 if b.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or b.get('epoch')!=10 or p.get('source_ids')!=8192:raise ValueError('Exact blind action and source prior required')
 blind=BlindActionNet();blind.load_state_dict(b['model'],strict=True);blind.eval().requires_grad_(False);stats=(p['means'],p['covariances'],p['priors']);fixed=torch.load(root/'runs'/s.GATE_RUN/'source/selected.pt',map_location='cpu',weights_only=False)
 if fixed.get('commit')!=s.GATE_COMMIT or fixed.get('selected',{}).get('kind')!='gated_direct_iq':raise ValueError('Exact fixed source gate required')
 gate=s.BlindGate();gate.load_state_dict(fixed['gate'],strict=True);gate.eval().requires_grad_(False);teacher,_=core.teacher(root,'cpu');teacher.eval().requires_grad_(False)
 return net,blind,stats,gate,teacher,torch.eye(512)

@torch.no_grad()
def predict_batch(x,models):
 net,blind,stats,gate,teacher,eye=models;raw=teacher.classify_features(teacher.features(x)).argmax(1);active=gate(x).sigmoid()>.8;out=x.clone();parent=x.clone()
 if active.any():
  signal=x[active];h,lv=blind(signal);action=action_matrix(h);states,lp=posterior(signal.reshape(len(signal),512),action,eye[None]*lv.exp()[:,None,None],*stats);base=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(signal);candidates=torch.cat((signal[:,None],states.reshape(-1,6,2,256)),1);parent[active]=base;out[active]=net(base,candidates,lv)
 pred=teacher.classify_features(teacher.features(out)).argmax(1);physical=teacher.classify_features(teacher.features(parent)).argmax(1)
 if not all(torch.isfinite(t).all() for t in (out,parent)):raise ValueError('Finite restored IQ required')
 return raw,pred,active,physical


@torch.no_grad()
def smoke(root,out,commit):
 if out.exists():raise FileExistsError('Immutable run output required')
 source=root/'runs'/TRAIN;m=core.read(source/'prediction_complete.json');path=source/'source_predictions.npz'
 if digest(path)!=m.get('prediction_sha256') or m.get('count')!=672:raise ValueError('Exact source predictions required')
 with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
 ids=data['indices'];rx=data['rx'];scene=data['scene'];positions=[]
 for group in ((4,6),(8,)):
  for view in core.VIEWS:
   ix=np.flatnonzero(np.isin(rx,group)&(scene==view))
   if len(ix)<8:raise ValueError('Eight source records per stratum required')
   positions.extend(ix[:8].tolist())
 pos=np.asarray(positions,dtype=np.int64);selected=ids[pos];cache,clean,allrx,training,pv,ph=s.source_contract(root);received,source_scene=s.epoch_data(cache,19)
 if not np.array_equal(allrx[selected],rx[pos]) or not np.array_equal(source_scene[selected],scene[pos]):raise ValueError('Exact source identities and scenes required')
 models=load_fixed(root);net,blind,stats,gate,teacher,eye=models;chunks=[]
 for start in range(0,len(selected),8):
  x=core.tensor(received[selected[start:start+8]],'cpu');h,lv=blind(x);A=action_matrix(h);states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],*stats);base=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x);candidates=torch.cat((x[:,None],states.reshape(-1,6,2,256)),1);wave=net(base,candidates,lv)
  choices=[teacher.classify_features(teacher.features(v)).argmax(1) for v in (x,base,wave)];chunks.append(torch.stack(choices,1).cpu())
 actual=np.from_dlpack(torch.cat(chunks).contiguous());expected=data['values'][pos][:,[2,4,6]].astype(np.int64);mismatches=int(np.count_nonzero(np.any(actual!=expected,axis=1)))
 if mismatches:raise ValueError(f'Fixed source replay mismatch: {mismatches}')
 out.mkdir(parents=True);core.write(out/'prediction_smoke.json',dict(status='PASS',source_only=True,records=len(pos),batch=8,threads=2,mismatches=mismatches,commit=commit,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_IQ_reads=0));print('POSTERIOR_NONLOCAL_SMOKE_PASS',len(pos),flush=True)

@torch.no_grad()
def predict(root,out,commit):
 check=core.read(out/'prediction_smoke.json')
 if check.get('status')!='PASS' or check.get('commit')!=commit or check.get('test_IQ_reads')!=0:raise ValueError('Source-only replay required')
 models=load_fixed(root);folder,ids=core.reference(root);reference=root/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction'
 if not np.array_equal(ids,np.load(reference/'physical_ids.npy',allow_pickle=False)):raise ValueError('Exact ground physical IDs required')
 dst=out/'flow/test_prediction';dst.mkdir(parents=True,exist_ok=False);np.save(dst/'physical_ids.npy',ids,allow_pickle=False);accepted={};tick=time.monotonic()
 for view in core.VIEWS:
  values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False);baseline=np.load(reference/(view+'.raw.npy'),mmap_mode='r',allow_pickle=False)
  if values.shape!=(len(ids),2,256) or values.dtype!=np.float32 or baseline.shape!=(len(ids),) or baseline.dtype!=np.int64:raise ValueError('Complete fixed ground arrays required')
  raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));count=0
  for start in range(0,len(ids),8):
   x=core.tensor(values[start:start+8],'cpu');a,b,on,_=predict_batch(x,models);a=np.from_dlpack(a.contiguous());b=np.from_dlpack(b.contiguous());end=start+len(x)
   if not np.array_equal(a,baseline[start:end]):raise ValueError(f'Raw F baseline mismatch: {view} {start}')
   raw[start:end]=a;restored[start:end]=b;count+=int(on.sum())
   if end%16384==0:print('POSTERIOR_NONLOCAL_PROGRESS',view,end,'seconds',round(time.monotonic()-tick,1),flush=True)
  raw.flush();restored.flush();del raw,restored;accepted[view]=count;print('POSTERIOR_NONLOCAL_VIEW_COMPLETE',view,len(ids),flush=True)
 core.write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,commit=commit,TX_truth_reads=0,model_updates=0,accepted_by_view=accepted));print('POSTERIOR_NONLOCAL_PREDICTIONS_COMPLETE',flush=True)

def score(root,out,commit):
 dst=out/'flow/test_prediction';m=core.read(dst/'complete.json');core.validate_prediction_manifest(m,count=168000,commit=commit)
 if m.get('TX_truth_reads')!=0 or m.get('model_updates')!=0:raise ValueError('Frozen predictions required before truth')
 for view in core.VIEWS:
  for arm in ('raw','restored'):
   x=np.load(dst/(view+'.'+arm+'.npy'),mmap_mode='r',allow_pickle=False)
   if x.shape!=(168000,) or x.dtype!=np.int64:raise ValueError('Complete prediction arrays required')
 core.score(root,out,commit);print('POSTERIOR_NONLOCAL_TRUTH_SCORE_COMPLETE',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('smoke','predict','score'));ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cpu');a=ap.parse_args()
 if a.device!='cpu':raise ValueError('CPU-only runner required')
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.mode=='smoke':smoke(a.root,a.out,a.commit)
 elif a.mode=='predict':predict(a.root,a.out,a.commit)
 else:score(a.root,a.out,a.commit)
if __name__=='__main__':main()
