"""Truth-last Phase1 test of fixed gate-aware candidate IQ selection."""
import argparse,hashlib,sys,time
from pathlib import Path
import numpy as np
import torch
from gate_aware_router import GateAwareRouter,select_waveforms
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_prior_pnp'))
from pipeline_unroll import load_fixed as load_recovery,evaluate_wave,s,core
SOURCE='20261102-phase1-gate-aware-router-source-s2026092701-r01';SOURCE_COMMIT='9dac8eaa0a35e352652791bf0080ade59751af09';PRIOR_SHA='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5';NAMES=['raw','physical','unroll025','unroll05','unroll1']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_fixed(root):
 folder=root/'runs'/SOURCE;m=core.read(folder/'result.json');path=folder/'source_predictions.npz';ckpath=folder/'train46_eval8.pt'
 if m.get('status')!='SOURCE_GATE_ROUTER_TRAINING_COMPLETE' or m.get('commit')!=SOURCE_COMMIT or m.get('count')!=11761 or m.get('candidate_names')!=NAMES or m.get('prior_sha256')!=PRIOR_SHA or m.get('source_output_sha256')!=sha(path) or m.get('DURA_in_candidate') is not False or any(m.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')) or m.get('fold_model_sha256',{}).get('train46_eval8')!=sha(ckpath):raise ValueError('Exact frozen source router required')
 ck=torch.load(ckpath,map_location='cpu',weights_only=False)
 if ck.get('commit')!=SOURCE_COMMIT or ck.get('candidate_names')!=NAMES or ck.get('prior_sha256')!=PRIOR_SHA or ck.get('fold',{}).get('train_receivers')!=[4,6] or ck.get('fold',{}).get('eval_receivers')!=[8]:raise ValueError('Exact source receiver-fold model required')
 router=GateAwareRouter();router.load_state_dict(ck['model'],strict=True);router.eval().requires_grad_(False);recovery=load_recovery(root,'cpu');teacher,_=core.teacher(root,'cpu');teacher.eval().requires_grad_(False);return router,recovery,teacher

@torch.no_grad()
def predict_batch(x,models):
 router,recovery,teacher=models;prob=recovery[3](x).sigmoid();parent,full,active=evaluate_wave(x,recovery);delta=full-parent;waves=torch.stack((x,parent,parent+.25*delta,parent+.5*delta,full),1)
 if not torch.equal(active,prob>.8) or not torch.isfinite(waves).all() or not torch.isfinite(prob).all():raise ValueError('Finite IQ candidates and common gate required')
 pred=torch.stack([teacher.classify_features(teacher.features(waves[:,i])).argmax(1) for i in range(5)],1);choice,decision=router.choose(pred,prob);selected=select_waveforms(waves,choice);actual=teacher.classify_features(teacher.features(selected)).argmax(1)
 if not torch.equal(decision,actual):raise ValueError('Selected actual IQ waveform decision mismatch')
 return pred[:,0],actual,active,pred,prob,choice

def smoke(root,out,commit):
 if out.exists():raise FileExistsError('Immutable run output required')
 folder=root/'runs'/SOURCE
 with np.load(folder/'source_predictions.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
 rx=data['rx'];scene=data['scene'];positions=[]
 for view in core.VIEWS:
  ix=np.flatnonzero((rx==8)&(scene==view))
  if len(ix)<8:raise ValueError('Eight heldout source records per scene required')
  positions.extend(ix[:8].tolist())
 pos=np.asarray(positions,dtype=np.int64);ids=data['indices'][pos];cache,clean,allrx,training,pv,ph=s.source_contract(root);received,source_scene=s.epoch_data(cache,19)
 if len(pos)!=56 or not np.array_equal(allrx[ids],rx[pos]) or not np.array_equal(source_scene[ids],scene[pos]):raise ValueError('Source replay IDs and views required')
 models=load_fixed(root);outputs=[]
 for size in (8,32):
  chunks=[];prob=[]
  for start in range(0,len(ids),size):
   ix=ids[start:start+size];raw,pred,on,candidates,gate,choice=predict_batch(core.tensor(received[ix],'cpu'),models);chunks.append(torch.cat((candidates,choice[:,None],pred[:,None],on.long()[:,None]),1).cpu());prob.append(gate.cpu())
  outputs.append((np.from_dlpack(torch.cat(chunks).contiguous()),np.from_dlpack(torch.cat(prob).contiguous())))
 expected=np.column_stack((data['candidate_predictions'][pos],data['choice'][pos],data['prediction'][pos],data['prior_values'][pos,3]>.8)).astype(np.int64);baseline_gate=data['prior_values'][pos,3]
 if not np.array_equal(outputs[0][0],expected) or not np.array_equal(outputs[1][0],expected) or not np.allclose(outputs[0][1],baseline_gate,rtol=1e-5,atol=1e-6) or not np.allclose(outputs[1][1],baseline_gate,rtol=1e-5,atol=1e-6):raise ValueError('Fixed heldout source candidate and router replay mismatch')
 out.mkdir(parents=True);core.write(out/'prediction_smoke.json',dict(status='PASS',source_only=True,records=len(pos),batches=[8,32],candidate_and_route_mismatches=0,commit=commit,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_IQ_reads=0));print('GATE_ROUTER_CPU_SOURCE_SMOKE_PASS',len(pos),flush=True)

@torch.no_grad()
def predict(root,out,commit):
 check=core.read(out/'prediction_smoke.json')
 if check.get('status')!='PASS' or check.get('commit')!=commit or check.get('test_IQ_reads')!=0:raise ValueError('Heldout source replay required')
 models=load_fixed(root);folder,ids=core.reference(root);reference=root/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction'
 if not np.array_equal(ids,np.load(reference/'physical_ids.npy',allow_pickle=False)):raise ValueError('Exact ground physical IDs required')
 dst=out/'flow/test_prediction';dst.mkdir(parents=True,exist_ok=False);np.save(dst/'physical_ids.npy',ids,allow_pickle=False);accepted={};tick=time.monotonic()
 for view in core.VIEWS:
  values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False);baseline=np.load(reference/(view+'.raw.npy'),mmap_mode='r',allow_pickle=False)
  if values.shape!=(len(ids),2,256) or values.dtype!=np.float32 or baseline.shape!=(len(ids),) or baseline.dtype!=np.int64:raise ValueError('Complete fixed ground arrays required')
  raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));candidates=np.lib.format.open_memmap(dst/(view+'.candidates.npy'),mode='w+',dtype=np.int64,shape=(len(ids),5));gate=np.lib.format.open_memmap(dst/(view+'.gate.npy'),mode='w+',dtype=np.float32,shape=(len(ids),));route=np.lib.format.open_memmap(dst/(view+'.route.npy'),mode='w+',dtype=np.uint8,shape=(len(ids),));count=0
  for start in range(0,len(ids),32):
   x=core.tensor(values[start:start+32],'cpu');a,b,on,c,g,choice=predict_batch(x,models);a=np.from_dlpack(a.contiguous());b=np.from_dlpack(b.contiguous());c=np.from_dlpack(c.contiguous());g=np.from_dlpack(g.contiguous());choice=np.from_dlpack(choice.to(torch.uint8).contiguous());end=start+len(x)
   if not np.array_equal(a,baseline[start:end]):raise ValueError(f'Raw F baseline mismatch: {view} {start}')
   raw[start:end]=a;restored[start:end]=b;candidates[start:end]=c;gate[start:end]=g;route[start:end]=choice;count+=int(on.sum())
   if end%16384==0:print('GATE_ROUTER_CPU_PROGRESS',view,end,'seconds',round(time.monotonic()-tick,1),flush=True)
  for arr in (raw,restored,candidates,gate,route):arr.flush()
  del raw,restored,candidates,gate,route;accepted[view]=count;print('GATE_ROUTER_CPU_VIEW_COMPLETE',view,len(ids),flush=True)
 core.write(dst/'pending_complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,commit=commit,TX_truth_reads=0,model_updates=0,accepted_by_view=accepted));print('GATE_ROUTER_CPU_PREDICTIONS_PENDING_AUDIT',flush=True)

def score(root,out,commit):
 dst=out/'flow/test_prediction';m=core.read(dst/'complete.json');core.validate_prediction_manifest(m,count=168000,commit=commit)
 if m.get('TX_truth_reads')!=0 or m.get('model_updates')!=0:raise ValueError('Frozen predictions required before truth')
 for view in core.VIEWS:
  for arm in ('raw','restored'):
   x=np.load(dst/(view+'.'+arm+'.npy'),mmap_mode='r',allow_pickle=False)
   if x.shape!=(168000,) or x.dtype!=np.int64:raise ValueError('Complete fixed prediction arrays required')
 core.score(root,out,commit);print('GATE_ROUTER_CPU_TRUTH_SCORE_COMPLETE',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('smoke','predict','score'));ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cpu');a=ap.parse_args()
 if a.device!='cpu':raise ValueError('CPU-only runner required')
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.mode=='smoke':smoke(a.root,a.out,a.commit)
 elif a.mode=='predict':predict(a.root,a.out,a.commit)
 else:score(a.root,a.out,a.commit)
if __name__=='__main__':main()
