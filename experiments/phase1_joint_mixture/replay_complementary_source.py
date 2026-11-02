"""Save fixed source predictions for three complementary DURA-free restorers."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_prior_pnp'))
import phase1_attention as attention
import phase1_structured as structured
import phase1_pnp as white
from attention_restorer import AttentionRestorer
from source_pilot import s,core
LS='20261102-phase1-free-full-hard-labeled-source-s2026092701-r01'

def attention_fixed(root,device):
 train=root/'runs/20261102-phase1-attention-waveform-pilot-s2026092701-r02';result=core.read(train/'result.json');state=torch.load(train/'checkpoint.pt',map_location='cpu',weights_only=False);fixed=torch.load(root/'runs'/s.GATE_RUN/'source/selected.pt',map_location='cpu',weights_only=False)
 if result.get('status')!='TRAINING_COMPLETE' or state.get('commit')!='52df24a8215001bbd3e2e3bba158396ac33e270d' or fixed.get('commit')!=s.GATE_COMMIT or fixed.get('selected',{}).get('kind')!='gated_direct_iq':raise ValueError('Exact independent attention and gate required')
 net=AttentionRestorer().to(device);net.load_state_dict(state['model']);net.eval().requires_grad_(False);gate=s.BlindGate().to(device);gate.load_state_dict(fixed['gate'],strict=True);gate.eval().requires_grad_(False);teacher,_=core.teacher(root,device);teacher.eval().requires_grad_(False);return net,gate,teacher

def load(name,root):
 return attention_fixed(root,'cpu') if name=='attention' else structured.load_fixed(root,'cpu') if name=='structured_prior' else white.load_fixed(root,'cpu')

def predict(name,x,models):
 return attention.predict_batch(x,models) if name=='attention' else structured.predict_batch(x,models) if name=='structured_prior' else white.predict_batch(x,models)

def run(name,models,received,ids,batch,threads):
 torch.set_num_threads(threads);out=[]
 with torch.no_grad():
  for start in range(0,len(ids),batch):
   ix=ids[start:start+batch];raw,pred,active=predict(name,core.tensor(received[ix],'cpu'),models);out.append(torch.stack((raw,pred,active.long()),1).cpu())
 return np.from_dlpack(torch.cat(out).contiguous())

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;refroot=a.root/'runs'/LS;manifest=core.read(refroot/'prediction_complete.json');path=refroot/'predictions.npz'
 if manifest.get('status')!='SOURCE_PREDICTIONS_FIXED' or manifest.get('count')!=1047 or manifest.get('role')!='L_s' or manifest.get('TX_truth_reads')!=0 or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest.get('predictions_sha256'):raise ValueError('Exact source L_s prediction cohort required')
 with np.load(path,allow_pickle=False) as z:ref={k:z[k].copy() for k in ('indices','physical_ids','rx','scene','part','predictions','active')}
 ids=ref['indices'];cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);received,scene=s.epoch_data(cache,19)
 if len(ids)!=1047 or np.any(roles[ids]!='L_s') or not np.array_equal(physical[ids],ref['physical_ids']) or not np.array_equal(rx[ids],ref['rx']) or not np.array_equal(scene[ids],ref['scene']) or np.intersect1d(ids,training).size or not np.all(np.isin(ids,np.r_[validation,heldout])):raise ValueError('Exact source heldout L_s binding required')
 for run in ('20261102-phase1-attention-waveform-pilot-s2026092701-r02','20261102-phase1-structured-iq-prior-s2026092701-r01','20261102-phase1-independent-iq-prior-s2026092701-r01'):
  state=torch.load(a.root/'runs'/run/'checkpoint.pt',map_location='cpu',weights_only=False)
  if np.intersect1d(ids,state['fit_ids']).size:raise ValueError('Source replay must be disjoint from each restoration fit')
 names=['attention','structured_prior','white_prior'];target=ids[:64] if a.smoke else ids;out=[];tick=time.monotonic()
 for name in names:
  torch.set_num_threads(2);models=load(name,a.root)
  if a.smoke:
   v1=run(name,models,received,target,8,2);v2=run(name,models,received,target,32,2);v3=run(name,models,received,target,32,8)
   if not np.array_equal(v1,v2) or not np.array_equal(v2,v3):raise ValueError('Source batch and thread decision equivalence required: '+name)
   result=v3
  else:result=run(name,models,received,target,32,8)
  reference=np.column_stack((ref['predictions'][:len(target),1],ref['active'][:len(target)]))
  if not np.array_equal(result[:,[0,2]],reference):raise ValueError('Common raw F and gate decisions required: '+name)
  out.append(result[:,1]);print('COMPLEMENT_SOURCE_DONE',name,len(target),'seconds',round(time.monotonic()-tick,1),flush=True)
 if a.smoke:
  core.write(a.out/'result.json',dict(status='PASS',count=len(target),candidate_names=names,batch_thread_decisions_match=True,raw_gate_match=True,source_only=True,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0));return
 path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,physical_ids=ref['physical_ids'],rx=ref['rx'],scene=ref['scene'],part=ref['part'],raw=ref['predictions'][:,1],active=ref['active'],candidate_predictions=np.stack(out,axis=1))
 result=dict(status='SOURCE_PREDICTIONS_FIXED',commit=a.commit,count=len(ids),candidate_names=names,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),common_reference_sha256=manifest['predictions_sha256'],source_only=True,DURA_in_candidate=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates=0);core.write(a.out/'prediction_complete.json',result);print('COMPLEMENT_SOURCE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
