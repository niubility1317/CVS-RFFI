"""Read-only actual default-head gradients on paired legal L observations."""
import os,sys,time,json,traceback
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .tape import observe,FEATURES,PARAMETERS
from .maths import OBJECTIVES

def stack(parts):
 first=parts[0]
 if torch.is_tensor(first):return torch.cat(parts,dim=0)
 if isinstance(first,dict):return {k:stack([p[k] for p in parts]) for k in first}
 raise TypeError('Only actual recorded tensor dictionaries may be stacked')
def readout(m,x,y,path):
 chunks=[];calls=0
 for i in range(0,len(x),d.BATCH):
  z=observe(m,x[i:i+d.BATCH],y[i:i+d.BATCH]);calls+=z.pop('autograd_calls');z.pop('smoke_check');z.pop('local_activation_leaves');chunks.append(z)
 result=stack(chunks)
 if result['logits'].shape!=(900,6) or any(p.grad is not None for p in m.parameters()):raise ValueError('Read-only actual observations required')
 torch.save(result,path);return calls

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source diagnostic')
 torch.set_num_threads(2);tic=time.perf_counter();d.write(d.BASE/'resolved_config.json',d.config());d.write(d.BASE/'runtime.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,CUDA_VISIBLE_DEVICES='',threads=2))
 from experiments.cvs_Uurban39 import model as backend,design as old
 from experiments.cvs_equivariant_identity.precision import numerical_context
 with numerical_context(old.FULL_FP32_POLICY):
  c=old.config(next(r for r in old.rows() if r['arm']=='B0'));m=backend.own_model(c,'cpu');head=m.id_backbone.encoder.core.id_backbone.cls_head
  if sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206 or head.weight.shape!=(6,160) or float(head.scale)!=30.:raise ValueError('Actual F/defaultcos30/joint160 only')
  states={k:v.clone() for k,v in m.state_dict().items()};flags=[p.requires_grad for p in m.parameters()]
  g=torch.Generator().manual_seed(18460);smoke=observe(m,torch.randn(2,2,256,generator=g),torch.tensor([0,1]),smoke=True);autograd=smoke['autograd_calls']
  if any(not torch.equal(v,m.state_dict()[k]) for k,v in states.items()) or any(p.grad is not None for p in m.parameters()) or [p.requires_grad for p in m.parameters()]!=flags:raise ValueError('Real checkpoint smoke mutated state/gradient flags')
  d.write(d.BASE/'checkpoint_noquery_smoke.json',dict(status='PASS',NN=2,source_IQ_before_smoke=0,target_IQ=0,optimizer_updates=0,head_fit_updates=0,objective_checks=smoke['smoke_check'],checkpoint=str(d.PARENT/'B0-s2026092701/source/final_ssdg.pth'),model_state_unchanged=True,contract='EXACT_MATCH;own20scratch+own10;fixed B0 before target results'))
  print('SMOKE PASS actual checkpoint/default W and gates;2syntheticNN;0query',flush=True)
  from .panel import load
  data,roles=load();d.write(d.BASE/'data_roles.json',roles);torch.save({k:v for k,v in data.items() if k!='x'},d.BASE/'source_labels_ids.pt')
  autograd+=readout(m,data['x'],data['y'],d.BASE/'clean.pt');print('clean L900 gradients complete',flush=True)
  from .controls import Generator
  generator=Generator();raw=np.asarray(data['x'].tolist(),dtype=np.float64)
  for draw in range(2):
   for si,scene in enumerate(d.SCENES):
    row=d.BASE/f'draw{draw}'/scene;row.mkdir(parents=True,exist_ok=False);iq=np.empty((3,900,2,256),np.float32);records=[]
    for i in range(900):
     v,r=generator.apply(raw[i,0]+1j*raw[i,1],data['ids'][i],int(data['rx'][i]),int(data['day'][i]),draw,scene)
     for ci,name in enumerate(d.CONDITIONS):iq[ci,i]=np.stack((v[name].real,v[name].imag))
     records.append(r)
    np.save(row/'observations.npy',iq,allow_pickle=False);d.write(row/'physics.json',records)
    for ci,name in enumerate(d.CONDITIONS):
     a=np.ascontiguousarray(iq[ci]);x=torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape);autograd+=readout(m,x,data['y'],row/(name.replace('/','_')+'.pt'))
     if any(not torch.equal(v,m.state_dict()[k]) for k,v in states.items()) or any(p.grad is not None for p in m.parameters()) or [p.requires_grad for p in m.parameters()]!=flags:raise ValueError('Frozen source diagnostic changed model state')
     d.write(d.BASE/'progress.json',dict(draw=draw,scene=scene,condition=name,NN=m.id_backbone.encoder._full_eval_packets,autograd_calls=autograd,training_updates=0,target_access=False));print(draw,scene,name,'paired gradient readout complete',flush=True)
  audit=generator.audit();d.write(d.BASE/'actual_channel.json',audit)
  if generator.native_calls!=3600 or generator.render_calls!=14400 or len(generator.sessions)!=15 or m.id_backbone.encoder._full_eval_packets!=d.EXPECTED_NN or m.id_backbone.encoder._full_train_packets or autograd!=d.EXPECTED_AUTOGRAD:raise ValueError('Actual source NN/channel/autograd budget differs')
  d.write(d.BASE/'observation_complete.json',dict(status='SOURCE_GRADIENT_OBSERVATIONS_COMPLETE',NN=d.EXPECTED_NN,autograd_calls=autograd,unique_L=900,original_L=6300,source_only=True,V_forward=0,V_backprop=0,U_truth_read=False,target_access=False,training_updates=0,head_fit_updates=0,state_unchanged=True,default_head_unchanged=True,actual_gate_parameters=True,model_seed_count=1,channel_draws=2,readout_wall_seconds=time.perf_counter()-tic,code_commit=(d.ROOT/'release_commit.txt').read_text().strip()))
 from .analyze import collect
 result=collect(d.BASE)
 d.write(d.BASE/'completion.json',dict(status='ANALYZED_SOURCE_ONLY',NN=d.EXPECTED_NN,autograd_calls=autograd,training_updates=0,head_fit_updates=0,target_access=False,V_backprop=False,U_truth_read=False,source_decision=result['source_decision'],wall_seconds=time.perf_counter()-tic,goal_complete=False))
 print('ANALYZED_SOURCE_ONLY 11702NN/0updates/0target',flush=True)
def main():
 owned=False
 try:
  d.BASE.mkdir(exist_ok=False);owned=True;run()
 except BaseException as e:
  if owned:d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),preserve_outputs=True,no_automatic_retry=True))
  raise
