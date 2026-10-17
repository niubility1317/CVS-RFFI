import gc,os,sys,time,traceback
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .core import paired,stream,decisions
from .metrics import summarize
from experiments.cvs_noise_shape_factorial43 import design as ref,model as models

def load_L():
 expected=d.read(d.SOURCE);packed=torch.load(d.CACHE,map_location='cpu',weights_only=False)
 if any(packed['contract'].get(k)!=v for k,v in expected.items()) or packed['contract'].get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Unchanged source carrier contract mismatch')
 L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
 if L['x'].shape!=(6300,2,256) or not torch.isfinite(L['x']).all() or len(set(L['ids']))!=6300 or set(L['ids'])!=set(expected['role_ids']['L_s']) or set(L['ids'])&(set(expected['role_ids']['U_s'])|set(expected['role_ids']['V'])) or set(L['rx'].tolist())!=set(d.RX) or set(L['y'].tolist())!=set(range(6)):raise ValueError('Only exact L6300/all5RX permitted')
 if any(int(((L['y']==t)&(L['rx']==r)&(L['day']==day)).sum())!=70 for t in range(6) for r in d.RX for day in (1,2,3)):raise ValueError('All90 source cells require70')
 return L

def main():
 if d.BASE.exists():raise FileExistsError('Immutable output root exists')
 d.BASE.mkdir();start=time.monotonic();d.validate(d.config());torch.set_num_threads(2);device=torch.device('cuda:0')
 from experiments.cvs_receiver_residual_capacity4.control import lock
 from experiments.cvs_multi_state_action.dispatch import capacity
 gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
 with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
  if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner GPU capacity exceeded')
 try:
  from experiments.cvs_equivariant_identity.precision import numerical_context
  from SSDG import train_ssdg as native
  import DataAugmentation as augmodule
  ms=[];snap=[];configs=[]
  for arm in d.ARMS:
   c=ref.config(next(r for r in ref.rows() if r['arm']==arm));m=models.own_model(c,device);ss={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
   with torch.no_grad():z=decisions(m,torch.zeros(2,2,256,device=device))
   if z.shape!=(2,6) or any(not torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in ss.items()):raise ValueError('Real checkpoint no-query smoke/state failed')
   ms.append(m);snap.append(ss);configs.append(c)
  d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',NN=4,real_IQ=0,target_query=False,head_fit_updates=0,optimizer_updates=0,models=list(d.ARMS)))
  print('REAL_CP_NO_QUERY_SMOKE PASS4NN;continue exactsource',flush=True)
  L=load_L();a=ref.make_args(configs[0],'cuda:0');base=native.build_aug_base_cfg(a);saved=d.read(d.REFERENCE/'B00-s2026092701/source/ordinary_augmentation_actual.json')
  if base!=saved['base'] or saved['schedule_frozen_at']!=13 or not a.aug_enable_pa_normal:raise ValueError('Actual ordinary recipe differs')
  with stream(d.DRAW_SEED,0,device):aug=native.make_augmentor(base)
  schedule=native.configure_augmentor_for_epoch(aug,base,13,a)
  ranges={k:getattr(aug,k) for k in base if hasattr(aug,k) and isinstance(getattr(aug,k),(bool,int,float))}
  d.write(d.BASE/'resolved_config.json',dict(config=d.config(),actual_ordinary_base=base,actual_E13=ranges,schedule_state=schedule,module_files={k:str(v.__file__) for k,v in sys.modules.items() if k in ('DataAugmentation','SSDG.train_ssdg','leo_practical.channel','leo_practical.residual')},frozen_source_configs=configs,source_roles='EXACT_MATCH',U_TX_hidden=True,cache_other_source_roles_deserialized_not_used=True,checkpoint_contract='VERIFIED same allsource scratch20+own10/fixedfinal30/ownEMA;no targetselection',CWD=os.getcwd(),GPU=gpu,python=sys.executable,code_commit=(d.ROOT/'release_commit.txt').read_text().strip()))
  raw=np.empty((2,6300,6),np.float32);full=np.empty((2,2,6300,6),np.float32);bypass=np.empty_like(full);trigger=np.zeros((2,6300),bool);mse=np.zeros((2,6300),np.float64);window_points=np.zeros((2,6300),np.int64)
  idx=torch.randperm(6300,generator=torch.Generator().manual_seed(d.ORDER_SEED));nn=4;batches=0
  with numerical_context(ref.FULL_FP32_POLICY),torch.no_grad():
   for chunk in idx.split(128):
    x=L['x'][chunk].to(device)
    for mi,m in enumerate(ms):raw[mi,chunk.tolist()]=decisions(m,x).tolist();nn+=len(x)
   for draw in range(2):
    for bi,chunk in enumerate(idx.split(128)):
     x=L['x'][chunk].to(device);y=L['y'][chunk].long().to(device);f,b,meta=paired(augmodule,aug,x,y,d.DRAW_SEED,draw*1000+bi+1);ix=chunk.tolist();trigger[draw,ix]=meta['triggered'];window_points[draw,ix]=meta['window_points_below_one'];mse[draw,ix]=(f-b).square().flatten(1).mean(1).tolist()
     for mi,m in enumerate(ms):full[mi,draw,ix]=decisions(m,f).tolist();bypass[mi,draw,ix]=decisions(m,b).tolist();nn+=2*len(x)
     batches+=1
    d.write(d.BASE/'progress.json',dict(draws_completed=draw+1,NN=nn,paired_batches=batches,allsource_RX=list(d.RX),target_access=False))
    print('PROGRESS draw='+str(draw+1)+'/2 NN='+str(nn),flush=True)
  if nn!=63004 or batches!=100 or any(m.id_backbone.encoder._full_eval_packets!=31502 for m in ms):raise ValueError('Frozen diagnostic budget mismatch')
  for m,ss in zip(ms,snap):
   if m.training or any(p.requires_grad or p.grad is not None for p in m.parameters()) or any(not torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in ss.items()):raise ValueError('Frozen checkpoint state changed')
  y=np.asarray(L['y'].tolist(),np.int64);rx=np.asarray(L['rx'].tolist(),np.int64);result=summarize(raw,full,bypass,y,rx,trigger)
  result.update(wall_seconds=time.monotonic()-start,ordinary_pairs=12600,physical_unique=6300,per_draw_triggered=trigger.sum(-1).tolist(),unchanged_nontrigger_IQ=True,window_restored=True,RNG_restored=True,default_head_unchanged=True,states_unchanged=True)
  np.savez_compressed(d.BASE/'source_predictions.npz',raw=raw,full=full,bypass=bypass,y=y,rx=rx,day=L['day'].tolist(),ids=np.asarray(L['ids']),trigger=trigger,mse=mse,window_points=window_points)
  d.write(d.BASE/'result.json',result);d.write(d.BASE/'completion.json',dict(status='SOURCE_ANALYZED',decision=result['decision'],source_qualified=result['source_qualified'],NN=nn,IQ_unique=6300,training_updates=0,head_fit_updates=0,target_access=False,wall_seconds=result['wall_seconds']))
  print(result['decision'],flush=True)
 except Exception as exc:
  d.write(d.BASE/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),preserve_outputs=True));raise
