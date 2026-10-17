"""Reuse source full logits; evaluate only ten controlled observations on frozen models."""
import os,time
from collections import Counter
import numpy as np
from . import design as d

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU2 only, no GPU allocation')
 import torch
 from experiments.cvs_cycleorder_factorial32 import model as api,design as prior
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from .stats import collect
 out=d.BASE;out.mkdir(exist_ok=False);tic=time.perf_counter();torch.set_num_threads(2);models={};states={};configs={}
 d.write(out/'resolved_config.json',d.config());d.write(out/'runtime.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=__import__('sys').argv,CUDA_VISIBLE_DEVICES='',threads=2))
 with numerical_context(d.FP32):
  # First data-consuming action: own real CP provenance/load + synthetic no-query smoke.
  for arm in d.ARMS:
   c=prior.config(next(r for r in prior.rows() if r['arm']==arm));source=d.PARENT/(arm+'-s2026092701')/'source'
   if c['output_root']!=str(source) or c['source_roles']!=dict(L_s=6300,U_s=56700,V=27000) or c['held_RX'] is not None or d.read(source/'completion.json')['head_fit_updates']!=0:raise ValueError('Matched full-source zero-fit FINAL30 only')
   m=api.own_model(c,'cpu');head=m.id_backbone.encoder.core.id_backbone.cls_head
   if sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206 or head.weight.shape!=(6,160) or float(head.scale)!=30.:raise ValueError('Actual F/joint160/defaultcos30 differs')
   with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real FINAL30 no-query smoke failed')
   models[arm]=m;states[arm]={k:v.clone() for k,v in m.state_dict().items()};configs[arm]=c
  d.write(out/'checkpoint_noquery_smoke.json',dict(status='PASS',NN=4,source_IQ_before_smoke=0,target_IQ=0,head_fit_updates=0,training_updates=0,checkpoints={a:dict(path=str(d.PARENT/(a+'-s2026092701/source/final_ssdg.pth')),kind='model',epoch=30,updates=6660,scratch_only=True,role_handles='EXACT_MATCH',selection='FINAL30') for a in d.ARMS}))
  print('SMOKE PASS two real FINAL30/default heads,4synthetic,0query',flush=True)
  roles=d.read(d.PANEL/'data_roles.json');obs=d.read(d.PANEL/'observations/completion.json');d.validate_metadata(roles,obs);contract=d.read(d.SOURCE)
  ids=roles['physical_ids']
  if len(ids)!=d.N or len(set(ids))!=d.N or not set(ids)<=set(contract['role_ids']['V']):raise ValueError('Only exact original V physical IDs')
  d.write(out/'data_roles.json',roles);inputs={};logits={}
  for arm in d.ARMS:
   qroot=d.PARENT/(arm+'-s2026092701')/'source_qualification';q=d.read(qroot/'complete.json')
   if q['status']!='SOURCE_QUALIFIED_READOUT' or q['config']!=configs[arm] or q['NN']!=12602 or q['training_updates'] or q['head_fit_updates'] or q['target_access'] or q['formal_scores_read'] or not q['state_unchanged']:raise ValueError('Own frozen cached full/clean source readout differs')
   with np.load(qroot/'predictions.npz',allow_pickle=False) as f:inp={k:f[k].copy() for k in ('ids','TX','RX','day','clean','satellite')}
   if inp['ids'].tolist()!=ids or inp['clean'].shape!=(d.N,6) or inp['satellite'].shape!=(2,3,d.N,6) or not np.isfinite(inp['clean']).all() or not np.isfinite(inp['satellite']).all():raise ValueError('Same finite cached default logits and IDs required')
   if Counter(zip(inp['TX'].tolist(),inp['RX'].tolist(),inp['day'].tolist()))!=Counter({(t,r,day):20 for t in range(6) for r in (1,3,4,6,8) for day in (1,2,3)}):raise ValueError('Original balanced all5RX V1800 required')
   inputs[arm]=inp;logits[arm]=np.empty((2,3,11,d.N,6),np.float32);logits[arm][:,:,0]=inp['satellite']
  for k in ('ids','TX','RX','day'):
   if not np.array_equal(inputs['B00'][k],inputs['B01'][k]):raise ValueError('Paired models physical metadata differs')
  iq=np.load(d.PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
  if iq.shape!=(2,3,11,d.N,2,256) or iq.dtype!=np.float32:raise ValueError('Exact immutable controlled IQ matrix required')
  with np.load(d.PANEL/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in ('state','lock','signal_to_added_noise_db')}
  if any(a.shape!=(2,3,11,d.N) for a in physics.values()):raise ValueError('Controlled physics alignment differs')
  costs={}
  for arm,m in models.items():
   rowtic=time.perf_counter()
   with torch.no_grad():
    for draw in range(2):
     for si,scene in enumerate(d.SCENES):
      for ci in range(1,11):
       for start in range(0,d.N,d.BATCH):
        a=np.ascontiguousarray(iq[draw,si,ci,start:start+d.BATCH]);x=torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape)
        if not torch.isfinite(x).all():raise ValueError('Nonfinite controlled source IQ')
        z=m.forward_identity_only(x)['tx_logits']
        if z.shape!=(len(x),6) or not torch.isfinite(z).all():raise ValueError('Invalid actual frozen default logits')
        logits[arm][draw,si,ci,start:start+d.BATCH]=z.tolist()
      d.write(out/'progress.json',dict(arm=arm,draw=draw,scene=scene,NN_so_far=m.id_backbone.encoder._full_eval_packets,updates=0,target_access=False));print(arm,draw,scene,'controlled readout complete',flush=True)
   e=m.id_backbone.encoder
   if e._full_train_packets or e._full_eval_packets!=d.NEW_NN_PER_MODEL or any(not torch.equal(v,m.state_dict()[k]) for k,v in states[arm].items()) or any(p.grad is not None for p in m.parameters()):raise ValueError('Frozen state/forward budget differs')
   dest=out/arm;dest.mkdir();np.savez_compressed(dest/'predictions.npz',**{k:inputs[arm][k] for k in ('ids','TX','RX','day','clean')},controlled=logits[arm])
   costs[arm]=dict(NN=e._full_eval_packets,synthetic_NN=2,new_controlled_NN=108000,reused_clean_full_NN=12600,training_updates=0,head_fit_updates=0,state_unchanged=True,wall_seconds=time.perf_counter()-rowtic,training_parent_epochs=30,training_parent_updates=6660,new_epochs=0)
   d.write(dest/'complete.json',costs[arm])
  collect(out,inputs,logits,physics)
 d.write(out/'completion.json',dict(status='SOURCE_ATTRIBUTION_COMPLETE',source_only=True,target_access=False,formal_scores_read=False,training_updates=0,head_fit_updates=0,NN=sum(c['NN'] for c in costs.values()),native_channel_calls=0,reused_clean_full_NN=25200,unique_V=d.N,channel_draws=2,model_seed_count=1,crossseed_confirmed=False,wall_seconds=time.perf_counter()-tic,costs=costs,code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),goal_complete=False))
 print('SOURCE_ATTRIBUTION_COMPLETE 216004NN,0updates,0target',flush=True)

def main():
 try:run()
 except BaseException as e:
  if d.BASE.is_dir():
   p=d.BASE/'failure.json'
   if not p.exists():d.write(p,dict(status='FAILED',error=repr(e),preserve_outputs=True,no_automatic_retry=True))
  raise
