"""No target observations, no fitted head, no updates. Same IQ across all three models."""
from collections import Counter
from dataclasses import asdict
import gzip,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .engine import imports,configurations,audit
from .controls import compare
FIELDS=('quality_snr_db','white_proxy_quality_db','actual_waveform_quality_db','waveform_minus_white_power_db','output_frequency_hz','estimated_frequency_hz','signal_to_added_noise_db','noise_coloration_dynamic_range_db','agc_gain_db','agc_limited','residual_kernel_energy','kernel_length','numerical_delay','interior_count','residual_scale')
LOCKS=('locked','degraded','unlocked');STATES=('LOS','PARTIAL','BLOCKED')
def observations(panel,out):
 ch,_=imports();cfgs=configurations();folder=out/'observations';folder.mkdir();shape=(2,3,11,d.N)
 iq=np.lib.format.open_memmap(folder/'iq.npy',mode='w+',dtype=np.float32,shape=(*shape,2,256));physics={k:np.empty(shape,np.float64) for k in FIELDS};physics['lock']=np.empty(shape,np.int8);physics['state']=np.empty(shape,np.int8)
 sessions={};counts=Counter();d.write(folder/'actual_engine.json',audit(cfgs));raw=np.asarray(panel['x'].tolist(),np.float64)
 with gzip.open(folder/'physical_records.jsonl.gz','xt',encoding='utf8',compresslevel=3) as log:
  for draw in d.DRAWS:
   for si,scene in enumerate(d.SCENES):
    cfg=cfgs[scene]
    for i,pid in enumerate(panel['ids']):
     z,_=ch.normalize_input(raw[i,0]+1j*raw[i,1]);session=f"rx{int(panel['rx'][i])}:day{int(panel['day'][i])}:eq1";hardware=ch.receiver_for_session(cfg,d.RECEIVER_SEED,session)
     if sessions.setdefault(session,asdict(hardware))!=asdict(hardware):raise ValueError('RX hardware changed across TX/draw/scene')
     seed=ch.stable_seed(d.CHANNEL_SEED,'sourceV/residual_diag24/draw'+str(draw),scene,pid)
     values,records,native=compare(z,cfg,seed,hardware)
     if tuple(values)!=d.CONDITIONS or native['channel_equalization_applied'] or native['tx_impairment_correction_applied']:raise ValueError('Original G1 residual/condition/TX policy differs')
     if records['full']['boundary_interior_relative_error']>2e-6:raise ValueError('Boundary control changed baseline interior')
     for ci,name in enumerate(d.CONDITIONS):
      r=records[name];v=values[name]
      if v.shape!=(256,) or not np.isfinite(v).all():raise ValueError('Invalid controlled observation')
      if name.endswith('/fixed'):
       for k in ['receiver_lock_status','residual_scale','estimated_frequency_hz','phase_tracking_applied','channel_equalization_applied','agc_gain_db']:
        if r[k]!=records['full'][k]:raise ValueError('Fixed receiver compensation changed: '+k)
      iq[draw,si,ci,i]=np.stack((v.real,v.imag))
      for k in FIELDS:physics[k][draw,si,ci,i]=np.nan if r[k] is None else r[k]
      physics['lock'][draw,si,ci,i]=LOCKS.index(r['receiver_lock_status']);physics['state'][draw,si,ci,i]=STATES.index(r['state'])
     log.write(json.dumps(dict(draw=draw,scene=scene,index=i,physical_id=pid,TX=int(panel['y'][i]),RX=int(panel['rx'][i]),day=int(panel['day'][i]),session=session,seed=int(seed),conditions=records),allow_nan=False)+'\n')
     counts[(draw,scene,native['state_end'],native['receiver_lock_status'])]+=1
    iq.flush();log.flush();d.write(out/'progress.json',dict(stage='SOURCE_OBSERVATIONS',draw=draw,scene=scene,physical_draws=sum(counts.values()),NN=6,updates=0,target_access=False));print(json.dumps(dict(stage='OBSERVATIONS',draw=draw,scene=scene,count=sum(counts.values()))),flush=True)
 np.savez_compressed(folder/'physics.npz',**physics)
 if sum(counts.values())!=10800 or len(sessions)!=15:raise ValueError('Complete fixed source observation matrix required')
 d.write(folder/'completion.json',dict(status='SOURCE_OBSERVATIONS_COMPLETE',unique_V=d.N,draws=2,scenes=list(d.SCENES),conditions=list(d.CONDITIONS),physical_draws=10800,native_residual_calls=10800,controlled_renderer_calls=129600,training_updates=0,target_access=False,receiver_sessions=sessions,counts=[dict(draw=k[0],scene=k[1],state=k[2],lock=k[3],count=v) for k,v in sorted(counts.items())],formal_test_changed=False))
 return iq,physics
def run(device):
 from .model import load,gates
 from .panel import load as load_panel
 from .observe import predict,gradients,capture,forward
 from experiments.cvs_equivariant_identity.precision import numerical_context
 d.validate(d.config());out=d.BASE/'diagnostic';out.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(2);models={};states={};provenance={}
 d.write(out/'resolved_config.json',d.config())
 with numerical_context(d.FP32):
  # Real-checkpoint no-query smoke is the first data-consuming step; no IQ cache is open.
  for arm in d.REFERENCES:
   m,p=load(arm,device);m.eval()
   with capture(m) as readouts,torch.no_grad():q,_=forward(m,torch.zeros(2,2,256,device=device),readouts)
   q=q['tx_logits']
   if q.shape!=(2,6) or not torch.isfinite(q).all():raise ValueError('Real checkpoint no-query smoke failed')
   models[arm]=m;provenance[arm]=p;states[arm]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
  d.write(out/'checkpoint_noquery_smoke.json',dict(status='PASS',checkpoints=3,synthetic_NN=6,source_IQ_before_smoke=0,target_IQ=0,updates=0,head_fit_updates=0,provenance=provenance));print('SMOKE PASS real3/defaulthead/6synthetic/0query',flush=True)
  panel,roles=load_panel();d.write(out/'data_roles.json',roles);iq,physics=observations(panel,out);results={}
  for arm,m in models.items():
   dest=out/arm;dest.mkdir();clean,sat,branch=predict(m,panel,iq,device,dest);grad=gradients(m,panel,iq,device)
   if any(not torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in states[arm].items()) or any(p.grad is not None for p in m.parameters()):raise ValueError('Frozen model state/grad mutated')
   actual=m.id_backbone.encoder._full_eval_packets
   if actual!=121232 or m.id_backbone.encoder._full_train_packets:raise ValueError('Exact readonly diagnostic forward budget differs')
   result=dict(arm=arm,status='SOURCE_DIAGNOSED',NN=actual,prediction_NN=120600,gradient_forward_NN=630,synthetic_smoke_NN=2,training_updates=0,head_fit_updates=0,source_unique_V=d.N,target_access=False,state_unchanged=True,gates=gates(m),gradients=grad,provenance=provenance[arm])
   d.write(dest/'result.json',result);results[arm]=result;d.write(out/'progress.json',dict(stage='MODELS',completed=list(results),NN=sum(r['NN'] for r in results.values()),updates=0,target_access=False));print(json.dumps(dict(stage='MODEL_COMPLETE',arm=arm,NN=actual)),flush=True)
 del models,states,iq
 from .summarize import collect
 summary=collect(out)
 d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',model_seed_count=1,checkpoint_count=3,source_unique_V=d.N,channel_draws=2,NN=sum(r['NN'] for r in results.values()),training_updates=0,head_fit_updates=0,target_access=False,formal_test_changed=False,wall_seconds=time.perf_counter()-start,code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),goal_complete=False))
 print(json.dumps(dict(status='SOURCE_DIAGNOSTIC_COMPLETE',NN=sum(r['NN'] for r in results.values()),updates=0,goal_complete=False)),flush=True)
