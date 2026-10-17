"""Source-only flat-response timing/phase diagnosis; original received tests stay fixed."""
import gzip,os,time
from dataclasses import asdict
import numpy as np
from . import design as d

def run():
 import torch
 from experiments.cvs_cycleorder_factorial32 import model as api,design as prior
 from experiments.cvs_fullsource_cycleorder30.probe import panel
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from .render import render,imports
 from .stats import collect
 tic=time.perf_counter();torch.set_num_threads(2);out=d.BASE;models={};states={};configs={};d.validate(d.config());d.write(out/'resolved_config.json',d.config());d.write(out/'runtime.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=__import__('sys').argv,CUDA_VISIBLE_DEVICES='',threads=2))
 with numerical_context(d.FP32):
  for arm in d.ARMS:
   c=prior.config(next(r for r in prior.rows() if r['arm']==arm));m=api.own_model(c,'cpu');h=m.id_backbone.encoder.core.id_backbone.cls_head
   if sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206 or h.weight.shape!=(6,160) or float(h.scale)!=30.:raise ValueError('Actual F/defaultcos30 differs')
   with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Own FINAL30 no-query smoke failed')
   models[arm]=m;states[arm]={k:v.clone() for k,v in m.state_dict().items()};configs[arm]=c
  d.write(out/'checkpoint_noquery_smoke.json',dict(status='PASS',NN=4,source_IQ_before_smoke=0,target_IQ=0,training_updates=0,checkpoints={a:dict(path=str(d.PARENT/(a+'-s2026092701/source/final_ssdg.pth')),kind='model',epoch=30,updates=6660,source_roles='EXACT_MATCH',scratch_only=True) for a in d.ARMS}));print('SMOKE PASS two real FINAL30 default heads,4synthetic,0query',flush=True)
  data,roles=panel(torch);d.write(out/'data_roles.json',roles);ids=data['ids'];raw=np.asarray(data['x'].tolist(),np.float64);inputs={};logits={}
  olddone=d.read(d.ATTRIBUTION/'completion.json');oldcfg=d.read(d.ATTRIBUTION/'resolved_config.json');oldsmoke=d.read(d.ATTRIBUTION/'checkpoint_noquery_smoke.json')
  if olddone['status']!='SOURCE_ATTRIBUTION_COMPLETE' or olddone['target_access'] or olddone['training_updates'] or oldcfg['checkpoint_parent']!=str(d.PARENT) or oldcfg['conditions'][3]!='flat_H/fixed' or oldcfg['source_role']!='V':raise ValueError('Existing matched flat-H source predictions differ')
  for arm in d.ARMS:
   if oldsmoke['checkpoints'][arm]['path']!=str(d.PARENT/(arm+'-s2026092701/source/final_ssdg.pth')):raise ValueError('Cached logits model source differs')
   with np.load(d.ATTRIBUTION/arm/'predictions.npz',allow_pickle=False) as f:inp={k:f[k].copy() for k in ('ids','TX','RX','day','clean')};controlled=f['controlled'].copy()
   if inp['ids'].tolist()!=ids or controlled.shape!=(2,3,11,d.N,6) or not np.isfinite(controlled).all() or not np.isfinite(inp['clean']).all():raise ValueError('Cached source default logits differ')
   for key,actual in [('TX',data['y']),('RX',data['rx']),('day',data['day'])]:
    if not np.array_equal(inp[key],np.asarray(actual.tolist())):raise ValueError('Original balanced V physical metadata differs')
   inputs[arm]=inp;logits[arm]=np.empty((2,2,4,d.N,6),np.float32);logits[arm][:,:,0]=np.stack([controlled[:,1,3],controlled[:,2,3]],axis=1)
  ch,residual=imports();engine=d.read(d.PANEL/'observations/actual_engine.json');completion=d.read(d.PANEL/'observations/completion.json')
  if ch.__file__!=engine['channel_module_file'] or residual.__file__!=engine['residual_module_file'] or not engine['EQ_off_G_equals_one'] or engine['formal_test_changed']:raise ValueError('Actual original channel module/G1 contract differs')
  cfgs={s:ch.Config(**engine['actual_configs'][s]) for s in d.SCENES};d.write(out/'actual_engine.json',dict(**engine,new_renderer=__import__('sys').modules[render.__module__].__file__))
  oldiq=np.load(d.PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
  if oldiq.shape!=(2,3,11,d.N,2,256):raise ValueError('Original source observations differ')
  array=np.lib.format.open_memmap(out/'controlled_iq.npy',mode='w+',dtype=np.float32,shape=(2,2,4,d.N,2,256));seen=set();replaymax=0.;noisemax=0.;energymax=0.;rendercalls=0;render_tic=time.perf_counter()
  with gzip.open(d.PANEL/'observations/physical_records.jsonl.gz','rt',encoding='utf8') as src,gzip.open(out/'control_records.jsonl.gz','xt',encoding='utf8') as log:
   for line in src:
    rec=__import__('json').loads(line);scene=rec['scene']
    if scene not in d.SCENES:continue
    draw=rec['draw'];si=d.SCENES.index(scene);i=rec['index'];key=(draw,si,i)
    if key in seen or draw not in (0,1) or not 0<=i<d.N or rec['physical_id']!=ids[i] or (rec['TX'],rec['RX'],rec['day'])!=(int(data['y'][i]),int(data['rx'][i]),int(data['day'][i])):raise ValueError('Physical controlled-record ordering differs')
    seen.add(key);cfg=cfgs[scene];session='rx'+str(rec['RX'])+':day'+str(rec['day'])+':eq1';expected=ch.stable_seed(202610172402,'sourceV/residual_diag24/draw'+str(draw),scene,ids[i]);hw=ch.receiver_for_session(cfg,2026092701,session)
    if rec['seed']!=expected or rec['session']!=session or asdict(hw)!=completion['receiver_sessions'][session]:raise ValueError('Same named stream/RXday hardware required')
    x,_=ch.normalize_input(raw[i,0]+1j*raw[i,1]);frozen=rec['conditions']['full'];details=[];baseline_noise=None
    for mi,(delay,zero) in enumerate(d.GRID):
     y,r=render(x,cfg,rec['seed'],hw,frozen=frozen,delay_samples=delay,zero_phase=zero);rendercalls+=1;arr=np.stack((y.real,y.imag))
     if not np.isfinite(arr).all():raise ValueError('Nonfinite source counterfactual')
     if mi==0:
      old=oldiq[draw,d.CACHED_SCENES.index(scene),3,i];replaymax=max(replaymax,float(np.max(abs(arr-old))))
      if not np.allclose(arr,old,atol=2e-6,rtol=2e-6):raise ValueError('Existing flat-H/fixed source IQ replay differs')
      baseline_noise=r['_noise']
     noisemax=max(noisemax,float(np.max(abs(r['_noise']-baseline_noise))));energymax=max(energymax,abs(r['residual_kernel_energy']-frozen['residual_kernel_energy'])/max(frozen['residual_kernel_energy'],1e-30))
     for name in ('receiver_lock_status','residual_scale','estimated_frequency_hz','phase_tracking_applied','channel_equalization_applied','agc_gain_db'):
      if r[name]!=frozen[name]:raise ValueError('Fixed receiver compensation differs:'+name)
     array[draw,si,mi,i]=arr;details.append(dict(mode=d.MODES[mi],**{k:v for k,v in r.items() if not k.startswith('_')}))
    log.write(__import__('json').dumps(dict(draw=draw,scene=scene,index=i,physical_id=ids[i],controls=details),allow_nan=False)+'\n')
    if i==d.N-1:array.flush();print('SOURCE_RENDER',draw,scene,'complete',flush=True)
  if len(seen)!=7200 or rendercalls!=28800 or noisemax!=0. or energymax>5e-12:raise ValueError('Exact paired controls/noise/energy budget differs')
  d.write(out/'observation_completion.json',dict(status='SOURCE_CONTROLS_COMPLETE',unique_V=d.N,physical_draws=len(seen),controlled_renderer_calls=rendercalls,native_residual_calls=0,replay_max_abs=replaymax,noise_max_abs=noisemax,max_relative_kernel_energy_error=energymax,fixed_receiver=True,wall_seconds=time.perf_counter()-render_tic,target_access=False));costs={}
  for arm,m in models.items():
   t=time.perf_counter()
   with torch.no_grad():
    for draw in range(2):
     for si,scene in enumerate(d.SCENES):
      for mi in (1,2,3):
       for start in range(0,d.N,d.BATCH):
        a=np.ascontiguousarray(array[draw,si,mi,start:start+d.BATCH]);x=torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape);z=m.forward_identity_only(x)['tx_logits']
        if z.shape!=(len(x),6) or not torch.isfinite(z).all():raise ValueError('Finite actual default logits required')
        logits[arm][draw,si,mi,start:start+d.BATCH]=z.tolist()
      print('READOUT',arm,draw,scene,'complete',flush=True)
   e=m.id_backbone.encoder
   if e._full_train_packets or e._full_eval_packets!=d.NN_PER_MODEL or any(not torch.equal(v,m.state_dict()[k]) for k,v in states[arm].items()) or any(p.grad is not None for p in m.parameters()):raise ValueError('Frozen state or exact NN differs')
   folder=out/arm;folder.mkdir();np.savez_compressed(folder/'predictions.npz',**inputs[arm],logits=logits[arm]);costs[arm]=dict(NN=e._full_eval_packets,new_control_NN=21600,synthetic_NN=2,state_unchanged=True,training_updates=0,head_fit_updates=0,wall_seconds=time.perf_counter()-t);d.write(folder/'complete.json',costs[arm])
  result=collect(out,inputs,logits)
 d.write(out/'completion.json',dict(status='SOURCE_DELAY_PHASE_COMPLETE',code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),NN=43204,source_only=True,target_access=False,formal_scores_read=False,training_updates=0,head_fit_updates=0,source_unique_V=d.N,channel_draws=2,controlled_renderer_calls=28800,native_residual_calls=0,model_seed_count=1,crossseed_confirmed=False,source_decision=result['source_decision'],costs=costs,wall_seconds=time.perf_counter()-tic,goal_complete=False));print('SOURCE_DELAY_PHASE_COMPLETE,43204NN,0updates,0target',flush=True)

def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU2 only')
 d.BASE.mkdir(exist_ok=False) # before try: an existing root can never receive a new failure file
 try:run()
 except BaseException as e:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),preserve_outputs=True,no_automatic_retry=True));raise
