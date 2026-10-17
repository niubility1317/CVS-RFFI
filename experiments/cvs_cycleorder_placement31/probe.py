"""Same fixed IQ operator before native residual; exact old-input replay control."""
import gzip,json,os,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .maths import replay_guard,source_rule
P=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261017-diagnostic-FcrossRX-cycleorder-placement-sourceV-e30-s2026092701-m2-r01'
ROOT=P/'runs'/RUN
OLD=P/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
POST=P/'runs/20261017-diagnostic-FcrossRX-cycleorder-sourceV-e30-s2026092701-m2-r01'
STUDENT=P/'runs/20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01/source_boundary_mechanism_r01'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
FIELDS=('quality_snr_db','residual_scale','output_frequency_hz','estimated_frequency_hz','signal_to_added_noise_db','agc_gain_db','residual_kernel_energy','numerical_filter_latency_samples','added_noise_variance_reference','measured_added_noise_power_pre_agc','residual_kernel_length')
N=1800;NN=10802;BATCH=90
STATES=('LOS','PARTIAL','BLOCKED');LOCKS=('locked','degraded','unlocked')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def prior_guard(r):
 checks=r['decision']['checks']
 if set(checks)!={'clean_protected','every_RX_protected','every_TX_protected','clean_correct_retained','periodic_reference_supported','nontrivial_view','clean_phase_all_defined','urban_protected'}:raise ValueError('Declared previous clean guard required')
 return all(v for k,v in checks.items() if k!='urban_protected')
def configs():
 import leo_practical.channel as ch
 import leo_practical.residual as residual
 original=read(OLD/'observations/actual_engine.json')
 if ch.__file__!=original['channel_module_file'] or residual.__file__!=original['residual_module_file']:raise ValueError('Same actual leo_practical implementation required')
 result={scene:ch.Config(**original['actual_configs'][scene]) for scene in SCENES}
 required=dict(processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,fs_hz=25000000.,fc_hz=2450000000.,quality_adaptation_enabled=True,phase_tracking_enabled=True,iq_compensation_enabled=True,boundary='reflect')
 for scene,cfg in result.items():
  if any(asdict(cfg).get(k)!=v for k,v in required.items()) or cfg.scenario!=scene:raise ValueError('Full original practical residual values required')
 write(ROOT/'actual_engine.json',dict(channel_module=ch.__file__,residual_module=residual.__file__,actual_configs={s:asdict(c) for s,c in result.items()},EQ_off_G_equals_one=True,formal_test_changed=False))
 return ch,residual,result

def observations(torch,data):
 from experiments.cvs_fullsource_cycleorder30.iq_order import transform
 ch,residual,cfgs=configs();folder=ROOT/'observations';folder.mkdir();iq=np.lib.format.open_memmap(folder/'iq.npy',mode='w+',dtype=np.float32,shape=(2,3,N,2,256));cached=np.load(OLD/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
 if cached.shape!=(2,3,11,N,2,256):raise ValueError('Same P24 source observation matrix required')
 original_raw=np.asarray(data['x'].tolist(),np.float64);post_trace=read(POST/'B01/iq_operator_trace.json');traces={};sessions={};count=0;max_replay=0.;max_noise_relative=0.;state_changes=0;lock_changes=0
 physical={k:np.empty((2,3,2,N),np.float64) for k in FIELDS};physical['state']=np.empty((2,3,2,N),np.int8);physical['lock']=np.empty((2,3,2,N),np.int8)
 with gzip.open(folder/'physical_records.jsonl.gz','xt',encoding='utf8') as log:
  for draw in range(2):
   for si,scene in enumerate(SCENES):
    tag='sourceV/draw'+str(draw)+'/'+scene;xx,meta=transform(data['x'],data['ids'],tag);traces[tag]=meta
    if meta['permutation']!=post_trace[tag]['permutation']:raise ValueError('The two orders must use identical permutations')
    raw=np.asarray(xx.tolist(),np.float64);cfg=cfgs[scene]
    for i,pid in enumerate(data['ids']):
     z,_=ch.normalize_input(original_raw[i,0]+1j*original_raw[i,1]);a,_=ch.normalize_input(raw[i,0]+1j*raw[i,1]);session=f"rx{int(data['rx'][i])}:day{int(data['day'][i])}:eq1";hardware=ch.receiver_for_session(cfg,2026092701,session)
     if sessions.setdefault(session,asdict(hardware))!=asdict(hardware):raise ValueError('RX session changed across TX/draw/scene')
     seed=ch.stable_seed(202610172402,'sourceV/residual_diag24/draw'+str(draw),scene,pid)
     original,m0,t0=residual.ResidualChannel(cfg,seed,hardware).process(z,return_traces=True)
     before,m1,t1=residual.ResidualChannel(cfg,seed,hardware).process(a,return_traces=True)
     old=cached[draw,si,0,i,0]+1j*cached[draw,si,0,i,1]
     error=replay_guard(original,old,t0['residual_kernel'],t1['residual_kernel'],t0['equalizer_kernel'],t1['equalizer_kernel']);max_replay=max(max_replay,error)
     if m0['geometry']!=m1['geometry'] or m0['receiver']!=m1['receiver'] or m0['state_end']!=m1['state_end'] or m0['added_noise_variance_reference']!=m1['added_noise_variance_reference']:raise ValueError('Same physical latent, hardware and noise variance required')
     for m in (m0,m1):
      if m['channel_equalization_applied'] or m['tx_impairment_correction_applied'] or m['true_total_snr_known']:raise ValueError('Original G1/no unknown TX correction/unknown total SNR required')
     if before.shape!=(256,) or not np.isfinite(before).all():raise ValueError('Finite native pre-order IQ required')
     iq[draw,si,i]=np.stack((before.real,before.imag));count+=1;state_changes+=m0['state_end']!=m1['state_end'];lock_changes+=m0['receiver_lock_status']!=m1['receiver_lock_status']
     noise=float(np.linalg.norm(t1['pre_agc_noise']-t0['pre_agc_noise'])/max(np.linalg.norm(t0['pre_agc_noise']),1e-30));max_noise_relative=max(max_noise_relative,noise)
     for bi,m in enumerate((m0,m1)):
      for key in FIELDS:
       value=m[key]
       if value is None or not np.isfinite(value):raise ValueError('Required native physical metric unavailable')
       physical[key][draw,si,bi,i]=value
      physical['state'][draw,si,bi,i]=STATES.index(m['state_end']);physical['lock'][draw,si,bi,i]=LOCKS.index(m['receiver_lock_status'])
     log.write(json.dumps(dict(draw=draw,scene=scene,physical_id=pid,session=session,seed=int(seed),permutation=meta['permutation'][i],replay_max_abs=error,noise_relative_error=noise,original={k:m0[k] for k in FIELDS},before={k:m1[k] for k in FIELDS}),allow_nan=False)+'\n')
    iq.flush();log.flush();write(ROOT/'progress.json',dict(stage='NATIVE_ORDER_OBSERVATIONS',draw=draw,scene=scene,unique_latent_pairs=count,native_calls=count*2,NN=4,updates=0,target_access=False));print(json.dumps(dict(stage='NATIVE_ORDER_OBSERVATIONS',draw=draw,scene=scene,count=count,native_calls=count*2)),flush=True)
 np.savez_compressed(folder/'physics.npz',**physical);write(folder/'iq_operator_trace.json',traces)
 if count!=10800 or len(sessions)!=15:raise ValueError('Complete all-source paired latent matrix required')
 write(folder/'completion.json',dict(status='PRE_RECEIVER_SOURCE_OBSERVATIONS_COMPLETE',native_calls=count*2,unique_existing_latent_pairs=count,new_latent_seeds=0,new_IQ_augment_views=count,original_replay_max_abs=max_replay,pre_agc_noise_max_relative=max_noise_relative,state_changes=int(state_changes),lock_changes=int(lock_changes),RX_sessions=sessions,updates=0,target_access=False))
 return iq

def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU2 source-only job required')
 import torch
 from experiments.cvs_fullsource_concat30_25 import model as api,design as design
 from experiments.cvs_fullsource_cycleorder30.probe import panel
 from experiments.cvs_fullsource_cycleorder30.stats import metric
 tic=time.perf_counter();torch.set_num_threads(2);ROOT.mkdir(exist_ok=False);models={};states={}
 for arm in ('B00','B01'):
  cfg=design.config(next(r for r in design.rows() if r['arm']==arm));m=api.own_model(cfg,'cpu');m.eval().requires_grad_(False)
  h=m.id_backbone.encoder.core.id_backbone.cls_head
  if h.weight.shape!=(6,160) or float(h.scale)!=30. or sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206:raise ValueError('Fixed F and original defaulthead required')
  with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
  if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Actual FINAL30 student no-query smoke failed')
  models[arm]=m;states[arm]={k:v.clone() for k,v in m.state_dict().items()}
 write(ROOT/'checkpoint_noquery_smoke.json',dict(status='PASS',CP_unique=2,NN=4,real_source_IQ_before_smoke=0,target_access=False,updates=0));print('REAL FINAL30 DEFAULT HEAD SMOKE PASS /4NN/0query',flush=True)
 prior=read(POST/'completion.json')
 if prior['status']!='SOURCE_CYCLE_ORDER_DIAGNOSTIC_COMPLETE' or prior['NN']!=25204 or prior['updates'] or prior['target_access']:raise ValueError('Completed fixed post-order source experiment required')
 data,roles=panel(torch);write(ROOT/'data_roles.json',roles);iq=observations(torch,data);y=np.asarray(data['y'].tolist());results={}
 for arm,m in models.items():
  dest=ROOT/arm;dest.mkdir();out=np.empty((2,3,N,6),np.float32)
  with torch.no_grad():
   for draw in range(2):
    for si in range(3):
     for i in range(0,N,BATCH):
      array=np.ascontiguousarray(iq[draw,si,i:i+BATCH]);x=torch.frombuffer(bytearray(array.tobytes()),dtype=torch.float32).reshape(array.shape);z=m.forward_identity_only(x)['tx_logits']
      if z.shape!=(len(x),6) or not torch.isfinite(z).all():raise ValueError('Finite defaulthead predictions required')
      out[draw,si,i:i+BATCH]=z.tolist()
  if any(not torch.equal(v,m.state_dict()[k]) for k,v in states[arm].items()) or any(p.grad is not None for p in m.parameters()) or m.id_backbone.encoder._full_train_packets or m.id_backbone.encoder._full_eval_packets!=NN:raise ValueError('Frozen state or exact NN budget differs')
  baseline=np.load(STUDENT/arm/'predictions.npz',allow_pickle=False);post=np.load(POST/arm/'predictions.npz',allow_pickle=False)
  for a in (baseline,post):
   if not np.array_equal(a['ids'],np.asarray(data['ids'])) or not np.array_equal(a['TX'],y) or not np.array_equal(a['RX'],np.asarray(data['rx'].tolist())):raise ValueError('Same original source physical IDs/labels/RX required')
  if baseline['satellite'].shape!=(2,3,2,N,6) or post['satellite'].shape!=out.shape:raise ValueError('Complete existing source prediction shape required')
  delta=np.zeros((2,2));rows=[]
  for draw in range(2):
   for si,scene in enumerate(SCENES):
    old=metric(y,baseline['satellite'][draw,si,0]);after=metric(y,post['satellite'][draw,si]);before=metric(y,out[draw,si]);change=100*(before['accuracy']-old['accuracy'])
    rows.append(dict(draw=draw,scene=scene,original=old,post_order=after,pre_order=before,delta_pre_vs_original_pp=change,delta_pre_vs_post_pp=100*(before['accuracy']-after['accuracy'])))
    if si:delta[draw,si-1]=change
  old_result=read(POST/arm/'result.json');clean_ok=bool(prior_guard(old_result));decision=source_rule(clean_ok,delta,arm)
  result=dict(arm=arm,decision=decision,source_rows=rows,prior_clean_guard_reference=str(POST/arm/'result.json'),NN=NN,updates=0,state_unchanged=True,source_only=True,target_access=False)
  np.savez_compressed(dest/'predictions.npz',ids=np.asarray(data['ids']),TX=y,RX=data['rx'].tolist(),day=data['day'].tolist(),satellite=out);write(dest/'result.json',result);results[arm]=result;print(json.dumps(dict(stage='SOURCE_PLACEMENT_MODEL_COMPLETE',arm=arm,NN=NN,decision=decision)),flush=True)
 complete=dict(status='SOURCE_CYCLE_ORDER_PLACEMENT_COMPLETE',run_id=RUN,execution_commit=(Path(__file__).resolve().parents[2]/'release_commit.txt').read_text().strip(),models=results,NN=2*NN,native_calls=21600,IQ_augment_views=10800,source_V=1800,channel_draws=2,model_seed_count=1,source_rule_pass=all(r['decision']['status']=='PASS_SOURCE_PLACEMENT_ONLY' for r in results.values()),new_training_epochs=0,updates=0,head_fit_updates=0,target_access=False,wall_seconds=time.perf_counter()-tic,goal_completed=False)
 write(ROOT/'completion.json',complete);print(json.dumps(dict(stage=complete['status'],NN=2*NN,source_rule_pass=complete['source_rule_pass'],wall_seconds=complete['wall_seconds'])),flush=True)
if __name__=='__main__':main()
