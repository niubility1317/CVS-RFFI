"""Same-row frozen E30 source mechanism; reuse P24 V observations, no target truth."""
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01'
REFERENCE='20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01'
RELEASE=PROJECT/'releases/cvs_FcrossRX_fullsource_boundarymix_E30_20261017_r01'
OLD=PROJECT/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
PREP=PROJECT/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
CONTRACT=PROJECT/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
CONDITIONS=('full','edge_fixed_AGC')
N=1800;BATCH=90;NN=23402
BRANCHES=('time','frequency','PA','reference','signature')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def roots():return {n:PROJECT/'runs'/(RUN if n=='BCTX' else REFERENCE)/(n+'-s2026092701') for n in ('B00','B01','BCTX')}
def validate_handles(roles,observations):
 if roles.get('role')!='V' or roles.get('counts')!={'L_s':6300,'U_s':56700,'V':27000} or roles.get('V_panel')!=N or roles.get('held_RX') is not None or roles.get('extra_Ldev') is not False or roles.get('U_truth_used') is not False or roles.get('target_access') is not False or roles.get('updates')!=0:raise ValueError('Original V1800 full-source roles only')
 if observations.get('status')!='SOURCE_OBSERVATIONS_COMPLETE' or observations.get('unique_V')!=N or observations.get('draws')!=2 or observations.get('scenes')!=list(SCENES) or observations.get('physical_draws')!=10800 or observations.get('target_access') is not False or observations.get('training_updates')!=0 or observations.get('formal_test_changed') is not False:raise ValueError('Original source observations differ')
 names=observations.get('conditions',[])
 if len(names)!=11 or names[0]!='full' or names[10]!='edge_boundary':raise ValueError('Original fixed-AGC boundary condition required')
def completed_sources():
 spec=importlib.util.spec_from_file_location('boundary_source_completed',Path(__file__).with_name('cvs_fullsource_E30_results_20261017.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 for root in roots().values():m.completed(root)
 return m

def panel(torch):
 roles=read(OLD/'data_roles.json');validate_handles(roles,read(OLD/'observations/completion.json'))
 source=read(CONTRACT);prep=read(PREP/'preparation_complete.json')
 if prep.get('status')!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prep.get('target_access') or prep.get('identity_updates'):raise ValueError('Existing legal source cache only')
 packed=torch.load(PREP/'source_cache.pt',map_location='cpu',weights_only=False)
 if packed['contract']['role_ids']!=source['role_ids']:raise ValueError('Original cached role handles differ')
 V=packed['validation'];ids=roles['physical_ids']
 if V['role']!='V' or len(V['ids'])!=27000 or set(V['ids'])!=set(source['role_ids']['V']) or len(ids)!=N or len(set(ids))!=N or not set(ids)<=set(V['ids']):raise ValueError('Original fixed V1800 IDs required')
 mapping={pid:i for i,pid in enumerate(V['ids'])};ix=torch.tensor([mapping[pid] for pid in ids]);out={k:V[k][ix].clone() for k in ('x','y','rx','day')};out['ids']=ids
 from collections import Counter
 if out['x'].shape!=(N,2,256) or not torch.isfinite(out['x']).all() or Counter(zip(out['y'].tolist(),out['rx'].tolist(),out['day'].tolist()))!=Counter({(t,r,d):20 for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)}):raise ValueError('Balanced finite original source V panel required')
 return out,roles

def modules(m):
 e=m.id_backbone.encoder;b=e.core.id_backbone
 return dict(time=b.t_proj,frequency=b.f_proj,PA=b.pa_proj,reference=e.response_projection,signature=e.sequence_pool)
def gates(m):
 e=m.id_backbone.encoder;h=e.core.id_backbone.cls_head
 return dict(PA=h.gain.tanh().detach().cpu().tolist(),reference=e.response_gain.tanh().detach().cpu().tolist(),signature=(.2*e.sequence_pool.gain_raw.tanh()).detach().cpu().tolist(),kind='static coordinate fusion gains, not per-packet gates',cosine_scale=float(h.scale))
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
 completed=completed_sources() # Only terminal completion metadata; no target scores/truth.
 spec=importlib.util.spec_from_file_location('boundary_source_bootstrap',RELEASE/'experiments/cvs_fullsource_boundary26/runner.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);runtime.bootstrap()
 import torch
 from torch.nn import functional as F
 from experiments.cvs_fullsource_concat30_25 import model as ref_model,design as ref_design
 from experiments.cvs_fullsource_boundary26 import model as own_model,design as own_design
 from experiments.cvs_equivariant_identity.precision import numerical_context
 out=PROJECT/'runs'/RUN/'source_boundary_mechanism_r01';out.mkdir(exist_ok=False);tic=time.perf_counter();torch.set_num_threads(2)
 models={};states={};origins={}
 with numerical_context(own_design.FULL_FP32_POLICY):
  for arm,root in roots().items():
   api,design=(own_model,own_design) if arm=='BCTX' else (ref_model,ref_design)
   c=design.config(next(r for r in design.rows() if r['arm']==arm));m=api.own_model(c,'cpu');m.eval().requires_grad_(False)
   h=m.id_backbone.encoder.core.id_backbone.cls_head
   if h.weight.shape!=(6,160) or float(h.scale)!=30. or sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206:raise ValueError('Original F/joint160/defaultcos30 required')
   with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real E30 default CP no-query smoke failed')
   models[arm]=m;states[arm]={k:v.clone() for k,v in m.state_dict().items()};origins[arm]=str(root/'source/final_ssdg.pth')
  write(out/'checkpoint_noquery_smoke.json',dict(status='PASS',NN=6,real_IQ_before_smoke=0,target_access=False,CP=origins,training_updates=0,head_fit_updates=0))
  data,roles=panel(torch);write(out/'data_roles.json',roles)
  iq=np.load(OLD/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
  if iq.shape!=(2,3,11,N,2,256) or iq.dtype!=np.float32:raise ValueError('Same P24 source IQ matrix required')
  physics=np.load(OLD/'observations/physics.npz',allow_pickle=False)
  outputs={};branches={};cost={}
  for arm,m in models.items():
   dest=out/arm;dest.mkdir();captured={};handles=[]
   for name,module in modules(m).items():
    def record(module,args,value,name=name):captured[name]=value.detach().flatten(1)
    handles.append(module.register_forward_hook(record))
   clean=[];cb={k:[] for k in BRANCHES};sat=np.empty((2,3,2,N,6),np.float32);similar=np.empty((2,3,2,N,5,3),np.float32)
   def forward(x):
    captured.clear();z=m.forward_identity_only(x)['tx_logits']
    if z.shape!=(len(x),6) or not torch.isfinite(z).all() or set(captured)!=set(BRANCHES):raise ValueError('Actual defaulthead/branch readout invalid')
    return z,{k:v.clone() for k,v in captured.items()}
   try:
    with torch.no_grad():
     for i in range(0,N,BATCH):
      z,b=forward(data['x'][i:i+BATCH]);clean.extend(z.tolist())
      for k in BRANCHES:cb[k].append(b[k])
     cb={k:torch.cat(v) for k,v in cb.items()}
     for draw in range(2):
      for si in range(3):
       for ci,oldci in enumerate((0,10)):
        for i in range(0,N,BATCH):
         array=np.ascontiguousarray(iq[draw,si,oldci,i:i+BATCH]);x=torch.frombuffer(bytearray(array.tobytes()),dtype=torch.float32).reshape(array.shape)
         z,b=forward(x);sat[draw,si,ci,i:i+BATCH]=z.tolist()
         for ki,k in enumerate(BRANCHES):similar[draw,si,ci,i:i+BATCH,ki]=torch.stack((F.cosine_similarity(cb[k][i:i+BATCH],b[k]),cb[k][i:i+BATCH].norm(dim=1),b[k].norm(dim=1)),1).tolist()
   finally:
    for handle in handles:handle.remove()
   if any(not torch.equal(v,m.state_dict()[k]) for k,v in states[arm].items()) or any(p.grad is not None for p in m.parameters()) or m.id_backbone.encoder._full_train_packets or m.id_backbone.encoder._full_eval_packets!=NN:raise ValueError('Frozen state or exact source NN differs')
   clean=np.asarray(clean,np.float32);outputs[arm]=(clean,sat);branches[arm]=similar
   np.savez_compressed(dest/'predictions.npz',ids=np.asarray(data['ids']),TX=data['y'].tolist(),RX=data['rx'].tolist(),day=data['day'].tolist(),clean=clean,satellite=sat)
   np.savez_compressed(dest/'branch_metrics.npz',names=np.asarray(BRANCHES),values=similar)
   cost[arm]=dict(NN=NN,synthetic_smoke_NN=2,source_prediction_NN=23400,training_updates=0,head_fit_updates=0,state_unchanged=True,CP=origins[arm],gates=gates(m))
   write(dest/'result.json',cost[arm]);print(json.dumps(dict(stage='SOURCE_MODEL_COMPLETE',arm=arm,NN=NN)),flush=True)
  records=[];y=np.asarray(data['y'].tolist());rx=np.asarray(data['rx'].tolist());cohort=outputs['B00'][0].argmax(1)==y
  for arm,(clean,sat) in outputs.items():
   clean_result=completed.metric(y,clean.argmax(1));write(out/arm/'clean_metrics.json',clean_result)
   for drawset in ((0,),(1,),(0,1)):
    draw=-1 if len(drawset)==2 else drawset[0];yy=np.tile(y,len(drawset));rr=np.tile(rx,len(drawset));cc=np.tile(cohort,len(drawset))
    for si,scene in enumerate(SCENES):
     state=np.concatenate([physics['state'][dd,si,0] for dd in drawset]);lock=np.concatenate([physics['lock'][dd,si,0] for dd in drawset]);quality=np.concatenate([physics['signal_to_added_noise_db'][dd,si,0] for dd in drawset]);strata=[('overall','ALL',np.ones(len(yy),bool))]
     strata += [('RX',str(r),rr==r) for r in (1,3,4,6,8)]
     strata += [('state',s,state==j) for j,s in enumerate(('LOS','PARTIAL','BLOCKED'))]
     strata += [('lock',s,lock==j) for j,s in enumerate(('locked','degraded','unlocked'))]
     strata += [('state_lock',s+'/'+l,(state==j)&(lock==k)) for j,s in enumerate(('LOS','PARTIAL','BLOCKED')) for k,l in enumerate(('locked','degraded','unlocked'))]
     strata += [('added_noise_ratio',label,(quality>=lo)&(quality<hi)) for lo,hi,label in [(-np.inf,3,'<3'),(3,8,'3-8'),(8,15,'8-15'),(15,np.inf,'>=15')]]
     for ci,condition in enumerate(CONDITIONS):
      logits=np.concatenate([sat[dd,si,ci] for dd in drawset]);pred=logits.argmax(1);full=np.concatenate([sat[dd,si,0].argmax(1) for dd in drawset]);ref=np.concatenate([outputs['B01'][1][dd,si,ci].argmax(1) for dd in drawset]);competitors=logits.copy();competitors[np.arange(len(yy)),yy]=-np.inf;margin=(logits[np.arange(len(yy)),yy]-competitors.max(1))/30.
      for dimension,stratum,mask in strata:
       if not mask.any():continue
       q=completed.metric(yy[mask],pred[mask]);q.update(arm=arm,model_seed=2026092701,draw=draw,scene=scene,condition=condition,dimension=dimension,stratum=stratum,margin_over30=float(margin[mask].mean()),repairs_vs_own_full=int(((pred==yy)&(full!=yy)&mask).sum()),breaks_vs_own_full=int(((pred!=yy)&(full==yy)&mask).sum()),gain_vs_B01_pp=100*float(((pred==yy)[mask].mean()-(ref==yy)[mask].mean())),fixed_B00_clean_correct_count=int((cc&mask).sum()),retention_on_B00_clean_correct=float((pred==yy)[cc&mask].mean()) if (cc&mask).any() else None)
       records.append(q)
  completed.write_csv(out/'source_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in records]);write(out/'source_metrics.json',dict(records=records,quality_is_total_SNR=False,edge_receiver='same full state and compensation including AGC; controlled effect, not recomputed total effect',crossseed_confirmed=False,formal_target_access=False))
  write(out/'completion.json',dict(status='SOURCE_BOUNDARY_MECHANISM_COMPLETE',code_commit=a.commit,NN=sum(r['NN'] for r in cost.values()),models=3,model_seed_count=1,channel_draws=2,unique_source_V=N,training_updates=0,head_fit_updates=0,teacher_NN=0,target_access=False,target_scores_or_truth_read=False,new_channel_calls=0,state_unchanged=True,wall_seconds=time.perf_counter()-tic,device='CPU/threads2',source_IQ_cache_V_loaded=27000,L_loaded_unused=6300,U_loaded=0,observation_source=str(OLD),formal_IQ_unchanged=True,interpretation='Source fixed-AGC edge penalty must improve>=3pp at both urban scenes and both draws with full-source accuracy drop<=0.5pp per cell to support reduced boundary sensitivity; no targetfeedback/promotion/retuning',goal_complete=False))
  print(json.dumps(dict(status='SOURCE_BOUNDARY_MECHANISM_COMPLETE',NN=3*NN,updates=0)),flush=True)
if __name__=='__main__':main()
