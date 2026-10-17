"""Frozen source-only replication evidence; never a gate to fixed formal scoring."""
import time
from pathlib import Path
import numpy as np
import torch
from . import design as d,model as api
from .channel import tensor_iq
PROJECT=Path(d.PROJECT)
REFERENCES=PROJECT/'runs/20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01'
PANEL=PROJECT/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
BASE_PRED=PROJECT/'runs/20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01/source_boundary_mechanism_r01'
PREP=PROJECT/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
def decide(arm,cleanV,referenceV,urban,referenceUrban):
 if arm not in ('B10','B11') or not np.isfinite(cleanV) or not np.isfinite(referenceV) or not 0<=cleanV<=1 or not 0<=referenceV<=1:raise ValueError('Registered source probabilities required')
 if np.shape(urban)!=(2,2) or np.shape(referenceUrban)!=(2,2) or not np.isfinite(urban).all() or not np.isfinite(referenceUrban).all() or np.any((urban<0)|(urban>1)) or np.any((referenceUrban<0)|(referenceUrban>1)):raise ValueError('Two source draws by two urban scenes required')
 delta=100*(np.asarray(urban)-np.asarray(referenceUrban));clean_delta=100*(cleanV-referenceV)
 checks=dict(clean_source_health=clean_delta>=-.5-1e-10)
 if arm=='B11':checks['each_urban_draw']=bool((delta>=1.-1e-10).all())
 return dict(source_only=True,checks=checks,source_replication_evidence=all(checks.values()),clean_V_delta_pp=clean_delta,urban_draw_delta_pp=delta.tolist(),formal_access=False,automatic_extension=False,Ag_proved=False,crossseed_confirmed=False)
def run(row):
 c=d.config(next(r for r in d.rows() if r['row_id']==row));out=d.BASE/row/'source_qualification';out.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(2)
 from experiments.cvs_equivariant_identity.precision import numerical_context
 with numerical_context(d.FULL_FP32_POLICY):
  m=api.own_model(c,'cpu');before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256,device='cpu'))['tx_logits']
  if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Own FINAL30 source no-query smoke failed')
  d.write(out/'smoke.json',dict(status='PASS',NN=2,IQ_before_smoke=0,target_access=False,head_fit_updates=0))
  source=d.read(d.SOURCE);roles=d.read(PANEL/'data_roles.json');ids=roles['physical_ids']
  if roles['role']!='V' or roles['V_panel']!=1800 or roles['counts']!=dict(L_s=6300,U_s=56700,V=27000) or roles['held_RX'] is not None or roles['extra_Ldev'] or roles['U_truth_used'] or roles['target_access']:raise ValueError('Original full-source V1800 only')
  prep=d.read(PREP/'preparation_complete.json')
  if prep['target_access'] or prep['identity_updates'] or prep['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE':raise ValueError('Existing source cache provenance differs')
  cache=torch.load(PREP/'source_cache.pt',map_location='cpu',weights_only=False)
  if cache['contract']['role_ids']!=source['role_ids']:raise ValueError('Source physical role handles differ')
  V=cache['validation']
  if V['role']!='V' or len(V['ids'])!=27000 or set(V['ids'])!=set(source['role_ids']['V']) or len(ids)!=1800 or len(set(ids))!=1800 or not set(ids)<=set(V['ids']):raise ValueError('Fixed original V panel IDs required')
  indices={pid:i for i,pid in enumerate(V['ids'])};ix=torch.tensor([indices[pid] for pid in ids]);x=V['x'][ix];y=np.asarray(V['y'][ix].tolist(),dtype=np.int64)
  if x.shape!=(1800,2,256) or not torch.isfinite(x).all():raise ValueError('Finite original source IQ256 required')
  obs=d.read(PANEL/'observations/completion.json')
  if obs['status']!='SOURCE_OBSERVATIONS_COMPLETE' or obs['target_access'] or obs['draws']!=2 or obs['scenes']!=list(d.TRAIN_SCENES) or obs['conditions'][0]!='full':raise ValueError('Fixed source native observations required')
  array=np.load(PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
  if array.shape!=(2,3,11,1800,2,256) or array.dtype!=np.float32:raise ValueError('Original source observation shape differs')
  clean=np.empty((1800,6),np.float32);sat=np.empty((2,3,1800,6),np.float32)
  with torch.no_grad():
   for i in range(0,1800,256):clean[i:i+256]=m.forward_identity_only(x[i:i+256].to('cpu'))['tx_logits'].cpu().tolist()
   for draw in range(2):
    for scene in range(3):
     for i in range(0,1800,256):sat[draw,scene,i:i+256]=m.forward_identity_only(tensor_iq(array[draw,scene,0,i:i+256],'cpu'))['tx_logits'].cpu().tolist()
  if not np.isfinite(clean).all() or not np.isfinite(sat).all() or m.id_backbone.encoder._full_eval_packets!=12602 or m.id_backbone.encoder._full_train_packets or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen source readout or exact NN differs')
  ref='B00' if c['arm']=='B10' else 'B01';done=d.read(REFERENCES/(ref+'-s'+str(d.SEED))/'source/completion.json');own=d.read(Path(c['output_root'])/'completion.json')
  if done['epoch']!=30 or done['optimizer_steps']!=6660 or done['target_access'] or done['head_fit_updates'] or done['config']['source_roles']!=dict(L_s=6300,U_s=56700,V=27000):raise ValueError('Matching source reference metadata required')
  with np.load(BASE_PRED/ref/'predictions.npz',allow_pickle=False) as f:
   if not np.array_equal(f['ids'],np.asarray(ids)) or not np.array_equal(f['TX'],y):raise ValueError('Matched source decisions/labels required')
   refsat=f['satellite'][:,:,0].copy()
  accuracies=(sat.argmax(-1)==y).mean(-1);refacc=(refsat.argmax(-1)==y).mean(-1)
  verdict=decide(c['arm'],own['final_source_metrics']['source_val_accuracy'],done['final_source_metrics']['source_val_accuracy'],accuracies[:,1:],refacc[:,1:])
  np.savez_compressed(out/'predictions.npz',ids=np.asarray(ids),TX=y,clean=clean,satellite=sat)
  d.write(out/'complete.json',dict(status='SOURCE_QUALIFIED_READOUT',config=c,source_roles='EXACT_ORIGINAL_V',source_unique_V=1800,device='cpu',torch_threads=2,reference_device='cpu',reference_torch_threads=2,channel_draws=2,clean_accuracy=float((clean.argmax(1)==y).mean()),source_accuracies=accuracies.tolist(),reference_accuracies=refacc.tolist(),verdict=verdict,NN=12602,synthetic_smoke_NN=2,teacher_NN=0,head_fit_updates=0,training_updates=0,target_access=False,formal_scores_read=False,state_unchanged=True,source_cache_V_loaded=27000,L_loaded_unused=6300,U_loaded=0,new_channel_calls=0,wall_seconds=time.perf_counter()-start))
  return verdict
