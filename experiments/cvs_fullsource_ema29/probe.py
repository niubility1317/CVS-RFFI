"""Two existing all-source EMA states; no training, augmentation or target reads."""
import json,os,time
from pathlib import Path
import numpy as np
from .stats import SCENES,metric,verdict
P=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261017-diagnostic-FcrossRX-EMA-sourceV-e30-s2026092701-m2-r01'
ROOT=P/'runs'/RUN
REF=P/'runs/20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01'
OLD=P/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
STUDENT=P/'runs/20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01/source_boundary_mechanism_r01'
PREP=P/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
CONTRACT=P/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
N=1800;BATCH=90;NN=12602
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def valid_roles(roles,obs):
 if roles.get('role')!='V' or roles.get('counts')!={'L_s':6300,'U_s':56700,'V':27000} or roles.get('V_panel')!=N or roles.get('held_RX') is not None or roles.get('extra_Ldev') is not False or roles.get('U_truth_used') is not False or roles.get('target_access') is not False or roles.get('updates')!=0:raise ValueError('Original all-source V1800 only')
 if obs.get('status')!='SOURCE_OBSERVATIONS_COMPLETE' or obs.get('unique_V')!=N or obs.get('draws')!=2 or obs.get('scenes')!=list(SCENES) or obs.get('physical_draws')!=10800 or obs.get('target_access') is not False or obs.get('training_updates')!=0 or obs.get('formal_test_changed') is not False or len(obs.get('conditions',[]))!=11 or obs['conditions'][0]!='full':raise ValueError('Original fixed full source observations required')
def panel(torch):
 from collections import Counter
 roles=read(OLD/'data_roles.json');valid_roles(roles,read(OLD/'observations/completion.json'))
 contract=read(CONTRACT);done=read(PREP/'preparation_complete.json')
 if done.get('status')!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done.get('target_access') or done.get('identity_updates'):raise ValueError('Existing source cache not qualified')
 cache=torch.load(PREP/'source_cache.pt',map_location='cpu',weights_only=False)
 if cache['contract']['role_ids']!=contract['role_ids']:raise ValueError('Existing source role handles differ')
 v=cache['validation'];ids=roles['physical_ids'];lookup={pid:i for i,pid in enumerate(v['ids'])}
 if v['role']!='V' or len(v['ids'])!=27000 or set(v['ids'])!=set(contract['role_ids']['V']) or len(ids)!=N or len(set(ids))!=N or not set(ids)<=set(lookup):raise ValueError('Source V IDs differ')
 idx=torch.tensor([lookup[pid] for pid in ids]);data={k:v[k][idx].clone() for k in ('x','y','rx','day')};data['ids']=ids
 if data['x'].shape!=(N,2,256) or not torch.isfinite(data['x']).all() or Counter(zip(data['y'].tolist(),data['rx'].tolist(),data['day'].tolist()))!=Counter({(t,r,d):20 for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)}):raise ValueError('Original balanced finite V1800 required')
 return data,roles

def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source diagnostic')
 import torch
 from experiments.cvs_fullsource_concat30_25 import model as api,design as design
 tic=time.perf_counter();torch.set_num_threads(2);ROOT.mkdir(exist_ok=False)
 commit=(Path(__file__).resolve().parents[2]/'release_commit.txt').read_text().strip()
 models={};states={};cpmeta={}
 for arm in ('B00','B01'):
  c=design.config(next(r for r in design.rows() if r['arm']==arm));m=api.own_model(c,'cpu') # Full own scratch30/6660 role/provenance validation before load.
  path=REF/(arm+'-s2026092701/source/final_ssdg.pth');ck=torch.load(path,map_location='cpu',weights_only=False)
  if ck.get('a1_ema_successful_updates')!=6660 or not isinstance(ck.get('ema_model'),dict) or set(ck['ema_model'])!=set(ck['model']):raise ValueError('Own complete final EMA missing')
  m.load_state_dict(ck['ema_model'],strict=True);m.eval().requires_grad_(False)
  h=m.id_backbone.encoder.core.id_backbone.cls_head
  if sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206 or h.weight.shape!=(6,160) or float(h.scale)!=30.:raise ValueError('Fixed F/joint160/defaultcos30 differs')
  with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
  if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Actual EMA CP no-query smoke failed')
  models[arm]=m;states[arm]={k:v.clone() for k,v in m.state_dict().items()}
  cpmeta[arm]=dict(path=str(path),state='ema_model',epoch=30,updates=6660,training_contract='original full L6300/U56700 hidden/V27000',parent_epochs=30,new_epochs=0,NN_smoke=2)
 write(ROOT/'checkpoint_noquery_smoke.json',dict(status='PASS',real_source_IQ_before_smoke=0,NN=4,CP=cpmeta,target_access=False,updates=0))
 print(json.dumps(dict(stage='EMA_REAL_CP_NOQUERY_SMOKE_PASS',NN=4)),flush=True)
 data,roles=panel(torch);write(ROOT/'data_roles.json',roles)
 iq=np.load(OLD/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
 if iq.shape!=(2,3,11,N,2,256) or iq.dtype!=np.float32:raise ValueError('Existing source IQ shape differs')
 y=np.asarray(data['y'].tolist());rx=np.asarray(data['rx'].tolist());summaries={}
 for arm,m in models.items():
  dest=ROOT/arm;dest.mkdir();clean=[];sat=np.empty((2,3,N,6),np.float32)
  with torch.no_grad():
   for i in range(0,N,BATCH):clean.extend(m.forward_identity_only(data['x'][i:i+BATCH])['tx_logits'].tolist())
   for draw in range(2):
    for si in range(3):
     for i in range(0,N,BATCH):
      a=np.ascontiguousarray(iq[draw,si,0,i:i+BATCH]);x=torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape)
      z=m.forward_identity_only(x)['tx_logits']
      if z.shape!=(len(x),6) or not torch.isfinite(z).all():raise ValueError('Finite default logits required')
      sat[draw,si,i:i+BATCH]=z.tolist()
  clean=np.asarray(clean,np.float32)
  if any(not torch.equal(v,m.state_dict()[k]) for k,v in states[arm].items()) or any(p.grad is not None for p in m.parameters()) or m.id_backbone.encoder._full_train_packets or m.id_backbone.encoder._full_eval_packets!=NN:raise ValueError('Model changed or NN cost differs')
  baseline=np.load(STUDENT/arm/'predictions.npz',allow_pickle=False)
  if baseline['satellite'].shape!=(2,3,2,N,6) or baseline['clean'].shape!=(N,6) or not np.array_equal(baseline['ids'],np.asarray(data['ids'])) or not np.array_equal(baseline['TX'],y) or not np.array_equal(baseline['RX'],rx) or read(STUDENT/arm/'result.json').get('NN')!=23402:raise ValueError('Same source student reference differs')
  cz=baseline['clean'];sz=baseline['satellite'][:,:,0];clean_metric=metric(y,clean);old_clean=metric(y,cz)
  rx_delta=[100*(metric(y[rx==r],clean[rx==r])['accuracy']-metric(y[rx==r],cz[rx==r])['accuracy']) for r in (1,3,4,6,8)]
  rows=[];deltas=np.zeros((2,2))
  for draw in range(2):
   for si,scene in enumerate(SCENES):
    old=metric(y,sz[draw,si]);new=metric(y,sat[draw,si]);delta=100*(new['accuracy']-old['accuracy'])
    rows.append(dict(draw=draw,scene=scene,student=old,EMA=new,delta_accuracy_pp=delta,repair=int(((sz[draw,si].argmax(1)!=y)&(sat[draw,si].argmax(1)==y)).sum()),new_error=int(((sz[draw,si].argmax(1)==y)&(sat[draw,si].argmax(1)!=y)).sum())))
    if si: deltas[draw,si-1]=delta
  result=dict(arm=arm,clean_student=old_clean,clean_EMA=clean_metric,source_rows=rows,decision=verdict(100*(clean_metric['accuracy']-old_clean['accuracy']),rx_delta,deltas),NN=NN,updates=0,head_fit_updates=0,source_only=True,state_unchanged=True,CP=cpmeta[arm])
  np.savez_compressed(dest/'predictions.npz',ids=np.asarray(data['ids']),TX=y,RX=rx,day=data['day'].tolist(),clean=clean,satellite=sat)
  write(dest/'result.json',result);summaries[arm]=result
  print(json.dumps(dict(stage='SOURCE_EMA_COMPLETE',arm=arm,decision=result['decision'],NN=NN)),flush=True)
 result=dict(status='SOURCE_EMA_DIAGNOSTIC_COMPLETE',run_id=RUN,commit=commit,models=summaries,NN=2*NN,new_training_epochs=0,updates=0,head_fit_updates=0,teacher_training=0,new_channel_draws=0,source_V=1800,draws=2,model_seeds=1,wall_seconds=time.perf_counter()-tic,target_access=False,Ag_proved=False,goal_completed=False)
 write(ROOT/'completion.json',result);print(json.dumps(dict(stage='SOURCE_EMA_DIAGNOSTIC_COMPLETE',NN=2*NN,wall_seconds=result['wall_seconds'])),flush=True)
if __name__=='__main__':main()
