"""Generate one complete source training observation stream, shared by all six rows."""
import gzip,json,time
from collections import Counter
from dataclasses import asdict
import numpy as np
import torch
from . import design as d
from .data import load,diagnostic_indices
from experiments.cvs_fullsource_concat20.channel import Practical,serial
from experiments.cvs_fullsource_concat20.design import CHANNEL_SEEDS,TRAIN_SCENES

def sequence():
 rng=np.random.default_rng(20261019643);values=[]
 while len(values)<d.UPDATES*d.BATCH:values.extend(rng.permutation(6300).tolist())
 return np.asarray(values[:d.UPDATES*d.BATCH],dtype=np.int64).reshape(d.UPDATES,d.BATCH)

def observation(engine,raw,pid,rx,day,scene,namespace,slot):
 cfg=engine.configs[scene];session=f'rx{rx}:day{day}:eq1';hw=engine.ch.receiver_for_session(cfg,CHANNEL_SEEDS['receiver'],session);hardware=asdict(hw)
 if engine.sessions.setdefault(session,hardware)!=hardware:raise ValueError('Hardware must be RX/day, never TX')
 seed=engine.ch.stable_seed(CHANNEL_SEEDS['train'],namespace,scene,pid,slot);normalized,_=engine.ch.normalize_input(raw[0]+1j*raw[1]);z,meta=engine.residual.ResidualChannel(cfg,seed,hw).process(normalized)
 if not np.isfinite(z).all() or meta['channel_equalization_applied']:raise ValueError('Finite complete residual/noEQ required; no dropping')
 return np.asarray([z.real,z.imag],np.float32),serial(dict(meta,physical_id=pid,RX=rx,day=day,session_id=session,scene=scene,namespace=namespace,slot=slot))

def build():
 out=d.BASE/'observation_bank';out.mkdir(exist_ok=False);tic=time.perf_counter();L=load();idx=sequence();diag=diagnostic_indices(L);positions={v:i for i,v in enumerate(diag)};seen=set();engine=Practical();raw=np.asarray(L['x'].tolist(),np.float64);rx=L['rx'].tolist();day=L['day'].tolist();ids=L['ids']
 torch.save(L,out/'labeled.pt');np.save(out/'indices.npy',idx);arr=np.lib.format.open_memmap(out/'training_As.npy',mode='w+',dtype=np.float32,shape=(d.UPDATES,d.BATCH,2,256));first=np.empty((1800,2,256),np.float32);firstmeta=[None]*1800;coverage=Counter()
 d.write(out/'source_roles.json',dict(fit_role='L_s',L=6300,U_used=False,U_truth_read=False,V_fitted=False,target_access=False,source_cache_other_roles_deserialized_but_unused=True,role_ids=d.read(d.SOURCE)['role_ids'],L_diagnostic_indices=diag,L_diagnostic_ids=[ids[i] for i in diag]))
 with gzip.open(out/'training_metadata.jsonl.gz','wt',encoding='utf8') as stream:
  for s,indices in enumerate(idx):
   epoch=21+s//d.STEPS;namespace=f'F_featureAdapters/train/e{epoch}/step{s+1}/L_s'
   for j,value in enumerate(indices.tolist()):
    pid=ids[value];rng=np.random.default_rng(engine.ch.stable_seed(CHANNEL_SEEDS['train'],namespace,'scene',pid,j));scene=str(rng.choice(TRAIN_SCENES));z,meta=observation(engine,raw[value],pid,rx[value],day[value],scene,namespace,j);arr[s,j]=z;coverage[(rx[value],day[value],scene,meta['state_end'],meta['receiver_lock_status'])]+=1
    stream.write(json.dumps(dict(step=s+1,L_index=value,**meta))+'\n')
    if value in positions and value not in seen:first[positions[value]]=z;firstmeta[positions[value]]=dict(step=s+1,**meta);seen.add(value)
   if (s+1)%22==0 or s+1==d.UPDATES:
    arr.flush();d.write(out/'progress.json',dict(status='GENERATING',steps=s+1,total=d.UPDATES,physical_observations=(s+1)*128,seconds=time.perf_counter()-tic));print('BANK '+str(s+1)+'/'+str(d.UPDATES),flush=True)
 if len(seen)!=1800:raise ValueError('Actual first train observations missing')
 np.save(out/'actual_first_training_As.npy',first);d.write(out/'actual_first_training_metadata.json',firstmeta)
 fresh=np.lib.format.open_memmap(out/'same_L_new_channel.npy',mode='w+',dtype=np.float32,shape=(2,2,1800,2,256));freshmeta=[]
 for draw in range(2):
  for si,scene in enumerate(('practical_mid_urban','practical_low_urban')):
   for j,value in enumerate(diag):
    z,meta=observation(engine,raw[value],ids[value],rx[value],day[value],scene,f'F_featureAdapters/sourceL/new/draw{draw}',j);fresh[draw,si,j]=z;freshmeta.append(meta)
 fresh.flush();d.write(out/'same_L_new_metadata.json',freshmeta)
 d.write(out/'complete.json',dict(status='SHARED_OBSERVATIONS_COMPLETE',updates=d.UPDATES,batch=d.BATCH,physical_training_observations=d.UPDATES*d.BATCH,diagnostic_new_observations=7200,actual_first_training_IQ_saved=True,source_sampler_seed=20261019643,cumulative_epochs=[21,30],all_six_share_actual_IQ=True,channel_audit=engine.audit(),sessions=engine.sessions,coverage=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(coverage.items())],target_access=False,U_used=False,V_fitted=False,wall_seconds=time.perf_counter()-tic))

def open_bank():
 out=d.BASE/'observation_bank';done=d.read(out/'complete.json')
 if done['status']!='SHARED_OBSERVATIONS_COMPLETE' or done['updates']!=2220 or done['batch']!=128 or done['U_used'] or done['target_access'] or done['V_fitted']:raise ValueError('Fixed legal complete shared stream required')
 L=torch.load(out/'labeled.pt',map_location='cpu',weights_only=False);contract=d.read(d.SOURCE)
 if len(L['ids'])!=6300 or set(L['ids'])!=set(contract['role_ids']['L_s']):raise ValueError('Fitting only L IDs required')
 indices=np.load(out/'indices.npy',allow_pickle=False);views=np.load(out/'training_As.npy',mmap_mode='r',allow_pickle=False)
 if not np.array_equal(indices,sequence()) or views.shape!=(2220,128,2,256) or views.dtype!=np.float32:raise ValueError('Shared observation stream differs')
 return L,indices,views
