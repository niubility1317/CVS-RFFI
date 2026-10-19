"""Existing source handles and one shared source-only full residual observation set."""
from collections import Counter
from dataclasses import asdict
import numpy as np
import torch
from . import design as d
from experiments.cvs_F_nontrue20 import design as base
from experiments.cvs_F_nontrue20.channel import tensor_iq
from experiments.cvs_fullsource_concat20.channel import Practical,serial

def selection(L,contract):
 if len(L['ids'])!=6300 or len(set(L['ids']))!=6300 or set(L['ids'])!=set(contract['role_ids']['L_s']):raise ValueError('Existing L physical handles differ')
 groups={}
 for i,key in enumerate(zip(L['y'].tolist(),L['rx'].tolist(),L['day'].tolist())):groups.setdefault(tuple(key),[]).append(i)
 expected={(t,r,day):70 for t in range(6) for r in (1,3,4,6,8) for day in (1,2,3)}
 if {k:len(v) for k,v in groups.items()}!=expected:raise ValueError('Full balanced source L contract differs')
 keys=sorted(groups,key=lambda k:(k[2],k[1],k[0]));a=[groups[k][j] for j in (0,1) for k in keys];b=[groups[k][j] for j in (2,3) for k in keys]
 if len(a)!=180 or len(b)!=180 or set(a)&set(b):raise ValueError('Disjoint operator packet roles differ')
 return a,b

def observations(out):
 contract=base.read(base.SOURCE);done=base.read(base.SOURCE_CACHE/'preparation_complete.json')
 if done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done['target_access'] or done['identity_updates']:raise ValueError('Existing source cache qualification differs')
 packed=torch.load(base.SOURCE_CACHE/'source_cache.pt',map_location='cpu',weights_only=False)
 if packed['contract']['role_ids']!=contract['role_ids']:raise ValueError('Existing source handles differ')
 L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
 a,b=selection(L,contract);indices=a+b
 raw=L['x'][indices].clone();y=L['y'][indices].long();rx=L['rx'][indices].long();day=L['day'][indices].long();ids=[L['ids'][i] for i in indices]
 if raw.shape!=(360,2,256) or not torch.isfinite(raw).all():raise ValueError('Finite source-only IQ shape differs')
 engine=Practical();views={'raw':raw};metadata={}
 for scene in d.SCENES:
  cfg=engine.configs[scene];values=[];records=[]
  for i in range(360):
   session=f'rx{int(rx[i])}:day{int(day[i])}:eq1';hw=engine.ch.receiver_for_session(cfg,d.RECEIVER_SEED,session)
   seed=engine.ch.stable_seed(d.CHANNEL_SEED,'F_cleanLookahead/sourceL/draw0',scene,ids[i])
   v=np.asarray(raw[i].tolist(),dtype=np.float64);normalized,_=engine.ch.normalize_input(v[0]+1j*v[1]);z,m=engine.residual.ResidualChannel(cfg,seed,hw).process(normalized)
   if not np.isfinite(z).all() or m['channel_equalization_applied']:raise ValueError('Complete finite source residual required')
   values.append(np.stack((z.real,z.imag)));records.append(serial(dict(m,physical_id=ids[i],RX=int(rx[i]),day=int(day[i]),session=session,scene=scene)))
  views[scene]=tensor_iq(np.asarray(values),'cpu');metadata[scene]=records
 d.write(out/'source_roles.json',dict(source_role='L_s',step_ids=ids[:180],response_ids=ids[180:],both_historically_trained_L=True,source_cache_other_roles_deserialized_but_unused=True,U_truth_read=False,V_used=False,target_access=False))
 d.write(out/'actual_channel.json',dict(audit=engine.audit(),diagnostic_namespace='F_cleanLookahead/sourceL/draw0',receiver_seed=d.RECEIVER_SEED,channel_seed=d.CHANNEL_SEED,physical_draws=720,all_observations_shared_across_models=True,raw_ordinary_augmentation=False))
 d.write(out/'source_observations_metadata.json',metadata)
 np.savez_compressed(out/'source_observations.npz',ids=np.asarray(ids),TX=y.tolist(),RX=rx.tolist(),day=day.tolist(),**{k:np.asarray(v.tolist(),dtype=np.float32) for k,v in views.items()})
 return dict(views=views,y=y,rx=rx,day=day,ids=ids)
