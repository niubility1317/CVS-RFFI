"""Fixed source-V default-head qualification;no training,teacher,U or target."""
import time,os
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from . import design as d
SOURCE_SEED=20261018303
def run(c):
 from .proof import load
 from experiments.cvs_clock_factorial30.channel import Practical
 from experiments.cvs_equivariant_identity.precision import numerical_context
 out=d.BASE/c['row_id']/'source_qualification';out.mkdir(exist_ok=False);start=time.perf_counter();contract=d.read(d.SOURCE);prep=Path(d.PROJECT)/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701';ready=d.read(prep/'preparation_complete.json');packed=torch.load(prep/'source_cache.pt',map_location='cpu',weights_only=False)
 if ready['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or ready['target_access'] or packed['contract']['role_ids']!=contract['role_ids'] or ready['source_contract']['role_ids']!=contract['role_ids']:raise ValueError('Existing original source handles differ')
 V={k:packed['validation'][k] for k in ('x','y','rx','day','ids')};del packed;y,rx,day=(V[k].tolist() for k in ('y','rx','day'));ids=V['ids']
 if V['x'].shape!=(27000,2,256) or len(ids)!=27000 or set(ids)!=set(contract['role_ids']['V']) or Counter(zip(rx,day,y))!=Counter({(r,z,t):300 for r in d.FIT for z in (1,2,3) for t in range(6)}):raise ValueError('Original all5RX V27000 required')
 selected=[]
 for t in range(6):
  for r in d.FIT:
   for z in (1,2,3):selected+=sorted((i for i in range(27000) if (y[i],rx[i],day[i])==(t,r,z)),key=lambda i:ids[i])[:20]
 x=V['x'][selected].float();labels=[y[i] for i in selected];selected_ids=[ids[i] for i in selected];metadata=dict(ids=selected_ids,TX=labels,RX=[rx[i] for i in selected],day=[day[i] for i in selected]);raw=np.asarray(x.tolist(),dtype=np.float64);ch=Practical('train');m=load(c,'cuda:0');state={k:v.detach().clone() for k,v in m.state_dict().items()};metrics={};records=[];NN=physical=0
 with numerical_context(d.FULL_FP32_POLICY),torch.no_grad():
  for view in d.VIEWS:
   preds=[]
   for draw in range(1 if view=='clean' else 2):
    if view=='clean':obs=x
    else:
     cfg=ch.configs[view];values=[]
     for j,pid in enumerate(selected_ids):
      session='rx'+str(metadata['RX'][j])+':day'+str(metadata['day'][j])+':eq1';hw=ch.ch.receiver_for_session(cfg,d.CHANNEL_SEEDS['receiver'],session);seed=ch.ch.stable_seed(SOURCE_SEED,'LSAM/sourceV/fixed2draw',view,draw,pid);signal,_=ch.ch.normalize_input(raw[j,0]+1j*raw[j,1]);v,meta=ch.residual.ResidualChannel(cfg,seed,hw).process(signal)
      if not np.isfinite(v).all() or meta['channel_equalization_applied']:raise ValueError('Source practical residual changed or invalid;no filtering')
      values.append(np.stack((v.real,v.imag)));physical+=1
     a=np.ascontiguousarray(values,dtype=np.float32);obs=torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape)
    pp=[]
    for i in range(0,1800,128):
     z=m.forward_identity_only(obs[i:i+128].to('cuda:0'))['tx_logits'];NN+=len(z)
     if z.shape!=(len(obs[i:i+128]),6) or not bool(torch.isfinite(z).all()):raise ValueError('Default head source prediction invalid')
     pp.extend(z.argmax(1).cpu().tolist())
    preds.extend(pp);cm=np.zeros((6,6),dtype=np.int64)
    for a,b in zip(labels,pp):cm[a,b]+=1
    records.append(dict(view=view,draw=draw,count=1800,correct=int(np.trace(cm)),confusion=cm.tolist(),per_RX={str(r):dict(count=sum(rr==r for rr in metadata['RX']),correct=sum(a==b and rr==r for a,b,rr in zip(labels,pp,metadata['RX']))) for r in d.FIT}))
   metrics[view]=dict(accuracy=sum(a==b for a,b in zip(labels*(1 if view=='clean' else 2),preds))/len(preds),decisions=len(preds))
 if NN!=23400 or physical!=21600 or any(not torch.equal(v,state[k]) for k,v in m.state_dict().items()):raise ValueError('Source cost/state differs')
 d.write(out/'source_ids.json',metadata);result=dict(status='SOURCE_QUALIFIED_FROZEN_READOUT_ONLY',row_id=c['row_id'],metrics=metrics,records=records,source_V_unique=1800,source_cells=90,all_RX=list(d.FIT),source_RX_holdout=None,source_seed=SOURCE_SEED,IQ_points=256,CP=1,NN=NN,teacher_NN=0,physical_channel_calls=physical,cache_loads=1,updates=0,head_fit_updates=0,U_truth_read=False,target=False,model_state_unchanged=True,wall_seconds=time.perf_counter()-start)
 d.write(out/'result.json',result);print('SOURCE_QUALIFICATION_COMPLETE '+c['row_id'],flush=True);return result
def matrix():
 data={r['row_id'].split('-s')[0]:d.read(d.BASE/r['row_id']/'source_qualification/result.json') for r in d.rows()};a=data['AS_LSAM']['metrics'];b=data['AS_REPLAY']['metrics'];delta={v:(a[v]['accuracy']-b[v]['accuracy'])*100 for v in d.VIEWS};checks=dict(clean=delta['clean']>=0.,midurban=delta['practical_mid_urban']>=0.,lowurban=delta['practical_low_urban']>=.5,suburban=all(delta[v]>=-.25 for v in d.VIEWS[1:4]));passed=all(checks.values());result=dict(status='SOURCE_MATRIX_COMPLETE',source_pass=passed,decision='SOURCE_SINGLE_SEED_ONLY_NO_TARGET_BASED_PROMOTION' if passed else 'CLOSED_FIXED_LSAM_SOURCE_RECOGNITION_FAILED',delta_pp=delta,conditions=checks,rows=data,target_access=False,joint_goal_complete=False,new_Ag=False);d.write(d.BASE/'source_matrix_decision.json',result);return result
