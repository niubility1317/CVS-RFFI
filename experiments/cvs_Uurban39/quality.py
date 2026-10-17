"""One fixed first-U source check; no fitting or target feedback."""
from collections import Counter
import numpy as np
import torch
from . import design as d
from .channel import tensor_iq
from experiments.cvs_U_teacher38.maths import teacher_summary

def panel(torch):
 from collections import Counter
 roles=d.read(d.PANEL/'data_roles.json');obs=d.read(d.PANEL/'observations/completion.json')
 if roles['role']!='V' or roles['V_panel']!=1800 or roles['U_truth_used'] or roles['target_access'] or obs['target_access'] or obs['draws']!=2 or obs['conditions'][0]!='full':raise ValueError('Existing source-only panel required')
 contract=d.read(d.SOURCE);done=d.read(d.SOURCE_CACHE/'preparation_complete.json')
 if done.get('status')!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done.get('target_access') or done.get('identity_updates'):raise ValueError('Existing source cache not qualified')
 cache=torch.load(d.SOURCE_CACHE/'source_cache.pt',map_location='cpu',weights_only=False)
 if cache['contract']['role_ids']!=contract['role_ids']:raise ValueError('Existing source role handles differ')
 v=cache['validation'];ids=roles['physical_ids'];lookup={pid:i for i,pid in enumerate(v['ids'])}
 if v['role']!='V' or len(v['ids'])!=27000 or set(v['ids'])!=set(contract['role_ids']['V']) or len(ids)!=1800 or len(set(ids))!=1800 or not set(ids)<=set(lookup):raise ValueError('Source V IDs differ')
 idx=torch.tensor([lookup[pid] for pid in ids]);data={k:v[k][idx].clone() for k in ('x','y','rx','day')};data['ids']=ids
 if data['x'].shape!=(1800,2,256) or not torch.isfinite(data['x']).all() or Counter(zip(data['y'].tolist(),data['rx'].tolist(),data['day'].tolist()))!=Counter({(t,r,d):20 for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)}):raise ValueError('Original balanced finite V1800 required')
 return data,roles

def check(student,teacher,out):
 data,roles=panel(torch);x=data['x'];device=next(student.parameters()).device
 obs=np.load(d.PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
 if obs.shape!=(2,3,11,1800,2,256):raise ValueError('Full original source observations required')
 models=(student,teacher);flags=[[(mod,mod.training) for mod in m.modules()] for m in models];states=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in models]
 z=np.empty((1800,6),np.float32);t=np.empty_like(z);sat=np.empty((2,3,1800,6),np.float32)
 try:
  for m in models:m.eval()
  with torch.no_grad():
   for i in range(0,1800,90):
    xx=x[i:i+90].to(device);z[i:i+90]=student.forward_identity_only(xx)['tx_logits'].tolist();t[i:i+90]=teacher.forward_identity_only(xx)['tx_logits'].tolist()
   for draw in range(2):
    for si in range(3):
     for i in range(0,1800,90):sat[draw,si,i:i+90]=student.forward_identity_only(tensor_iq(obs[draw,si,0,i:i+90],device))['tx_logits'].tolist()
 finally:
  for mm in flags:
   for mod,flag in mm:mod.training=flag
 if any(not torch.equal(v,m.state_dict()[k]) for m,state in zip(models,states) for k,v in state.items()):raise ValueError('Source check mutated model state')
 r,q=teacher_summary(t,z,sat,np.asarray(data['y'].tolist(),np.int64),np.asarray(data['rx'].tolist(),np.int64),d.RULE)
 r.update(NN=14400,student_NN=12600,teacher_NN=1800,training_updates=0,target_access=False,U_truth_read=False,state_unchanged=True)
 d.write(out,r);return r
