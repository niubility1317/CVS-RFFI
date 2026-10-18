"""Reuse only the existing legal frozen source V panel;no first-U extra model readout."""
from collections import Counter
import numpy as np
import torch
from . import design as d
from .channel import tensor_iq

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
