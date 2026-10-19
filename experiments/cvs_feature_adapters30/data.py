"""Consume existing validated source handles; only L is exposed to fitting."""
from collections import Counter
import torch
from . import design as d
from experiments.cvs_F_nontrue20 import design as old

def load(role='L'):
 if role not in ('L','V'):raise ValueError('No U or target fitting interface')
 contract=d.read(d.SOURCE);done=d.read(old.SOURCE_CACHE/'preparation_complete.json')
 if done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done['target_access'] or done['identity_updates']:raise ValueError('Qualified source cache required')
 packed=torch.load(old.SOURCE_CACHE/'source_cache.pt',map_location='cpu',weights_only=False)
 if packed['contract']['role_ids']!=contract['role_ids']:raise ValueError('Existing source role handles differ')
 key='labeled' if role=='L' else 'validation';item=packed[key];values={k:item[k] for k in ('x','y','rx','day','ids')};del packed
 n=6300 if role=='L' else 27000;label='L_s' if role=='L' else 'V';per=70 if role=='L' else 300
 if len(values['ids'])!=n or len(set(values['ids']))!=n or set(values['ids'])!=set(contract['role_ids'][label]):raise ValueError('Source physical IDs/role differ')
 if values['x'].shape!=(n,2,256) or not torch.isfinite(values['x']).all() or values['y'].dtype!=torch.long or Counter(zip(values['y'].tolist(),values['rx'].tolist(),values['day'].tolist()))!=Counter({(t,r,day):per for t in range(6) for r in d.FIT for day in (1,2,3)}):raise ValueError('Balanced source raw IQ/labels differ')
 return values

def diagnostic_indices(L):
 groups={}
 for i,key in enumerate(zip(L['y'].tolist(),L['rx'].tolist(),L['day'].tolist())):groups.setdefault(key,[]).append(i)
 idx=[i for key in sorted(groups) for i in groups[key][:20]]
 if len(idx)!=1800 or len(set(idx))!=1800:raise ValueError('Fixed per-cell L1800 required')
 return idx
