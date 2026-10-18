import numpy as np
from . import design as d

def load_source():
 import torch
 expected=d.read(d.SOURCE);packed=torch.load(d.CACHE,map_location='cpu',weights_only=False)
 if any(packed['contract'].get(k)!=v for k,v in expected.items()) or packed['contract'].get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Unchanged source carrier contract mismatch')
 roles={}
 for name,key,n,count in [('L_s','labeled',6300,70),('V','validation',27000,300)]:
  source=packed[key];out={k:source[k] for k in ('x','y','rx','day','ids')}
  if out['x'].shape!=(n,2,256) or len(set(out['ids']))!=n or set(out['ids'])!=set(expected['role_ids'][name]):raise ValueError('Exact source role ID mismatch')
  for k in ('y','rx','day'):out[k]=np.asarray(out[k].tolist(),np.int64)
  if any(int(((out['y']==t)&(out['rx']==r)&(out['day']==day)).sum())!=count for t in range(6) for r in d.RX for day in (1,2,3)):raise ValueError('All90 source cells required')
  roles[name]=out
 if set(roles['L_s']['ids'])&set(roles['V']['ids']) or (set(roles['L_s']['ids'])|set(roles['V']['ids']))&set(expected['role_ids']['U_s']):raise ValueError('Source roles overlap')
 del packed
 return roles
