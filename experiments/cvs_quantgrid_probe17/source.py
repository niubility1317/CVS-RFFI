"""Reuse exact authorized fit L;do not consume U/V/held values."""
from collections import Counter
import torch
from . import design as d

def load(contract,carrier):
 prep=d.read(d.PREPARATION+'/preparation_complete.json')
 if prep['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prep['target_access'] or prep['identity_updates']:raise ValueError('Known legal source preparation required')
 packed=torch.load(d.PREPARATION+'/source_cache.pt',map_location='cpu',weights_only=False)
 for actual in (packed['contract'],prep['source_contract']):
  if any(actual.get(k)!=v for k,v in carrier.items()) or actual['native_role_comparison']!='EXACT_MATCH':raise ValueError('Unchanged VALIDATED_ONCE source carrier required')
 L=packed['labeled'];ids=L['ids']
 if L['role']!='L_s' or len(ids)!=6300 or len(set(ids))!=6300 or set(ids)!=set(carrier['role_ids']['L_s']):raise ValueError('Original L6300 carrier required')
 ix=[i for i,r in enumerate(L['rx'].tolist()) if int(r) in d.FIT]
 if len(ix)!=5040 or {ids[i] for i in ix}!=set(contract['role_ids']['L_s']):raise ValueError('Only exact fitL5040 values consumed')
 z={k:L[k][ix].clone() for k in ('x','y','rx','day','domain')};z['ids']=[ids[i] for i in ix]
 if z['x'].shape!=(5040,2,256) or not bool(torch.isfinite(z['x']).all()) or Counter(zip(z['y'].tolist(),z['rx'].tolist(),z['day'].tolist()))!=Counter({(t,r,dd):70 for t in d.TX for r in d.FIT for dd in d.DAYS}):raise ValueError('All72source cells70 each required')
 return z
