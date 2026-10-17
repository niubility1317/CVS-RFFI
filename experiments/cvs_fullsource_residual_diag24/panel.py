"""Read the existing source cache. Selection never changes any training role."""
from collections import Counter
import torch
from . import design as d
def load():
 expected=d.read(d.SOURCE);prep=d.read(d.PREPARATION/'preparation_complete.json')
 if prep['target_access'] or prep['identity_updates'] or prep['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE':raise ValueError('Existing legal source cache required')
 packed=torch.load(d.PREPARATION/'source_cache.pt',map_location='cpu',weights_only=False)
 if packed['contract']['role_ids']!=expected['role_ids']:raise ValueError('Original source role handles differ')
 V=packed['validation']
 if V['role']!='V' or len(V['ids'])!=27000 or set(V['ids'])!=set(expected['role_ids']['V']):raise ValueError('Original source V27000 only')
 if V['x'].shape!=(27000,2,256) or not torch.isfinite(V['x']).all():raise ValueError('Native finite normalized IQ256 required')
 d.validate_input(dict(role='V',counts={k:len(v) for k,v in expected['role_ids'].items()},held_RX=None,U_truth_used=False,target_access=False,updates=0))
 order=torch.randperm(27000,generator=torch.Generator().manual_seed(d.PANEL_SEED));cells={}
 for tx in range(6):
  for rx in d.RX:
   for day in d.DAYS:
    ix=order[(V['y'][order]==tx)&(V['rx'][order]==rx)&(V['day'][order]==day)]
    if len(ix)!=300:raise ValueError('Original 90 V cells required')
    cells[(tx,rx,day)]=ix[:20].tolist()
 indices=[cells[key][j] for j in range(20) for key in cells]
 ix=torch.tensor(indices);out={k:V[k][ix].clone() for k in ('x','y','rx','day','domain')};out['ids']=[V['ids'][i] for i in indices]
 if len(set(out['ids']))!=d.N or Counter(zip(out['y'].tolist(),out['rx'].tolist(),out['day'].tolist()))!=Counter({k:20 for k in cells}):raise ValueError('Balanced original V panel required')
 return out,dict(role='V',counts=dict(L_s=6300,U_s=56700,V=27000),held_RX=None,extra_Ldev=False,V_panel=d.N,V_IQ_values_loaded=27000,L_IQ_values_loaded_but_unused=6300,U_IQ_values_loaded=0,old_parent_features_consumed=False,U_truth_used=False,target_access=False,updates=0,physical_ids=out['ids'],normalization='Existing VALIDATED_ONCE native per-packet complex RMS after center crop/pad, center=False, epsilon1e-12')
