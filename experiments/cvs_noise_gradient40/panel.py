"""Only legal L supplies IQ/labels/gradients;unused V carrier is never used."""
from collections import Counter
import torch
from . import design as d
from .protocol import validate_role
def load():
 contract=d.read(d.SOURCE);prep=d.read(d.CACHE/'preparation_complete.json')
 if prep['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prep['target_access'] or prep['identity_updates']:raise ValueError('Existing validated source carrier required')
 packed=torch.load(d.CACHE/'source_cache.pt',map_location='cpu',weights_only=False)
 if packed['contract']['role_ids']!=contract['role_ids']:raise ValueError('Same existing source handles required')
 L=packed['labeled'];validate_role(L['role'],L['ids'],contract,L['rx'].tolist(),L['day'].tolist(),L['y'].tolist())
 if L['x'].shape!=(6300,2,256) or L['x'].dtype!=torch.float32 or L['y'].dtype!=torch.long or not bool(torch.isfinite(L['x']).all()):raise ValueError('Native finite full L IQ256 required')
 order=torch.randperm(6300,generator=torch.Generator().manual_seed(d.PANEL_SEED));cells={}
 for tx in range(6):
  for rx in d.RX:
   for day in d.DAYS:
    ix=order[(L['y'][order]==tx)&(L['rx'][order]==rx)&(L['day'][order]==day)]
    if len(ix)!=70:raise ValueError('Original L has70 packets in all90 cells')
    cells[(tx,rx,day)]=ix[:10].tolist()
 indices=[cells[k][j] for j in range(10) for k in cells];idx=torch.tensor(indices);out={k:L[k][idx].clone() for k in ['x','y','rx','day','domain']};out['ids']=[L['ids'][i] for i in indices]
 if len(set(out['ids']))!=900 or Counter(zip(out['y'].tolist(),out['rx'].tolist(),out['day'].tolist()))!=Counter({k:10 for k in cells}):raise ValueError('Fixed balanced all-source L900 required')
 return out,dict(role='L_s',available6300=True,source_physical_count=900,per_TX_RX_day=10,all_source_RX=list(d.RX),U_truth_used=False,V_forward=0,V_backprop=0,unused_V_carrier_deserialized=True,old_parent_features_consumed=False,target_access=False,physical_ids=out['ids'],data_contract_role_handles='EXACT_MATCH',normalization=L['preprocessing'])
