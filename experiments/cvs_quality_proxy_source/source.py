import os,sys,time
from pathlib import Path
import numpy as np
from . import design as d

def source_observations():
    import torch
    torch.set_num_threads(2)
    expected=d.read(d.SOURCE);prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
    if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['target_access'] or prepared['identity_updates']:raise ValueError('Existing legal source preparation required')
    packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
    for actual in (packed['contract'],prepared['source_contract']):
        if any(actual.get(k)!=v for k,v in expected.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Existing source carrier differs')
    L=packed['labeled'];ids=L['ids'];all_ids=set(ids)
    if len(ids)!=6300 or len(all_ids)!=6300 or all_ids!=set(expected['role_ids']['L_s']) or all_ids&(set(expected['role_ids']['U_s'])|set(expected['role_ids']['V'])) or L['x'].shape!=(6300,2,256):raise ValueError('Original L6300 carrier only')
    selected=[i for i,r in enumerate(L['rx'].tolist()) if int(r) in d.FIT];reference=d.read(Path(d.REFERENCE)/'source_contract.json')
    if len(selected)!=5040 or {ids[i] for i in selected}!=set(reference['role_ids']['L_s']) or reference['held_RX']!=1 or reference['source_receivers']!=list(d.FIT):raise ValueError('Exact current C0 fitL5040 only')
    # RX1 and cached U/V values are not inspected. Cache deserialization is disclosed.
    x=np.asarray(L['x'][selected].double().tolist(),dtype=np.float64)
    tx,rx,day=(np.asarray(L[k][selected].tolist()) for k in ('y','rx','day'))
    return x,tx,rx,day,[ids[i] for i in selected],dict(selected_L_packets=5040,original_L_carrier=6300,source_cells=72,unused_cached_source_roles_deserialized=True,
        held_RX1_IQ_values_used=False,U_values_or_truth_used=False,V_values_used=False,target_access=False,source_derived_profile_not_IQ_or_hardware_identification=True)
