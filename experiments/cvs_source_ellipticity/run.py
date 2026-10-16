import json,os,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .core import descriptor,cells,analyze
def source_statistic():
    import torch
    torch.set_num_threads(2)
    expected=d.read(d.SOURCE);prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
    if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['target_access'] or prepared['identity_updates']:raise ValueError('Existing source preparation differs')
    packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
    for actual in (packed['contract'],prepared['source_contract']):
        if any(actual.get(k)!=v for k,v in expected.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Reused original source roles differ')
    L=packed['labeled'];ids=L['ids'];all_L=set(ids)
    if len(ids)!=6300 or len(all_L)!=6300 or all_L!=set(expected['role_ids']['L_s']) or all_L&(set(expected['role_ids']['U_s'])|set(expected['role_ids']['V'])) or L['x'].shape!=(6300,2,256):raise ValueError('Original labeled source carrier required')
    selected=[i for i,r in enumerate(L['rx'].tolist()) if int(r) in d.FIT]
    reference=d.read(Path(d.REFERENCE)/'source_contract.json')
    if len(selected)!=5040 or {ids[i] for i in selected}!=set(reference['role_ids']['L_s']) or reference['held_RX']!=1 or reference['source_receivers']!=list(d.FIT):raise ValueError('Exact currentC0 fitL5040 required')
    # Only selected legal L IQ values enter this statistic. Source RX1/U/V remain unused.
    x=np.asarray(L['x'][selected].double().tolist(),dtype=np.float64)
    tx=np.asarray(L['y'][selected].tolist());rx=np.asarray(L['rx'][selected].tolist());day=np.asarray(L['day'][selected].tolist())
    z=descriptor(x)
    return z,tx,rx,day,dict(selected_L_packets=5040,original_L_carrier=6300,source_cells=72,unused_cached_source_roles_deserialized=True,
        held_RX1_IQ_values_used=False,U_values_or_truth_used=False,V_values_used=False,target_access=False,
        packet_ellipticity_min=float(z.min()),packet_ellipticity_max=float(z.max()),packet_ellipticity_mean=float(z.mean()))
def run():
    c=d.config(d.rows()[0]);d.validate(c)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA must be hidden')
    if d.BASE.exists():raise FileExistsError('Reconcile immutable existing output, no duplicate')
    d.BASE.mkdir();out=Path(c['output_root']);out.mkdir();started=time.time()
    d.write(d.BASE/'execution.json',dict(status='RUNNING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,GPU='',actual_code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),read_at=started))
    try:
        d.write(out/'resolved_config.json',c);z,tx,rx,day,roles=source_statistic();cell=cells(z,tx,rx,day);s,profile=analyze(cell)
        np.savez(out/'source_ellipticity_statistics.npz',packet_descriptor=z,TX=tx,RX=rx,day=day,cell_means=cell,RX_mean_profile=profile)
        s.update(config=c,roles=roles,wall_seconds=time.time()-started,source_IQ_values_inspected=5040,model_forwards=0,checkpoint_loads=0,optimizer_updates=0,head_fit_updates=0,new_models=0)
        d.write(out/'summary.json',s)
        d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',scientific_status=s['status'],rows=[c['row_id']],source_only=True,target_access=False,new_models=0,model_forwards=0,optimizer_updates=0,wall_seconds=s['wall_seconds'],automatic_training=False,goal_completed=False))
        print('COMPLETE sourceELLIP '+json.dumps({k:s[k] for k in ('status','checks','aggregate_risk_reduction','positive_folds','RX_pair_ellipticity_p95','wall_seconds')}),flush=True)
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),preserve_outputs=True,no_automatic_retry=True));raise
