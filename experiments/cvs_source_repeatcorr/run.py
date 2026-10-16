import os,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .core import describe,cells,analyze

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
    return x,tx,rx,day,dict(selected_L_packets=5040,original_L_carrier=6300,source_cells=72,unused_cached_source_roles_deserialized=True,
        held_RX1_IQ_values_used=False,U_values_or_truth_used=False,V_values_used=False,target_access=False,source_derived_profile_not_IQ_or_hardware_identification=True)

def quality_of(q,rx):
    return dict(valid_packets=int(q['rho_valid'].sum()),coherence_p10=float(np.quantile(q['coherence'],.1)),RX_median_residual_ratio=[float(np.median(q['residual_ratio'][rx==r])) for r in d.FIT])

def run():
    c=d.config(d.rows()[0]);d.validate(c)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only G0, CUDA must be hidden')
    if d.BASE.exists():raise FileExistsError('Reconcile immutable source repeatcorr root; no duplicate launch')
    d.BASE.mkdir();out=Path(c['output_root']);out.mkdir();started=time.time()
    d.write(d.BASE/'execution.json',dict(status='RUNNING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,GPU='',actual_training_code=(d.ROOT/'release_commit.txt').read_text().strip(),read_at=started))
    try:
        d.write(out/'resolved_config.json',c);x,tx,rx,day,roles=source_observations();z,q=describe(x);a=cells(z,tx,rx,day);summary,profile=analyze(a,quality_of(q,rx))
        np.savez(out/'source_repeatcorr_statistics.npz',packet_descriptor=z,TX=tx,RX=rx,day=day,cell_means=a,RX_mean_profile=profile,**q)
        summary.update(config=c,roles=roles,wall_seconds=time.time()-started,source_IQ_values_inspected=5040,
            undefined_rho_packets=5040-int(q['rho_valid'].sum()),coherence_median=float(np.median(q['coherence'])),residual_ratio_median=float(np.median(q['residual_ratio'])))
        d.write(out/'summary.json',summary)
        d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',scientific_status=summary['status'],source_only=True,target_access=False,rows=[c['row_id']],new_models=0,model_forwards=0,optimizer_updates=0,wall_seconds=summary['wall_seconds'],automatic_training=False,goal_completed=False))
        print('COMPLETE sourceRepeatCorr '+str({k:summary[k] for k in ('status','checks','aggregate_risk_reduction','positive_folds','median_RX_profile_absolute_rho','RX_pair_profile_distance_p95','wall_seconds')}),flush=True)
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),preserve_outputs=True,no_automatic_retry=True));raise
