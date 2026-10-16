import json,os,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .core import cells,analyze

def source_descriptors():
    import torch
    torch.set_num_threads(2)
    expected=d.read(d.SOURCE);prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
    if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['target_access'] or prepared['identity_updates']:raise ValueError('Existing source preparation differs')
    packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
    for actual in (packed['contract'],prepared['source_contract']):
        if any(actual.get(k)!=v for k,v in expected.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Reused original source carrier roles differ')
    L=packed['labeled'];ids=L['ids'];all_L=set(ids)
    if len(ids)!=6300 or len(all_L)!=6300 or all_L!=set(expected['role_ids']['L_s']) or all_L&(set(expected['role_ids']['U_s'])|set(expected['role_ids']['V'])) or L['x'].shape!=(6300,2,256):raise ValueError('Original labeled source carrier required')
    selected=[i for i,r in enumerate(L['rx'].tolist()) if int(r) in d.FIT]
    selected_ids={ids[i] for i in selected};reference=d.read(Path(d.REFERENCE)/'source_contract.json')
    if len(selected)!=5040 or selected_ids!=set(reference['role_ids']['L_s']) or reference['held_RX']!=1 or reference['source_receivers']!=list(d.FIT):raise ValueError('Exact current C0 source fitL5040 required')
    # Only selected legal L values enter any statistic. Other cached roles and RX1 remain unused.
    x=L['x'][selected].double();tx=L['y'][selected].tolist();rx=L['rx'][selected].tolist();day=L['day'][selected].tolist()
    if not bool(torch.isfinite(x).all()):raise ValueError('Finite source L IQ required')
    power=x.square().sum(1).mean(1)
    if bool((power<=0).any()):raise ValueError('Positive original source IQ power required')
    z=np.asarray((x.mean(2)/power.sqrt()[:,None]).tolist(),dtype=np.float64)
    return z,np.asarray(tx),np.asarray(rx),np.asarray(day),dict(selected_L_packets=5040,original_L_carrier=6300,source_cells=72,
        unused_cached_source_roles_deserialized=True,held_RX1_IQ_values_used=False,U_values_or_truth_used=False,V_values_used=False,target_access=False,
        packet_power_min=float(power.min()),packet_power_max=float(power.max()),DC_energy_fraction_mean=float(np.square(z).sum(1).mean()))

def old_DC_bound():
    a=d.read(Path(d.REFERENCE)/'actual_native_args.json');a=a.get('args',a);base=d.read(Path(d.REFERENCE)/'ordinary_base_config.json')
    history=d.read(Path(d.REFERENCE)/'mechanism_progress.json')['records'];record=next(v for v in history if v['epoch']==13)
    p=record['ordinary']['probabilities']['p_dc_offset']
    if not a['use_aug'] or not base['p_dc_offset']>0 or not p>0 or a['aug_dc_offset_max']!=base['dc_offset_max']:raise ValueError('Actual old ordinary active DC metadata required')
    return float(base['dc_offset_max']*p/base['p_dc_offset'])

def run():
    c=d.config(d.rows()[0]);d.validate(c)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source statistics must not expose CUDA')
    if d.BASE.exists():raise FileExistsError('Reconcile existing immutable DC output, no duplicate launch')
    d.BASE.mkdir();out=Path(c['output_root']);out.mkdir();started=time.time()
    d.write(d.BASE/'execution.json',dict(status='RUNNING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,GPU='',
        actual_training_code=(d.ROOT/'release_commit.txt').read_text().strip(),read_at=started))
    try:
        d.write(out/'resolved_config.json',c);z,tx,rx,day,roles=source_descriptors();bound=old_DC_bound();cell=cells(z,tx,rx,day);summary,profile=analyze(cell,bound)
        np.savez(out/'source_DC_statistics.npz',packet_descriptor=z,TX=tx,RX=rx,day=day,cell_means=cell,RX_mean_profile=profile)
        summary.update(config=c,roles=roles,wall_seconds=time.time()-started,source_IQ_values_inspected=5040,
            model_forwards=0,checkpoint_loads=0,optimizer_updates=0,head_fit_updates=0,new_models=0)
        d.write(out/'summary.json',summary)
        d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',scientific_status=summary['status'],source_only=True,target_access=False,
            rows=[c['row_id']],new_models=0,model_forwards=0,optimizer_updates=0,wall_seconds=summary['wall_seconds'],automatic_training=False,goal_completed=False))
        print('COMPLETE sourceDC '+json.dumps({k:summary[k] for k in ('status','checks','aggregate_risk_reduction','positive_folds','RX_pair_DC_component_p95','old_ordinary_DC_component_max','wall_seconds')}),flush=True)
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),preserve_outputs=True,no_automatic_retry=True));raise
