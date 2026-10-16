"""Existing legal L only; no model, receiver replacement or target access."""
import os
from pathlib import Path
import sys
import time
import numpy as np
from . import design as d
from .core import observe,crossfit


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source diagnostic required')
    import torch
    torch.set_num_threads(2)
    c=d.config();d.validate(c);out=Path(d.BASE);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
    try:
        prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
        if prepared.get('status')!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared.get('target_access') is not False or prepared.get('identity_updates')!=0:raise ValueError('Existing source preparation differs')
        packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False);contract=d.read(d.SOURCE)
        for actual in (packed['contract'],prepared['source_contract']):
            if any(actual.get(k)!=v for k,v in contract.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Existing source role contract differs')
        L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
        ids=set(L['ids'])
        if len(ids)!=6300 or len(L['ids'])!=6300 or ids!=set(contract['role_ids']['L_s']) or ids&(set(contract['role_ids']['U_s'])|set(contract['role_ids']['V'])):raise ValueError('Original legal L6300 required')
        mask=L['rx']!=1
        x=np.asarray(L['x'][mask].tolist(),dtype=np.float64)
        tx,rx,day=(np.asarray(L[k][mask].tolist(),dtype=np.int64) for k in ('y','rx','day'))
        fit_ids=[i for i,keep in zip(L['ids'],mask.tolist()) if keep];del L
        if len(fit_ids)!=5040 or len(set(fit_ids))!=5040:raise ValueError('Exact fourRX L5040 required')
        print('INPUT PASS L5040; RX1/U/V not statistically used; no model/checkpoint/target',flush=True)
        d.write(out/'resolved_config.json',dict(config=c,commit=(d.ROOT/'release_commit.txt').read_text().strip(),python=sys.executable,torch_version=torch.__version__,numpy_version=np.__version__,source_roles_exact_match=True,fit_L_unique=5040,fit_RX=sorted(set(rx.tolist())),old_cache_other_roles_and_features_deserialized_but_never_used=True,held_RX1_used=False,network_constructed=False,checkpoint_access=False,target_constructed=False))
        observation=observe(x);result=crossfit(observation['slope_db'],tx,rx,day)
        median_r2=float(np.median(observation['linear_R2']));median_abs=float(np.median(np.abs(observation['slope_db'])))
        result['conditions'].update(median_packet_R2=median_r2>=.8,nontrivial_slope=median_abs>=.01,RMS_slope_retained=observation['max_RMS_slope_difference']<=1e-10)
        result['decision']='PASS' if all(result['conditions'].values()) else 'FAIL'
        result.update(median_packet_R2=median_r2,median_abs_slope_db=median_abs,max_RMS_slope_difference=observation['max_RMS_slope_difference'],
                      abs_slope_quantiles_db={str(q):float(np.quantile(np.abs(observation['slope_db']),q)) for q in (0,.5,.95,1)},
                      source_only=True,target_access=False,IQ_packets_statistically_used=5040,network_forwards=0,optimizer_steps=0)
        d.write(out/'gain_crossfit.json',result);d.write(out/'fit_L_ids.json',fit_ids)
        cells=[]
        for r in (3,4,6,8):
            for day_value in (1,2,3):
                for t in range(6):
                    take=(rx==r)&(day==day_value)&(tx==t)
                    cells.append(dict(RX=r,day=day_value,TX=t,count=int(take.sum()),median_abs_slope_db=float(np.median(np.abs(observation['slope_db'][take]))),median_signed_slope_db=float(np.median(observation['slope_db'][take])),median_R2=float(np.median(observation['linear_R2'][take]))))
        d.write(out/'gain_cells.json',cells)
        d.write(out/'packet_observations.json',dict(slope_db=observation['slope_db'].tolist(),linear_R2=observation['linear_R2'].tolist(),cycle_levels_centered_db=observation['cycle_levels_centered_db'].tolist()))
        complete=dict(status='ANALYZED',scientific_decision=result['decision'],fit_L_packets=5040,source_cells=72,folds=18,network_forwards=0,checkpoint_loads=0,optimizer_steps=0,U_used=False,V_used=False,held_RX1_used=False,target_access=False,automatic_training=False,seconds=time.perf_counter()-start)
        d.write(out/'completion.json',complete);print('COMPLETE '+str(complete),flush=True)
    except Exception as e:
        d.write(out/'failure.json',dict(status='FAILED',error=repr(e),no_automatic_retry=True));raise


if __name__=='__main__':run()
