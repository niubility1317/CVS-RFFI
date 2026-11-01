"""Source-only descriptive RX/day geometry from an existing frozen F0 cache."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np

RUN='20261101-diagnostic-Restore-Fgeometry-RXday-s2026092701-r01'
CACHE='20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01'
GEOMETRY='20261101-diagnostic-Restore-learned-prior-physics-s2026092701-r01'
F_COMMIT='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)


def check_handles(m,g,r):
    expected=dict(schema='p3_restore_source_cache_v1',status='VALIDATED_ONCE',
        capsule_id='p3restore-LU392005-G10MIX01-clean-pairs-20261101-r01',
        split_id='392005-LsUs-source5rx-3day-eq1',count=63000,counts={'L_s':6300,'U_s':56700},
        TX_labels_read=0,OE_reads=0,V_reads=0,target_inputs=0,optimizer_updates=0)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Original L_s/U_s cache handles required')
    identity=m.get('F0_identity',{})
    if (identity.get('source_code_commit'),identity.get('source_row'),identity.get('epoch'),identity.get('updates'))!=(
        F_COMMIT,'main_split-repl-shuffled_time-3view-s2026092701',50,11100):raise ValueError('Original ground Tri-SIG-TIME SSL required')
    if (g.get('count'),g.get('source_only'),g.get('TX_labels_read'))!=(63000,True,0):raise ValueError('Source-only F0 geometry required')
    if (r.get('status'),r.get('source_feature_moment_rows'),r.get('F0_identity',{}).get('source_code_commit'))!=(
        'LEARNED_PRIOR_PHYSICS_COMPLETE',63000,F_COMMIT):raise ValueError('Frozen source geometry producer required')
    if any(r.get(k)!=0 for k in ['TX_labels_read','V_reads','OE_reads','target_inputs','optimizer_updates','model_state_updates']):
        raise ValueError('No target or updated-model geometry allowed')
    scale=np.asarray(g['scale'],np.float64);bands=np.asarray(g['bands'],np.int64)
    if scale.shape!=(160,) or not np.isfinite(scale).all() or not (scale>0).all():raise ValueError('Finite fixed VarF scales required')
    if bands.shape!=(4,40) or not np.array_equal(np.sort(bands.ravel()),np.arange(160)):raise ValueError('Fixed four variance bands required')
    return scale,bands


def domain_geometry(features,rx,day,scale,bands):
    z=np.asarray(features,np.float64);rx=np.asarray(rx);day=np.asarray(day);scale=np.asarray(scale,np.float64)
    if z.ndim!=2 or z.shape[0]<2:raise ValueError('Source feature matrix with multiple records required')
    if rx.shape!=(len(z),) or day.shape!=(len(z),) or scale.shape!=(z.shape[1],):raise ValueError('Aligned source dimensions required')
    if not np.isfinite(z).all() or not np.isfinite(scale).all() or not (scale>0).all():raise ValueError('Finite features/positive scales required')
    bands=np.asarray(bands,np.int64)
    if bands.ndim!=2 or not np.array_equal(np.sort(bands.ravel()),np.arange(z.shape[1])):raise ValueError('Disjoint exhaustive feature bands required')
    z=z-z.mean(0);total=np.mean(z*z,axis=0)
    if not total.sum()>0:raise ValueError('All-constant features have no defined explained fraction')
    def between(groups):
        unique,index,counts=np.unique(groups,axis=0,return_inverse=True,return_counts=True)
        sums=np.zeros((len(counts),z.shape[1]),np.float64);np.add.at(sums,index,z)
        variance=np.sum(sums*sums/counts[:,None],axis=0)/len(z)
        return variance,dict(groups=unique.tolist(),counts=counts.tolist())
    vr,rc=between(rx);vd,dc=between(day);vj,jc=between(np.column_stack((rx,day)))
    if any(np.any(v>total+1e-10*np.maximum(total,1)) for v in (vr,vd,vj)):
        raise ValueError('Between-group variance exceeded total variation')
    # Nested cell partitions make these exact conditional increments. They include
    # interactions; they are not additive pure RX/day main-effect estimates.
    increments={'RX':vr,'day':vd,'RX_day_joint':vj,
        'RX_and_interaction_added_given_day':np.maximum(vj-vd,0),
        'day_and_interaction_added_given_RX':np.maximum(vj-vr,0)}
    active=total>0
    per_dim={name:np.divide(v,total,out=np.full_like(total,np.nan),where=active) for name,v in increments.items()}
    def aggregate(take):
        t=total[take];w=1/scale[take]
        def ratio(v,weights):
            denom=float(np.sum(t*weights))
            return None if denom==0 else float(np.sum(v[take]*weights)/denom)
        return {name:dict(raw_energy_fraction=ratio(v,np.ones(len(t))),VarF_scaled_energy_fraction=ratio(v,w)) for name,v in increments.items()}
    summaries={'ALL':aggregate(np.arange(z.shape[1]))}
    summaries.update({f'variance_band{i}':aggregate(b) for i,b in enumerate(bands)})
    return dict(count=len(z),dimensions=z.shape[1],source_total_variance=total.tolist(),
        source_between_variance={k:v.tolist() for k,v in increments.items()},
        per_dimension_fraction={k:[None if not np.isfinite(x) else float(x) for x in v] for k,v in per_dim.items()},
        zero_variance_dimensions=np.where(~active)[0].tolist(),groups={'RX':rc,'day':dc,'RX_day_joint':jc},summaries=summaries,
        conditional_increments_include_interactions=True,marginal_RX_day_fractions_must_not_be_added=True)


def run(args):
    started=time.perf_counter()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only diagnostic requires empty CUDA visibility')
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual Git release identity required')
    folder=args.root/'runs'/RUN;folder.mkdir()
    write(folder/'launch.json',dict(pid=os.getpid(),CWD=os.getcwd(),argv=__import__('sys').argv,code_commit=args.commit))
    cache=args.root/'runs'/CACHE/'cache';geo=args.root/'runs'/GEOMETRY
    m=read(cache/'manifest.json');g=read(geo/'feature_geometry.json');r=read(geo/'results.json')
    scale,bands=check_handles(m,g,r)
    producer=read(cache.parent/'results.json')
    if producer.get('status')!='P3_RESTORE_SOURCE_CACHE_COMPLETE' or producer.get('smoke',{}).get('status')!='PASS':
        raise ValueError('Completed original feature producer with real-checkpoint smoke required')
    write(folder/'smoke.json',dict(status='PASS',kind='data_only_source_handle_smoke',
        upstream_real_checkpoint_source_smoke='PASS',new_checkpoint_loads=0,new_neural_forwards=0,target_inputs=0))
    print('SOURCE_FEATURE_HANDLES_PASS',flush=True)
    arrays={}
    try:
        for name in ('F0_features','rx','day','roles'):arrays[name]=np.load(cache/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if arrays['F0_features'].shape!=(63000,160) or arrays['F0_features'].dtype!=np.float32:raise ValueError('Original source fused160 array required')
        if not np.array_equal(arrays['roles'],np.asarray(['L_s']*6300+['U_s']*56700)):raise ValueError('Original hidden-label source roles required')
        if set(np.unique(arrays['rx']))!={1,3,4,6,8} or set(np.unique(arrays['day']))!={1,2,3}:raise ValueError('Original source RX/day sets required')
        result=domain_geometry(arrays['F0_features'],arrays['rx'],arrays['day'],scale,bands)
    finally:
        for value in arrays.values():value._mmap.close()
    reference=np.asarray(g['variance'],np.float64)
    result.update(status='SOURCE_FEATURE_DOMAIN_ANALYSIS_COMPLETE',run_id=RUN,code_commit=args.commit,
        cache_run=CACHE,geometry_run=GEOMETRY,F0_identity=m['F0_identity'],
        fixed_VarF_scale=scale.tolist(),fixed_variance_bands=bands.tolist(),
        canonical_cache_variance_vs_R07_moment_max_abs=float(np.max(np.abs(np.asarray(result['source_total_variance'])-reference))),
        different_clean_FP32_preparation_and_feature_batch_sizes=True,
        model_loads=0,new_neural_forwards=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,
        fitted_classifier_heads=0,default_DURA_changed=False,goal_complete=False,
        association_is_causal_nuisance_or_identity_partition=False,
        source_feature_energy_is_not_restoration_error_or_gradient_energy=True,seconds=time.perf_counter()-started)
    write(folder/'results.json',result);write(folder/'complete.json',dict(status=result['status'],count=63000,code_commit=args.commit,new_neural_forwards=0,target_inputs=0))
    print(json.dumps(dict(status=result['status'],count=result['count'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True)
    run(p.parse_args())
