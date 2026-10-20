"""One shared CPU fit of existing GN affine values; no labels/clean/F/V/target."""
import json,time
import numpy as np
import torch
from . import design as d
from .tensor_io import array_tensor
from .normalization import collect
from experiments.pamf_r46_fold_compensate.calibration import validate_calibration as validate_fold
from experiments.pamf_r46_fold_compensate.models import compact_from_parent as full_fold
from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint as load_parent
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json

def validate_calibration(value):
    expected=dict(schema='R46_GN_response_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        fold_calibration_run=d.FOLD_CAL_RUN,fold_calibration_commit=d.FOLD_CAL_COMMIT,source_unique_IQ=128,L_s=64,U_s=64,
        received_views=['01','20'],sampling_seed=d.CAL_SEED,positions_per_GN_output=16,parent_IQ_inputs=256,student_IQ_inputs=256,
        total_new_G_inputs=512,reused_fold_parent_G_inputs=256,GN_ridge_fraction=d.GN_RIDGE_FRACTION,GN_slope_bounds=list(d.GN_SLOPE_BOUNDS),
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,parent_unchanged=True,initial_student_unchanged=True,
        calibration_shared_once=True,GN_affine_has_intercept=True,GN_modules=d.GN_MODULES)
    if not isinstance(value,dict) or any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed legal received-only GN calibration required')
    validate_fold(value.get('fold_calibration'))
    fits=value.get('GN_fits',{})
    if not fits or value.get('GN_modules')!=len(fits):raise ValueError('All existing GN module fits required')
    for item in fits.values():
        slope=np.asarray(item.get('slope'),dtype=np.float64);bias=np.asarray(item.get('intercept'),dtype=np.float64)
        if slope.ndim!=1 or not len(slope) or bias.shape!=slope.shape or len(item.get('outputs',[]))!=len(slope) or not np.isfinite(slope).all() or not np.isfinite(bias).all() or np.any(slope<.25) or np.any(slope>4) or item.get('sampled_channel_vectors')!=4096:
            raise ValueError('Fixed finite bounded paired GN fit required')
    return value

def prepare(args):
    if args.parent_action.name!=d.ACTION_RUN or args.compact_parent.name!=d.COMPACT_PARENT_RUN or args.output.name!=d.RUN or args.fold_calibration_run.name!=d.FOLD_CAL_RUN or args.fold_calibration_run.parent!=args.output.parent:
        raise ValueError('Fixed legal source and calibration roots required before IO')
    target=args.output/'calibration.json'
    if target.exists():raise FileExistsError('One shared immutable GN fit;never overwrite')
    started=time.perf_counter()
    if read_json(args.fold_calibration_run/'launch.json')['commit']!=d.FOLD_CAL_COMMIT:raise ValueError('Pinned source folding method required')
    folding=validate_fold(read_json(args.fold_calibration_run/'calibration.json'))
    cache=args.parent_action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if roles.shape!=(63000,) or sum(roles=='L_s')!=6300 or sum(roles=='U_s')!=56700:raise ValueError('Original source L/U roles required')
    rng=np.random.default_rng(d.CAL_SEED);indices=np.sort(np.concatenate([rng.choice(np.flatnonzero(roles==role),64,replace=False) for role in ('L_s','U_s')]))
    existing=read_json(args.fold_calibration_run/'calibration_source_rows.json')
    if existing!=dict(indices=indices.tolist(),roles=roles[indices].tolist()):raise ValueError('Reuse exactly fixed R55 source received population')
    views=[]
    for view in ('01','20'):
        array=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False)
        if array.shape!=(63000,2,256):raise ValueError('Original received source epoch required')
        views.append(array_tensor(array[indices],'cpu'))
    parent,payload=load_parent(args.compact_parent/d.COMPACT_PARENT_ROW/d.KIND/'final.pt','cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=(d.COMPACT_PARENT_ROW,20,4940):raise ValueError('Fixed human source-final parent required')
    del payload;parent.eval().requires_grad_(False);model=full_fold(parent,'fold_full',folding).eval().requires_grad_(False)
    maps={k:[torch.tensor(v,dtype=torch.int64) for v in levels] for k,levels in model.initialization_report['channel_maps'].items()}
    observed=collect(parent,model,maps,views,batch=8)
    result=dict(schema='R46_GN_response_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        fold_calibration_run=d.FOLD_CAL_RUN,fold_calibration_commit=d.FOLD_CAL_COMMIT,fold_calibration=folding,
        source_unique_IQ=128,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,positions_per_GN_output=16,
        total_new_G_inputs=512,reused_fold_parent_G_inputs=256,GN_ridge_fraction=d.GN_RIDGE_FRACTION,GN_slope_bounds=list(d.GN_SLOPE_BOUNDS),
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,calibration_shared_once=True,GN_affine_has_intercept=True,
        **{k:v for k,v in observed.items() if k!='moments'},seconds=time.perf_counter()-started)
    validate_calibration(result);write_json(target,result);write_json(args.output/'calibration_GN_moments.json',observed['moments'])
    write_json(args.output/'calibration_source_rows.json',existing)
    print(json.dumps(dict(event='SOURCE_GN_RESPONSE_COMPLETE',source_unique=128,parent_G_inputs=256,student_G_inputs=256,
        GN_modules=result['GN_modules'],TX_labels_read=0,F_reads=0,seconds=result['seconds'])),flush=True)
