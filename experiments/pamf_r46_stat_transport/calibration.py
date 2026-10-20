"""Frozen source parent/identity-child group statistics; no labels or clean."""
import json,time
import numpy as np
import torch
from . import design as d
from .normalization import StatisticGroupNorm,convert_normalization
from .tensor_io import array_tensor
from experiments.pamf_r46_gn_response.normalization import norm_links,PairedMoment,affine_fit
from experiments.pamf_r46_fold_compensate.calibration import validate_calibration as validate_fold
from experiments.pamf_r46_fold_compensate.models import compact_from_parent as full_fold
from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint as load_parent
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json

def aligned_links(parent,model,maps):
    links=norm_links(parent,model,maps)
    if len(links)!=168:raise ValueError('All168 group statistic locations required')
    for link in links:
        if not isinstance(link.target,StatisticGroupNorm):raise ValueError('Identical statistic-transport child graph required')
        quota=link.target.num_channels//4;old_quota=link.source.num_channels//4
        for group in range(4):
            if not torch.all(link.outputs[group*quota:(group+1)*quota]//old_quota==group):raise ValueError('Retained channels must preserve exact GN4 group quotas')
    return links

def group_values(value,groups,eps):
    if value.device.type!='cpu' or value.ndim!=3 or value.shape[1]%groups or not torch.isfinite(value).all():raise ValueError('Finite full CPU group input required')
    z=value.detach().double().reshape(value.shape[0],groups,-1);variance,mean=torch.var_mean(z,dim=-1,unbiased=False)
    return mean.T,(variance+eps).log().T

@torch.no_grad()
def collect(parent,model,maps,views,batch=8):
    if type(batch)!=int or batch<1 or any(v.training or any(p.requires_grad or p.device.type!='cpu' for p in v.parameters()) for v in (parent,model)):
        raise ValueError('Frozen eval CPU parent and identity child required')
    links=aligned_links(parent,model,maps)
    if any(torch.count_nonzero(v.target.stat_parameters) for v in links):raise ValueError('Only identity initial child may define calibration coordinates')
    stats={v.name:{k:PairedMoment(4) for k in ('mean','logvariance')} for v in links};cache={};seen=set();hooks=[];count=0
    snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in (parent,model)]
    try:
        for link in links:
            def teacher_hook(module,inputs,link=link):
                if link.name in cache:raise ValueError('One registered parent GN use per complete forward required')
                cache[link.name]=group_values(inputs[0],4,module.eps)
            def child_hook(module,inputs,link=link):
                if link.name not in cache or link.name in seen:raise ValueError('Aligned identity child group inputs required')
                values=group_values(inputs[0],4,module.eps)
                for key,x,y in zip(('mean','logvariance'),values,cache[link.name]):stats[link.name][key].update_columns(x,y)
                seen.add(link.name)
            hooks.extend((link.source.register_forward_pre_hook(teacher_hook),link.target.register_forward_pre_hook(child_hook)))
        for view in views:
            if view.device.type!='cpu' or view.ndim!=3 or view.shape[1:]!=(2,256) or not torch.isfinite(view).all():raise ValueError('Finite native received source IQ required')
            for start in range(0,len(view),batch):
                x=view[start:start+batch];cache.clear();seen.clear();a=parent(x);b=model(x)
                if a.shape!=x.shape or b.shape!=x.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all() or seen!=set(stats) or set(cache)!=seen:
                    raise ValueError('Full finite source parent/identity child forwards required')
                count+=len(x)
        if not count or any(any(p.grad is not None for p in m.parameters()) or any(not torch.equal(v,old[k]) for k,v in m.state_dict().items()) for m,old in zip((parent,model),snapshots)):
            raise ValueError('Source parent or identity child changed during statistics collection')
        fits={v.name:dict(outputs=v.outputs.tolist(),**{k:affine_fit(stats[v.name][k].value(),d.RIDGE_FRACTION,d.SLOPE_BOUNDS) for k in ('mean','logvariance')}) for v in links}
        if any(f[k]['sampled_channel_vectors']!=count for f in fits.values() for k in ('mean','logvariance')):raise ValueError('Exactly one complete group statistic pair per received source item required')
        return dict(stat_fits=fits,moments={name:{k:v.value() for k,v in group.items()} for name,group in stats.items()},
            statistic_modules=len(links),parent_G_inputs=count,identity_child_G_inputs=count,parent_unchanged=True,identity_child_unchanged=True)
    finally:
        for handle in hooks:handle.remove()

def validate_calibration(value):
    expected=dict(schema='R46_GN_stat_transport_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        fold_calibration_run=d.FOLD_CAL_RUN,fold_calibration_commit=d.FOLD_CAL_COMMIT,source_unique_IQ=128,L_s=64,U_s=64,received_views=['01','20'],
        sampling_seed=d.CAL_SEED,parent_G_inputs=256,identity_child_G_inputs=256,total_new_G_inputs=512,reused_fold_parent_G_inputs=256,
        ridge_fraction=d.RIDGE_FRACTION,slope_bounds=list(d.SLOPE_BOUNDS),statistic_modules=168,new_stat_parameters=2688,
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,parent_unchanged=True,identity_child_unchanged=True,calibration_shared_once=True)
    if not isinstance(value,dict) or any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed legal received-only group statistic calibration required')
    validate_fold(value.get('fold_calibration'));fits=value.get('stat_fits',{})
    if len(fits)!=168:raise ValueError('All168 fixed grouped statistic fits required')
    for item in fits.values():
        for key in ('mean','logvariance'):
            fit=item[key];a=np.asarray(fit.get('slope'),dtype=np.float64);b=np.asarray(fit.get('intercept'),dtype=np.float64)
            if a.shape!=(4,) or b.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a<.25) or np.any(a>4.) or fit.get('sampled_channel_vectors')!=256:
                raise ValueError('Four finite bounded full-group fits and exact source count required')
    return value

def prepare(args):
    if args.parent_action.name!=d.ACTION_RUN or args.compact_parent.name!=d.COMPACT_PARENT_RUN or args.output.name!=d.RUN or args.fold_calibration_run.name!=d.FOLD_CAL_RUN or args.fold_calibration_run.parent!=args.output.parent:
        raise ValueError('Fixed source/cache/parent/immutable output roots required before IO')
    target=args.output/'calibration.json'
    if target.exists():raise FileExistsError('Never overwrite the one shared statistic calibration')
    started=time.perf_counter()
    if read_json(args.fold_calibration_run/'launch.json')['commit']!=d.FOLD_CAL_COMMIT:raise ValueError('Pinned source-only folding method required')
    fold=validate_fold(read_json(args.fold_calibration_run/'calibration.json'));cache=args.parent_action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if roles.shape!=(63000,) or sum(roles=='L_s')!=6300 or sum(roles=='U_s')!=56700:raise ValueError('Original source L/U physical roles required')
    rng=np.random.default_rng(d.CAL_SEED);indices=np.sort(np.concatenate([rng.choice(np.flatnonzero(roles==role),64,replace=False) for role in ('L_s','U_s')]))
    rows=read_json(args.fold_calibration_run/'calibration_source_rows.json')
    if rows!=dict(indices=indices.tolist(),roles=roles[indices].tolist()):raise ValueError('Exactly fixed source calibration population required')
    views=[]
    for view in ('01','20'):
        array=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False)
        if array.shape!=(63000,2,256):raise ValueError('Original received source cache required')
        views.append(array_tensor(array[indices],'cpu'))
    parent,payload=load_parent(args.compact_parent/d.COMPACT_PARENT_ROW/d.KIND/'final.pt','cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=(d.COMPACT_PARENT_ROW,20,4940):raise ValueError('Human designated source-final parent required')
    del payload;parent.eval().requires_grad_(False);model=convert_normalization(full_fold(parent,'fold_full',fold)).eval().requires_grad_(False)
    maps={k:[torch.tensor(v,dtype=torch.int64) for v in levels] for k,levels in model.initialization_report['channel_maps'].items()}
    observed=collect(parent,model,maps,views,batch=8)
    result=dict(schema='R46_GN_stat_transport_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        fold_calibration_run=d.FOLD_CAL_RUN,fold_calibration_commit=d.FOLD_CAL_COMMIT,fold_calibration=fold,
        source_unique_IQ=128,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,total_new_G_inputs=512,reused_fold_parent_G_inputs=256,
        ridge_fraction=d.RIDGE_FRACTION,slope_bounds=list(d.SLOPE_BOUNDS),new_stat_parameters=2688,
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,calibration_shared_once=True,
        **{k:v for k,v in observed.items() if k!='moments'},seconds=time.perf_counter()-started)
    validate_calibration(result);write_json(target,result);write_json(args.output/'calibration_statistic_moments.json',observed['moments']);write_json(args.output/'calibration_source_rows.json',rows)
    print(json.dumps(dict(event='SOURCE_STATISTIC_TRANSPORT_COMPLETE',source_unique=128,parent_G_inputs=256,identity_child_G_inputs=256,modules=168,
        new_parameters=2688,TX_labels_read=0,F_reads=0,seconds=result['seconds'])),flush=True)
