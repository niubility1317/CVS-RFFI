"""One complete source-only CPU calibration; no F, clean IQ, TX labels or targets."""
import time
import numpy as np
import torch
from torch import nn
from . import design as d
from .models import context_links,grouped_links,group
from experiments.pamf_r46_fold_compensate.calibration import ChannelMoment
from experiments.pamf_r46_group_drop.tensor_io import array_tensor
from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint
from experiments.pamf_restoration.data import write_json


def metadata():
    return dict(schema='R46_whole_group_context_moments_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        source_unique_IQ=128,frozen_parent_IQ_inputs=256,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,
        positions_per_conv_input=d.CAL_POSITIONS,ridge_fraction=d.RIDGE_FRACTION,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,
        optimizer_updates=0,parent_unchanged=True,no_intercept=True,whole_context_groups=True,prior_copied_exactly=True,calibration_shared_once=True,
        context_moment_groups=d.CAL_MODULE_COUNT)


def validate_calibration(value):
    if not isinstance(value,dict) or any(value.get(k)!=v for k,v in metadata().items()):raise ValueError('Fixed complete source-only context calibration required')
    moments=value.get('moments',{})
    if not isinstance(moments,dict) or len(moments)!=d.CAL_MODULE_COUNT:raise ValueError('All103 context input moment groups required')
    for name,item in moments.items():
        members=item.get('members',[]);count=256*len(members)*(1 if name=='action.global_head' else d.CAL_POSITIONS)
        if (not members or len(members)!=len(set(members)) or item.get('sampled_channel_vectors')!=count
                or not name.startswith(('coarse_spectral.','coarse_temporal.','action.'))):raise ValueError('Complete context module exposure required')
    return value


@torch.no_grad()
def collect(parent,views,batch=d.CAL_BATCH):
    if parent.training or any(p.requires_grad or p.device.type!='cpu' for p in parent.parameters()) or type(batch)!=int or batch<1:
        raise ValueError('Frozen CPU eval parent and positive source batch required')
    model=group.compact_from_parent(parent,'group_preserve')
    maps={name:[torch.tensor(v,dtype=torch.int64) for v in levels] for name,levels in model.initialization_report['channel_maps'].items()}
    groups=[g for g in grouped_links(context_links(parent,model,maps)) if len(g[0].inputs)<g[0].source.weight.shape[1]]
    if len(groups)!=d.CAL_MODULE_COUNT:raise ValueError('Fixed103 full-context groups required')
    stats={g[0].name:ChannelMoment(g[0].source.weight.shape[1],d.CAL_POSITIONS) for g in groups};hooks=[];count=0
    before={k:v.detach().clone() for k,v in parent.state_dict().items()}
    try:
        for members in groups:
            stat=stats[members[0].name]
            for link in members:
                def observe(module,inputs,stat=stat):stat.update(inputs[0])
                hooks.append(link.source.register_forward_pre_hook(observe))
        for received in views:
            if received.device.type!='cpu' or received.ndim!=3 or received.shape[1:]!=(2,256) or not torch.isfinite(received).all():
                raise ValueError('Finite original source received IQ required')
            for start in range(0,len(received),batch):
                value=received[start:start+batch];restored=parent(value)
                if restored.shape!=value.shape or not torch.isfinite(restored).all():raise ValueError('Finite source calibration parent output required')
                count+=len(value)
        if not count or any(p.grad is not None for p in parent.parameters()) or any(not torch.equal(v,before[k]) for k,v in parent.state_dict().items()):
            raise ValueError('Calibration parent changed or no source observed')
        result={}
        for members in groups:
            name=members[0].name;stat=stats[name];expected=count*len(members)*(1 if isinstance(members[0].source,nn.Linear) else d.CAL_POSITIONS)
            if stat.count!=expected:raise ValueError('Every source context module exposure required')
            result[name]=dict(members=[v.name for v in members],second_moment=stat.value().tolist(),sampled_channel_vectors=stat.count)
        return dict(moments=result,frozen_parent_IQ_inputs=count)
    finally:
        for hook in hooks:hook.remove()


def prepare(args):
    if args.parent_action.name!=d.base.ACTION_RUN or args.compact_parent.name!=d.COMPACT_PARENT_RUN or args.output.name!=d.RUN:
        raise ValueError('Fixed source roots required before IO')
    target=args.output/'calibration.json'
    if target.exists():raise FileExistsError('Never overwrite the one shared source calibration')
    started=time.perf_counter();cache=args.parent_action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if roles.shape!=(63000,) or sum(roles=='L_s')!=6300 or sum(roles=='U_s')!=56700:raise ValueError('Original source role counts required')
    rng=np.random.default_rng(d.CAL_SEED);indices=np.sort(np.concatenate([rng.choice(np.flatnonzero(roles==role),d.CAL_PER_ROLE,replace=False) for role in ('L_s','U_s')]))
    views=[]
    for epoch in ('01','20'):
        source=np.load(cache/'epochs'/epoch/'received.npy',mmap_mode='r',allow_pickle=False)
        if source.shape!=(63000,2,256):raise ValueError('Original received cache shape required')
        views.append(array_tensor(source[indices],'cpu'))
    parent,payload=load_checkpoint(args.compact_parent/d.COMPACT_PARENT_ROW/d.base.KIND/'final.pt','cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=(d.COMPACT_PARENT_ROW,20,4940):raise ValueError('Only fixed source-final human parent required')
    parent.eval().requires_grad_(False);collected=collect(parent,views)
    result=dict(metadata(),**collected,seconds=time.perf_counter()-started);validate_calibration(result);write_json(target,result)
    write_json(args.output/'calibration_source_rows.json',dict(indices=indices.tolist(),roles=roles[indices].tolist()))
    print('SOURCE_COMPLETE_CONTEXT_MOMENTS 128source/256parentG/103groups/0cleanTXFVtarget',flush=True)
