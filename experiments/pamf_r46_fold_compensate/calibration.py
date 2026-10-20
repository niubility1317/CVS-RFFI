"""One shared CPU calibration from original source received IQ only."""
import json,time
import numpy as np
import torch
from torch import nn
from . import design as d
from .tensor_io import array_tensor
from .compensation import grouped_links,make_links,reconstruction
from .models import weight_initialization
from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint as load_parent
from experiments.pamf_restoration.data import write_json

class ChannelMoment:
    def __init__(self,channels,positions=d.CAL_POSITIONS):
        if type(channels)!=int or channels<1 or type(positions)!=int or positions<2:raise ValueError('Positive channel and spatial budget required')
        self.sum=torch.zeros(channels,channels,dtype=torch.float64);self.count=0;self.positions=positions

    @torch.no_grad()
    def update(self,value):
        if value.device.type!='cpu' or value.ndim not in (2,3) or value.shape[1]!=len(self.sum) or not torch.isfinite(value).all():raise ValueError('Finite CPU Conv/Linear input required')
        if value.ndim==3:
            if value.shape[-1]<self.positions:raise ValueError('Distinct fixed spatial positions required')
            selected=torch.arange(self.positions)*(value.shape[-1]-1)//(self.positions-1)
            values=value.index_select(-1,selected).permute(1,0,2).reshape(value.shape[1],-1)
        else:values=value.T
        values=values.detach().double();self.sum+=values@values.T;self.count+=values.shape[1]

    def value(self):
        if not self.count:raise ValueError('Observed source inputs required')
        return self.sum/self.count

@torch.no_grad()
def collect(parent,views,batch=d.CAL_BATCH):
    if parent.training or any(p.requires_grad or p.device.type!='cpu' for p in parent.parameters()) or type(batch)!=int or batch<1:raise ValueError('Frozen eval CPU parent and positive calibration batch required')
    model,maps=weight_initialization(parent)
    groups=[g for g in grouped_links(make_links(parent,model,maps)) if len(g[0].inputs)<g[0].source.weight.shape[1]]
    stats={g[0].name:ChannelMoment(g[0].source.weight.shape[1]) for g in groups};hooks=[];count=0
    before={k:v.detach().clone() for k,v in parent.state_dict().items()}
    try:
        for group in groups:
            stat=stats[group[0].name]
            for link in group:
                def observe(module,inputs,stat=stat):stat.update(inputs[0])
                hooks.append(link.source.register_forward_pre_hook(observe))
        for received in views:
            if received.device.type!='cpu' or received.ndim!=3 or received.shape[1:]!=(2,256) or not torch.isfinite(received).all():raise ValueError('Finite source received IQ2x256 required')
            for start in range(0,len(received),batch):
                x=received[start:start+batch];output=parent(x)
                if output.shape!=x.shape or not torch.isfinite(output).all():raise ValueError('Finite complete parent forward required')
                count+=len(x)
        if not count or any(p.grad is not None for p in parent.parameters()) or any(not torch.equal(v,before[k]) for k,v in parent.state_dict().items()):raise ValueError('Frozen source parent changed or no observations')
        projections={};moments={}
        for group in groups:
            link=group[0];stat=stats[link.name];gram=stat.value();projection,fit=reconstruction(gram,link.inputs,d.RIDGE_FRACTION)
            expected=count*len(group)*(1 if isinstance(link.source,nn.Linear) else d.CAL_POSITIONS)
            if stat.count!=expected:raise ValueError('Shared module uses or source position counts differ')
            projections[link.name]=dict(members=[v.name for v in group],inputs=link.inputs.tolist(),outputs=link.outputs.tolist(),
                projection=projection.tolist(),sampled_channel_vectors=stat.count,**fit)
            moments[link.name]=dict(second_moment=gram.tolist(),sampled_channel_vectors=stat.count)
        return dict(projections=projections,moments=moments,channel_maps={name:[v.tolist() for v in levels] for name,levels in maps.items()},
            frozen_parent_IQ_inputs=count,folded_unique_weights=len(groups),folded_module_uses=sum(len(g) for g in groups))
    finally:
        for handle in hooks:handle.remove()

def validate_calibration(value):
    expected=dict(schema='R46_source_channel_reconstruction_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        source_unique_IQ=128,frozen_parent_IQ_inputs=256,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,
        positions_per_conv_input=d.CAL_POSITIONS,ridge_fraction=d.RIDGE_FRACTION,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,
        optimizer_updates=0,parent_unchanged=True,no_intercept=True,weight_only_maps=True,calibration_shared_once=True)
    if not isinstance(value,dict) or any(value.get(k)!=v for k,v in expected.items()) or not isinstance(value.get('projections'),dict) or not value['projections']:raise ValueError('Fixed legal source reconstruction metadata required')
    maps=value.get('channel_maps',{})
    if set(maps)!={'coarse_spectral','coarse_temporal','action','prior0','prior1','prior2'} or any(maps['prior0']!=maps['prior'+str(i)] for i in (1,2)):raise ValueError('Fixed common prior maps required')
    if value.get('folded_unique_weights')!=len(value['projections']) or value.get('folded_module_uses')!=sum(len(p['members']) for p in value['projections'].values()):raise ValueError('Complete calibrated weight groups required')
    return value

def prepare(args):
    if args.parent_action.name!=d.ACTION_RUN or args.compact_parent.name!=d.COMPACT_PARENT_RUN or args.output.name!=d.RUN:raise ValueError('Fixed source roots required before IO')
    target=args.output/'calibration.json'
    if target.exists():raise FileExistsError('One shared calibration;never overwrite')
    started=time.perf_counter();cache=args.parent_action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if roles.shape!=(63000,) or sum(roles=='L_s')!=6300 or sum(roles=='U_s')!=56700:raise ValueError('Original source L/U roles required')
    rng=np.random.default_rng(d.CAL_SEED);indices=np.sort(np.concatenate([rng.choice(np.flatnonzero(roles==role),d.CAL_PER_ROLE,replace=False) for role in ('L_s','U_s')]))
    views=[]
    for view in ('01','20'):
        array=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False)
        if array.shape!=(63000,2,256):raise ValueError('Original received source epoch required')
        views.append(array_tensor(array[indices],'cpu'))
    model,payload=load_parent(args.compact_parent/d.COMPACT_PARENT_ROW/d.KIND/'final.pt','cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=(d.COMPACT_PARENT_ROW,20,4940):raise ValueError('Fixed source-final human parent required')
    del payload;model.eval().requires_grad_(False);collected=collect(model,views)
    result=dict(schema='R46_source_channel_reconstruction_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        source_unique_IQ=128,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,positions_per_conv_input=d.CAL_POSITIONS,
        ridge_fraction=d.RIDGE_FRACTION,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,
        parent_unchanged=True,no_intercept=True,weight_only_maps=True,calibration_shared_once=True,
        **{k:v for k,v in collected.items() if k!='moments'},seconds=time.perf_counter()-started)
    validate_calibration(result);write_json(target,result)
    write_json(args.output/'calibration_channel_moments.json',collected['moments'])
    write_json(args.output/'calibration_source_rows.json',dict(indices=indices.tolist(),roles=roles[indices].tolist()))
    print(json.dumps(dict(event='SOURCE_CHANNEL_RECONSTRUCTION_COMPLETE',source_unique=128,frozen_G_inputs=256,
        folded_unique_weights=result['folded_unique_weights'],folded_module_uses=result['folded_module_uses'],TX_labels_read=0,F_reads=0,seconds=result['seconds'])),flush=True)
