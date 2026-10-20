import json,time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .tensor_io import array_tensor
from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint as load_parent
from experiments.pamf_restoration.data import write_json

def cores(model):
    return dict(coarse_spectral=model.frozen.coarse.spectral,coarse_temporal=model.frozen.coarse.temporal,action=model.frozen.estimator.core,
        **{'prior'+str(i):core for i,core in enumerate(model.prior)})

def collect_scores(model,clean,views,*,batch=8,draws=2,seed=d.CAL_PROJECTION_SEED):
    if clean.ndim!=3 or clean.shape[1:]!=(2,256) or not torch.isfinite(clean).all() or any(v.shape!=clean.shape or not torch.isfinite(v).all() for v in views):raise ValueError('Matched finite source IQ required')
    if any(p.requires_grad or p.device.type!='cpu' for p in model.parameters()) or batch<1 or draws<1:raise ValueError('Frozen CPU parent and positive calibration budget required')
    model.eval();state={k:v.detach().clone() for k,v in model.state_dict().items()};interfaces=cores(model);activations={};hooks=[]
    totals={method:{name:[torch.zeros(block[0].net[0].out_channels,dtype=torch.float64) for block in core.encoder.blocks] for name,core in interfaces.items()} for method in d.ARMS}
    for name,core in interfaces.items():
        for level,block in enumerate(core.encoder.blocks):
            def capture(module,inputs,output,key=(name,level)):
                if key in activations:raise ValueError('Each core encoder level must execute exactly once')
                activations[key]=output
            hooks.append(block.register_forward_hook(capture))
    generator=torch.Generator(device='cpu').manual_seed(seed);count=0
    try:
        for view in views:
            for start in range(0,len(clean),batch):
                activations.clear();received=view[start:start+batch].detach().clone().requires_grad_(True);output=model(received)
                if output.shape!=received.shape or not torch.isfinite(output).all() or len(activations)!=24:raise ValueError('Complete six-core frozen parent graph required')
                keys=list(activations);values=tuple(activations[k] for k in keys)
                for (name,level),value in zip(keys,values):totals['activation_energy'][name][level]+=value.detach().double().square().mean(-1).sum(0)
                energy=clean[start:start+batch].detach().square().mean((1,2)).clamp_min(1e-4)
                for draw in range(draws):
                    signs=torch.randint(0,2,output.shape,generator=generator,dtype=torch.int64).to(output.dtype)*2-1
                    projected=(output*signs/energy.sqrt()[:,None,None]/(512**.5)).sum()
                    gradients=torch.autograd.grad(projected,values,retain_graph=draw+1<draws,allow_unused=False)
                    for (name,level),value,gradient in zip(keys,values,gradients):
                        gain=(value.detach().double()*gradient.detach().double()).sum(-1)
                        totals['projection_sensitivity'][name][level]+=gain.square().sum(0)
                count+=len(received)
        if any(p.grad is not None for p in model.parameters()) or any(not torch.equal(v,state[k]) for k,v in model.state_dict().items()):raise ValueError('Calibration changed frozen parent')
        for method,groups in totals.items():
            for name,levels in groups.items():
                for level,value in enumerate(levels):
                    value/=count*(draws if method=='projection_sensitivity' else 1)
                    if not torch.isfinite(value).all() or (value<0).any():raise ValueError('Finite source channel scores required')
        return totals,count
    finally:
        for hook in hooks:hook.remove()

def make_maps(scores):
    from .models import selected_channels
    result={}
    for method,groups in scores.items():
        shared=[sum(groups['prior'+str(i)][level] for i in range(3))/3 for level in range(4)]
        maps={}
        for name,levels in groups.items():
            target=d.SPECTRAL_WIDTHS[method] if name=='coarse_spectral' else d.WIDTHS[method]
            use=shared if name.startswith('prior') else levels
            maps[name]=[selected_channels(score,width).tolist() for score,width in zip(use,target)]
        result[method]=maps
    return result

def validate_calibration(value):
    expected=dict(schema='R46_source_channel_scores_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        source_unique_IQ=128,frozen_parent_IQ_inputs=256,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,projection_seed=d.CAL_PROJECTION_SEED,
        projection_draws=2,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True)
    if any(value.get(k)!=v for k,v in expected.items()) or set(value.get('maps',{}))!=set(d.ARMS):raise ValueError('Fixed legal source calibration metadata required')
    expected_keys={'coarse_spectral','coarse_temporal','action','prior0','prior1','prior2'}
    for method,maps in value['maps'].items():
        if set(maps)!=expected_keys or any(maps['prior0']!=maps['prior'+str(i)] for i in (1,2)):raise ValueError('All cores and common prior maps required')
    return value

def prepare(args):
    if args.parent_action.name!=d.ACTION_RUN or args.compact_parent.name!=d.COMPACT_PARENT_RUN or args.output.name!=d.RUN:raise ValueError('Fixed source paths required before calibration IO')
    target=args.output/'calibration.json'
    if target.exists():raise FileExistsError('Controller calibrates once;never overwrite')
    started=time.perf_counter();cache=args.parent_action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if roles.shape!=(63000,) or sum(roles=='L_s')!=6300 or sum(roles=='U_s')!=56700:raise ValueError('Original source L/U only;no V or TX labels')
    rng=np.random.default_rng(d.CAL_SEED);indices=np.sort(np.concatenate([rng.choice(np.flatnonzero(roles==role),64,replace=False) for role in ('L_s','U_s')]))
    clean=array_tensor(np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)[indices],'cpu')
    views=[array_tensor(np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False)[indices],'cpu') for view in ('01','20')]
    model,payload=load_parent(args.compact_parent/d.COMPACT_PARENT_ROW/d.KIND/'final.pt','cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=(d.COMPACT_PARENT_ROW,20,4940):raise ValueError('Only fixed source-final R46 parent allowed')
    del payload;model.eval().requires_grad_(False);scores,count=collect_scores(model,clean,views)
    if count!=256:raise ValueError('Exactly256 frozen source inputs required')
    result=dict(schema='R46_source_channel_scores_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,source_unique_IQ=128,frozen_parent_IQ_inputs=256,
        L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,projection_seed=d.CAL_PROJECTION_SEED,projection_draws=2,
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,
        maps=make_maps(scores),scores={m:{name:[v.tolist() for v in levels] for name,levels in groups.items()} for m,groups in scores.items()},seconds=time.perf_counter()-started)
    validate_calibration(result);write_json(target,result)
    write_json(args.output/'calibration_source_rows.json',dict(indices=indices.tolist(),roles=roles[indices].tolist()))
    print(json.dumps(dict(event='SOURCE_CHANNEL_CALIBRATION_COMPLETE',source_unique=128,frozen_G_inputs=256,TX_labels_read=0,F_reads=0,seconds=result['seconds'])),flush=True)
