"""Source-only GPU comparison of no-grad feature reference caching."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from . import feature_response_reuse_drift_probe as drift
from . import feature_response_reference_cache_probe as cached_method
from . import physical_feature_gradient_probe as controller
from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .covariance_detail import clip_branch_gradients
from .covariance_model import source_action_loss
from .feature_response_reuse_probe import cache_source_features, frozen_eval
from .objective import PairedObjective
from .response_joint_prior import selected_perturbation
from .source_pair import read, validate_source
from .variance_trial import load_scale

RUN = '20261108-diagnostic-Restore-FeatureReferenceGPU-sourceLU-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.feature_reference_gpu_probe'
RTOL, ATOL = 1e-5, 1e-6


def compare_tensors(left, right):
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError('Matching finite tensors required')
    return dict(max_abs=float((left-right).abs().max()),
                notclose=int((~torch.isclose(left,right,rtol=RTOL,atol=ATOL)).sum()),
                elements=left.numel())


def compare_named(left, right):
    if left.keys()!=right.keys(): raise ValueError('Matching parameter names required')
    stats=[]
    for key in left:
        a,b=left[key],right[key]
        if (a is None)!=(b is None): raise ValueError('Parameter gradient presence mismatch')
        if a is not None: stats.append(compare_tensors(a,b))
    if not stats: raise ValueError('Expected nonempty gradients')
    return dict(max_abs=max(v['max_abs'] for v in stats),
                notclose=sum(v['notclose'] for v in stats),
                elements=sum(v['elements'] for v in stats),tensors=len(stats))


def run_case(state, reader, clean, trace, indices, target, cache, scale, epoch):
    if epoch not in (1,3) or len(clean)!=256: raise ValueError('Fixed 256 source records and odd epoch required')
    device=clean.device; values=[]
    for use_cache in (False,True):
        model=drift.feature.factory(drift.feature.ARM).to(device).train()
        model.load_state_dict(state,strict=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        objective=PairedObjective().to(device); input_gradients=[]; step_loss=0.
        torch.cuda.synchronize();started=time.perf_counter()
        for start in range(0,256,32):
            sl=slice(start,start+32); batch={k:v[sl] for k,v in trace.items()}
            capture=drift.ReaderCapture(reader)
            if use_cache:
                base,aux,detail=cached_method.cached_readout(model,capture,batch['received'],clean[sl],
                    target[sl],scale,objective,indices[sl],epoch,batch,cache[sl])
            else:
                base,aux,detail,_=drift.feature.response_readout(model,capture,batch['received'],clean[sl],
                    target[sl],scale,objective,source_indices=indices[sl],epoch=epoch,
                    priors=None,assignment=None,source_trace=batch)
            physical,_=source_action_loss(aux,batch['kernel'],batch['coefficients'],batch['variance'])
            loss=(physical+detail).sum()/256
            if len(capture.inputs)!=2 or not torch.isfinite(loss) or aux['relative_solve_residual'].max()>1e-8:
                raise ValueError('Finite full production path and two differentiable F0 inputs required')
            loss.backward();step_loss+=float(loss.detach())
            input_gradients.extend(v.grad.detach().cpu().clone() for v in capture.inputs)
        gradients={k:None if p.grad is None else p.grad.detach().cpu().clone() for k,p in model.named_parameters()}
        clip_branch_gradients(model);optimizer.step();torch.cuda.synchronize()
        parameters={k:p.detach().cpu().clone() for k,p in model.named_parameters()}
        optimizer_state={name+'/'+key:value.detach().cpu().clone()
            for name,param in model.named_parameters() for key,value in optimizer.state[param].items()
            if torch.is_tensor(value)}
        values.append(dict(loss=step_loss,gradients=gradients,inputs=input_gradients,parameters=parameters,
                           optimizer_state=optimizer_state,elapsed_seconds=time.perf_counter()-started))
        del model,optimizer,objective,capture,base,aux,detail,physical,loss
    a,b=values
    inputs=[compare_tensors(x,y) for x,y in zip(a['inputs'],b['inputs'])]
    if len(inputs)!=16:raise ValueError('All 16 original F0 input gradient pairs required')
    g=compare_named(a['gradients'],b['gradients']);u=compare_named(a['parameters'],b['parameters'])
    optimizer_state=compare_named(a['optimizer_state'],b['optimizer_state'])
    delta=abs(a['loss']-b['loss'])
    return dict(epoch=epoch,loss_original=a['loss'],loss_cached=b['loss'],loss_abs=delta,
        loss_close=abs(delta)<=ATOL+RTOL*abs(a['loss']),
        input_gradients=dict(max_abs=max(x['max_abs'] for x in inputs),notclose=sum(x['notclose'] for x in inputs)),
        parameter_gradients=g,updated_parameters=u,optimizer_state=optimizer_state,
        original_seconds=a['elapsed_seconds'],cached_seconds=b['elapsed_seconds'],
        same_device_single_logical_batch_only=True)


def worker(args):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';started=time.perf_counter();cfg=validate_source(read(args.inputs))
    reader,identity=load_F0(cfg,device);drift.valid_identity(identity);frozen_eval(reader)
    if identity['source_code_commit']!=drift.F0_COMMIT:raise ValueError('Original F0 required')
    snapshot={k:v.detach().clone() for k,v in reader.state_dict().items()}
    parent=args.root/'runs'/drift.PARENT/'rows'/'TF_JointPrior'/'training';done=read(parent/'complete.json')
    if (done['status'],done['code_commit'],done['epoch'],done['updates'])!=('SOURCE_TRAINED',drift.PARENT_COMMIT,30,7410):
        raise ValueError('Fixed source-only final30 required')
    if any(done[k] for k in ('TX_labels_read','V_reads','OE_reads','target_training_reads','G_initialization_checkpoint_loads')):
        raise ValueError('Source-only parent provenance required')
    if done['F0_identity']!=identity:raise ValueError('Exact matched F0 required')
    ckpt=torch.load(parent/'final.pt',map_location='cpu',weights_only=False)
    if (ckpt['code_commit'],ckpt['epoch'],ckpt['updates'],ckpt['config'])!=(drift.PARENT_COMMIT,30,7410,drift.parent.configuration(drift.parent.ARM)):
        raise ValueError('Exact original parent checkpoint required')
    state=ckpt['state'];source,roles=drift.feature.trial.source(args)
    meta=read(source/'complete.json')
    if (meta['status'],meta['train_count'],meta['TX_labels_read'],meta['target_reads'])!=('ACTION_SOURCE_CACHE_COMPLETE',63000,0,0):
        raise ValueError('Original validated source cache required')
    indices_np=np.r_[np.arange(128),np.arange(6300,6428)]
    if not (np.all(roles[indices_np[:128]]=='L_s') and np.all(roles[indices_np[128:]]=='U_s')):
        raise ValueError('128 legal rows per source role required')
    def rows(path):
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        try:return tensor(array[indices_np].copy(),device)
        finally:array._mmap.close()
    clean=rows(source/'train/clean.npy');folder=source/'epochs'/'01';draw=read(folder/'complete.json')
    if (draw['count'],draw['TX_labels_read'],draw['target_reads'])!=(63000,0,0):raise ValueError('Original source draw required')
    trace={key:rows(folder/(key+'.npy')) for key in ('received','kernel','coefficients','variance')}
    indices=torch.as_tensor(indices_np,device=device,dtype=torch.long);scale=load_scale(args.root).to(device)
    model=drift.feature.factory(drift.feature.ARM).to(device).eval();model.load_state_dict(state,strict=True)
    with torch.no_grad():
        target_smoke=reader.features(clean[:2]).detach()
        c=drift.ReaderCapture(reader);base,aux,loss,_=drift.feature.response_readout(model,c,trace['received'][:2],clean[:2],
            target_smoke,scale,PairedObjective().to(device),source_indices=indices[:2],epoch=1,priors=None,assignment=None,
            source_trace={k:v[:2] for k,v in trace.items()})
        if not torch.isfinite(base).all() or not torch.isfinite(loss).all() or aux['relative_solve_residual'].max()>1e-8:
            raise ValueError('Real production G/F0 source smoke failed')
    del model,c,base,aux,loss
    out=args.root/'runs'/RUN
    write(out/'smoke.json',dict(status='PASS',source_rows=2,actual_source_G=True,actual_frozen_F0=True,
                               TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('REAL_GPU_SOURCE_G_F0_SMOKE_PASS',flush=True)
    with torch.no_grad():target=reader.features(clean).detach()
    cache=cache_source_features(reader,clean,indices)
    references=[]
    with torch.no_grad():
        for epoch in (1,3):
            for start in range(0,256,32):
                sl=slice(start,start+32);delta,choice=selected_perturbation(clean[sl],indices[sl],epoch)
                online=reader.features(clean[sl]+delta);saved=cache[sl][torch.arange(32,device=device),choice]
                references.append(compare_tensors(online,saved))
    cases=[run_case(state,reader,clean,trace,indices,target,cache,scale,epoch) for epoch in (1,3)]
    check_F0_state(reader,snapshot)
    reference=dict(max_abs=max(x['max_abs'] for x in references),notclose=sum(x['notclose'] for x in references),
                   comparisons=len(references))
    write(out/'results.json',dict(status='MEASURED',source_records=256,source_roles={'L_s':128,'U_s':128},
        source_cases=[1,3],logical_batch=256,microbatch=32,source_cache_batch=256,covered_directions=4,
        source_reference=reference,cases=cases,F0_identity=identity,parent_run=drift.PARENT,parent_commit=drift.PARENT_COMMIT,
        TF32=False,temporary_optimizer_updates=4,checkpoint_changes=0,TX_labels_read=0,V_reads=0,OE_reads=0,
        target_inputs=0,full_training_equivalence_verified=False,formal_quality_verified=False,
        timing_scope='one logical batch per path excludes cache build; not end-to-end speed',seconds=time.perf_counter()-started))
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',source_only=True,model_cases=4))
    print('SOURCE_GPU_CACHE_COMPARISON_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--commit',required=True)
    args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Release identity mismatch')
    if args.mode=='run':
        controller.RUN,controller.MODULE=RUN,MODULE
        controller.controller(args)
    else:worker(args)


if __name__=='__main__':main()
