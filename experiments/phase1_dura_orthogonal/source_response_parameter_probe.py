"""Source-only parameter-gradient probe for synthetic physical responses."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
import experiments

from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_orthogonal.source_context_cache import _contract, _validate_data
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import perturbations, DIRECTIONS, EPSILON
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
from experiments.phase1_dura_orthogonal.train_source import extract_context, source_loss


def response_loss(original, changed, delta):
    if original.shape!=changed.shape or original.shape!=delta.shape or original.ndim!=3:
        raise ValueError('Matching paired IQ responses required')
    numerator=(changed-original-delta).square().sum((1,2))
    denominator=delta.square().sum((1,2)).clamp_min(1e-20)
    return (numerator/denominator).mean()


def gradient_vector(loss,parameters,retain_graph):
    parts=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,parts)])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--initial-state',type=Path,required=True)
    p.add_argument('--initial-commit',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    payload=torch.load(args.cache,map_location='cpu',weights_only=True)
    fit=payload['contract']['fit_ids']
    if len(fit)!=32768 or payload['contract']!=_contract(fit,32,'cuda:0',deterministic=True):
        raise ValueError('Matching fixed source context required')
    _validate_data(payload['data'],payload['counts'],3*len(fit))
    state=torch.load(args.initial_state,map_location='cpu',weights_only=False)
    if (state.get('commit')!=args.initial_commit or state.get('seed')!=2026110612 or
            state.get('source_dura')!='dura_v1_R65_E20_4940' or
            not np.array_equal(np.asarray(state.get('source_fit_ids',[])),np.asarray(fit)) or
            any(state.get(k)!=0 for k in ('TX_labels_read','F_weights_read','F_outputs_read','V_reads','target_reads'))):
        raise ValueError('Matching source-only initialization required')
    data=tuple(torch.cat([v[i:i+32] for i in (0,32768,65536)]).to('cuda:0') for v in payload['data'])
    experiments.__path__=[str(args.root/'releases/pamf_stage2_inference_20261021_r01/experiments'),*experiments.__path__]
    dura=frozen_dura(args.root,'cuda:0')
    model=DuraMultiscaleSpectral(width=32,taps=9).to('cuda:0')
    model.load_state_dict(state['model'],strict=True)
    model.train()
    parameters=tuple(p for p in model.parameters() if p.requires_grad)
    sums=None
    base_sum=None
    rows=[]
    started=time.monotonic()
    for start in range(0,96,32):
        x,_,_,_,clean=(v[start:start+32] for v in data)
        with torch.no_grad():
            anchor,direction,variance=extract_context(dura,x)
            action=dura.estimate_context(x)[1]
            deltas=perturbations(clean)
            pairs=[]
            for delta in deltas:
                received_delta=action(delta)
                changed_x=x+torch.stack((received_delta.real,received_delta.imag),dim=1)
                changed_context=extract_context(dura,changed_x)
                pairs.append((changed_x,changed_context,torch.stack((delta.real,delta.imag),dim=1)))
        original=model(x,anchor,direction,variance,validate=False)
        base=source_loss(original,anchor,clean,objective='dura_guard')
        bg=gradient_vector(base,parameters,True).detach()
        if base_sum is None:
            base_sum=torch.zeros_like(bg)
            sums=[torch.zeros_like(bg) for _ in DIRECTIONS]
        base_sum+=bg/3
        losses=[]
        for i,(changed_x,changed_context,delta) in enumerate(pairs):
            changed=model(changed_x,*changed_context,validate=False)
            term=response_loss(original,changed,delta)
            cg=gradient_vector(term,parameters,True).detach()
            sums[i]+=cg/3
            losses.append(float(term.detach()))
        rows.append(dict(source_view=21+start//32,base_loss=float(base.detach()),response_losses=losses))
    if not torch.isfinite(base_sum).all() or any(not torch.isfinite(g).all() for g in sums):
        raise FloatingPointError('Finite source parameter gradients required')
    base_norm=base_sum.norm().clamp_min(1e-20)
    metrics={}
    for name,g in zip(DIRECTIONS,sums):
        norm=g.norm()
        metrics[name]=dict(gradient_cosine=float((base_sum@g)/(base_norm*norm.clamp_min(1e-20))),
                           weighted_001_to_base_norm=float(.01*norm/base_norm))
    if any(p.grad is not None for p in dura.parameters()):
        raise ValueError('DURA must remain frozen')
    torch.cuda.synchronize()
    args.out.mkdir(parents=True)
    result=dict(status='COMPLETE',commit=args.commit,source_ids=32,views=3,records=96,
        perturbations=list(DIRECTIONS),epsilon_relative_rms=EPSILON,reference_weight=.01,
        batches=rows,parameter_gradients=metrics,elapsed_seconds=time.monotonic()-started,
        device=torch.cuda.get_device_name(),TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,
        model_updates=0,weights_saved=False,
        caveat='Synthetic perturbations and DURA estimated action are source-only proxies, not TX identity or target benefit.')
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('SOURCE_RESPONSE_PARAMETER_PROBE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
