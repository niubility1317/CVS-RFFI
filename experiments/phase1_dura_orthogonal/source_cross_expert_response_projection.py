"""Source-only test of waveform-protecting response gradient projection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner
from experiments.phase1_dura_orthogonal.source_cross_expert_response_step import prepare, losses, values, SEED, VIEWS, RESPONSE_WEIGHT
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import EPSILON

RATES=(3e-5,3e-6)
MODES=('wave_only','joint','projected')


def project_response(wave, response):
    """Remove only the response component opposing waveform descent."""
    if len(wave)!=len(response) or not wave:
        raise ValueError('Matching nonempty gradients required')
    dot=sum((a*b).sum() for a,b in zip(wave,response))
    norm=sum((a*a).sum() for a in wave)
    factor=torch.where((dot<0)&(norm>0),dot/norm.clamp_min(1e-30),torch.zeros_like(dot))
    return tuple(b-factor*a for a,b in zip(wave,response))


def dot(left,right):
    return sum((a*b).sum() for a,b in zip(left,right))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Unique source diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids,rx,scene,_,received,clean=source_inputs(args.root)
    fit=[];held=[]
    for receivers in ((4,6),(8,)):
        for view in VIEWS:
            group=np.flatnonzero(np.isin(rx,receivers)&(scene==view))
            if len(group)!=48:raise ValueError('Complete source groups required')
            fit.extend(group[:8].tolist());held.extend(group[8:16].tolist())
    fit=np.asarray(fit,dtype=np.int64);held=np.asarray(held,dtype=np.int64)
    if len(fit)!=32 or len(held)!=32 or len(np.unique(ids[np.r_[fit,held]]))!=64:
        raise ValueError('Disjoint fit and heldout source IDs required')
    direct=load_direct(args.root,ids);dura=frozen_dura(args.root,'cpu')
    began=time.monotonic()
    train=prepare(args.root,dura,direct,ids,received,clean,fit)
    heldout=prepare(args.root,dura,direct,ids,received,clean,held)
    with torch.no_grad():torch.testing.assert_close(train[0][1],dura(train[0][0]),rtol=1e-5,atol=3e-6)
    torch.manual_seed(SEED)
    smoke=CrossExpertRefiner().train()
    with torch.no_grad():
        x,a,d,v,proposal=train[0][:5]
        torch.testing.assert_close(smoke(x,a,d,proposal,v),(a+proposal)*.5,rtol=1e-5,atol=3e-6)
    del smoke
    print('CROSS_PROJECTION_SMOKE_PASS',8,flush=True)
    results={}; initial=None
    for lr in RATES:
        for mode in MODES:
            torch.manual_seed(SEED);model=CrossExpertRefiner().train()
            for p in model.parameters():p.requires_grad_(False)
            params=tuple(model.time_out.parameters())+tuple(model.freq_out.parameters())
            for p in params:p.requires_grad_(True)
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if initial is None:initial=state
            elif any(not torch.equal(initial[k],state[k]) for k in initial):raise ValueError('Matching initialization required')
            before=dict(fit=values(model,train),heldout=values(model,heldout))
            optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=.001)
            wave,response=losses(model,train)
            if not torch.isfinite(wave) or not torch.isfinite(response):raise FloatingPointError('Finite source losses required')
            gw=torch.autograd.grad(wave,params,retain_graph=True)
            gr=torch.autograd.grad(response,params)
            projected=project_response(gw,gr)
            for p,a,b,c in zip(params,gw,gr,projected):
                p.grad=(a if mode=='wave_only' else a+RESPONSE_WEIGHT*(b if mode=='joint' else c)).detach().clone()
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            previous=[p.detach().clone() for p in params]
            optimizer.step()
            update=[p.detach()-q for p,q in zip(params,previous)]
            after=dict(fit=values(model,train),heldout=values(model,heldout))
            results[str(lr)+'/'+mode]=dict(lr=lr,mode=mode,before=before,after=after,
                wave_response_dot=float(dot(gw,gr)),wave_projected_dot=float(dot(gw,projected)),
                wave_actual_update_dot=float(dot(gw,update)))
            print('CROSS_PROJECTION_MODE',lr,mode,flush=True)
    if any(p.grad is not None or p.requires_grad for obj in (dura,direct) for p in obj.parameters()):
        raise ValueError('Frozen source experts must remain unchanged')
    payload=dict(status='CROSS_PROJECTION_COMPLETE',commit=args.commit,source_fit_records=32,
        source_heldout_records=32,source_groups=4,model_seed=SEED,optimized_parameters=2308,
        optimizer='AdamW',learning_rates=list(RATES),weight_decay=.001,response_weight=RESPONSE_WEIGHT,
        perturbation='amplitude_cubic',epsilon_relative_rms=EPSILON,results=results,
        elapsed_seconds=time.monotonic()-began,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,
        target_reads=0,optimizer_steps_in_memory=6,saved_model=False,
        caveat='Exploratory one-step CPU source diagnostic, fresh model output heads only; source pilot explored, no full-model training, independent blind test, TX accuracy or target inference. AdamW is preconditioned so gradient projection need not protect actual loss.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('CROSS_PROJECTION_COMPLETE',flush=True)


if __name__=='__main__':main()
