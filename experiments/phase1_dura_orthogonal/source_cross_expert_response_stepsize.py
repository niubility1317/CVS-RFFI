"""Single in-memory source diagnostic for wave and response head updates."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.counterfactual_response import response_loss
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner
from experiments.phase1_dura_orthogonal.signal_confidence_gate import source_gate_loss
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import EPSILON,perturbations
from experiments.phase1_dura_orthogonal.train_source import extract_context

SEED=2026110830
RATES=(.00003,.000003)
RESPONSE_WEIGHT=.01
VIEWS=('practical_high','practical_low_urban')


def objective(wave,response,mode):
    if mode=='wave_only':return wave
    if mode=='wave_plus_response':return wave+RESPONSE_WEIGHT*response
    raise ValueError('Fixed diagnostic mode required')


def losses(model,batches):
    wave=[];response=[]
    for x,a,d,v,p,clean,changed,ca,cd,cv,cp,delta in batches:
        original=model(x,a,d,p,v)
        altered=model(changed,ca,cd,cp,cv)
        wave.append(source_gate_loss(original,a,clean))
        response.append(response_loss(original,altered,delta))
    return torch.stack(wave).mean(),torch.stack(response).mean()


def values(model,batches):
    with torch.no_grad():
        wave,response=losses(model,batches)
        return dict(wave=float(wave),response=float(response))


def prepare(root,dura,direct,ids,received,clean,positions):
    batches=[]
    with torch.no_grad():
        for start in range(0,len(positions),8):
            sel=ids[positions[start:start+8]]
            x,y=core.tensor(received[sel],'cpu'),core.tensor(clean[sel],'cpu')
            action=dura.estimate_context(x)[1]
            a,d,v=extract_context(dura,x);proposal=direct(x,validate=False)
            delta=perturbations(y)[1]
            observed=action(delta)
            changed=x+torch.stack((observed.real,observed.imag),1)
            ca,cd,cv=extract_context(dura,changed);cp=direct(changed,validate=False)
            batches.append((x,a,d,v,proposal,y,changed,ca,cd,cv,cp,torch.stack((delta.real,delta.imag),1)))
    return batches


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Unique source-only output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids,rx,scene,_,received,clean=source_inputs(args.root)
    fit=[];held=[]
    for receivers in ((4,6),(8,)):
        for view in VIEWS:
            found=np.flatnonzero(np.isin(rx,receivers)&(scene==view))
            if len(found)!=48:raise ValueError('Complete fixed source groups required')
            fit.extend(found[:8].tolist());held.extend(found[8:16].tolist())
    fit=np.asarray(fit,dtype=np.int64);held=np.asarray(held,dtype=np.int64)
    if len(fit)!=32 or len(held)!=32 or len(np.unique(ids[np.r_[fit,held]]))!=64:
        raise ValueError('Disjoint fit and heldout source records required')
    direct=load_direct(args.root,ids)
    dura=frozen_dura(args.root,'cpu')
    started=time.monotonic()
    train=prepare(args.root,dura,direct,ids,received,clean,fit)
    heldout=prepare(args.root,dura,direct,ids,received,clean,held)
    with torch.no_grad():
        torch.testing.assert_close(train[0][1],dura(train[0][0]),rtol=1e-5,atol=3e-6)
    torch.manual_seed(SEED)
    smoke=CrossExpertRefiner().train()
    with torch.no_grad():
        x,a,d,v,proposal=train[0][:5]
        torch.testing.assert_close(smoke(x,a,d,proposal,v),(a+proposal)*.5,rtol=1e-5,atol=3e-6)
    del smoke
    print('CROSS_RESPONSE_STEPSIZE_SMOKE_PASS',8,flush=True)
    results={};initial=None
    for lr in RATES:
        for mode in ('wave_only','wave_plus_response'):
            torch.manual_seed(SEED)
            model=CrossExpertRefiner().train()
            for parameter in model.parameters():parameter.requires_grad_(False)
            params=tuple(model.time_out.parameters())+tuple(model.freq_out.parameters())
            for parameter in params:parameter.requires_grad_(True)
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if initial is None:initial=state
            elif any(not torch.equal(initial[k],state[k]) for k in initial):raise ValueError('Identical model initialization required')
            before=dict(fit=values(model,train),heldout=values(model,heldout))
            optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=.001)
            optimizer.zero_grad(set_to_none=True)
            wave,response=losses(model,train)
            loss=objective(wave,response,mode)
            if not torch.isfinite(loss):raise FloatingPointError('Finite objective required')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
            after=dict(fit=values(model,train),heldout=values(model,heldout))
            results[str(lr)+'/'+mode]=dict(lr=lr,mode=mode,before=before,after=after)
            print('CROSS_RESPONSE_STEPSIZE_MODE',lr,mode,flush=True)
    if any(p.grad is not None or p.requires_grad for obj in (dura,direct) for p in obj.parameters()):
        raise ValueError('Fixed source experts must remain frozen')
    payload=dict(status='CROSS_RESPONSE_STEPSIZE_COMPLETE',commit=args.commit,
        source_fit_records=32,source_heldout_records=32,source_groups=4,model_seed=SEED,
        optimized_parameters=2308,optimizer='AdamW',learning_rates=list(RATES),weight_decay=.001,response_weight=RESPONSE_WEIGHT,
        perturbation='amplitude_cubic',epsilon_relative_rms=EPSILON,results=results,
        elapsed_seconds=time.monotonic()-started,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,
        V_reads=0,target_reads=0,optimizer_steps_in_memory=4,saved_model=False,
        caveat='One fresh CPU AdamW step for each rate and condition on output heads only; fixed explored source records. Not GPU full training, an independent generalization test, identity accuracy or target performance.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('CROSS_RESPONSE_STEPSIZE_COMPLETE',flush=True)


if __name__=='__main__':main()
