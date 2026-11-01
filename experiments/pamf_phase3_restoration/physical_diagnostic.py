"""Frozen source-only factorial intervention on the joint PHY-MSAT system."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .physical import Action,complex_iq,pair,soft_map,MIN_NOISE,PRIOR_VARIANCE,CG_STEPS
from . import physical_trial as parent
from .cache import tensor,load_F0,check_F0_state
from .source_pair import read,validate_source
from .source_inverse import panel,SCENES
from .cost_probe import write

RUN='20261101-diagnostic-Restore-PHYMSAT-factorial-source-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.physical_diagnostic'
G_COMMIT='8198a2e70c655f24ebaf321f232a676cfb46d916'
METHODS=('blind','true_H','true_noise','true_H_noise')
METRICS=('nmse','feature_cosine_distance','feature_standardized_mse','clean_decision_agreement')


def true_trace(kernel,coefficients,variance):
    n=len(kernel)
    if (kernel.shape!=(n,2,64) or coefficients.shape!=(n,4,256) or variance.shape!=(n,)
            or not all(torch.isfinite(v).all() for v in (kernel,coefficients,variance)) or (variance<0).any()):
        raise ValueError('Matched finite original source simulator traces required')
    return Action(complex_iq(kernel),torch.complex(coefficients[:,0],coefficients[:,1]),
        torch.complex(coefficients[:,2],coefficients[:,3])),variance.clamp_min(MIN_NOISE)


def restore_with(model,received,action,variance):
    """Exactly the deployment iteration with only the two explicit factors replaced."""
    if variance.shape!=(len(received),) or not torch.isfinite(variance).all() or not (variance>0).all():
        raise ValueError('Finite positive per-record noise required')
    observed=complex_iq(received);current=observed
    condition=variance.log()[:,None,None].expand(-1,1,256)
    for prior in model.priors:
        back=action.adjoint(observed-action(current))
        mean=complex_iq(prior(pair(current),torch.cat((received,pair(back),condition),1)))
        current=soft_map(action,observed,mean,variance,PRIOR_VARIANCE,steps=CG_STEPS)
    return pair(current)


def factorial(model,received,action,variance):
    estimated,noise=model.estimator(received)
    values={name:restore_with(model,received,h,v) for name,h,v in (
        ('blind',estimated,noise),('true_H',action,noise),
        ('true_noise',estimated,variance),('true_H_noise',action,variance))}
    return values,estimated,noise


def summarize(records):
    scenes=np.asarray(records['scene'])
    if len(scenes)!=10240 or set(scenes)!=set(SCENES):raise ValueError('All10240 fixed source cases/scenes required')
    for method in METHODS:
        for metric in METRICS:
            value=records[method+'__'+metric]
            if np.shape(value)!=(10240,) or not np.isfinite(value).all():raise ValueError('Complete finite source metrics required')
    result={}
    for scene in ('ALL',*SCENES):
        mask=np.ones(len(scenes),bool) if scene=='ALL' else scenes==scene
        item=dict(count=int(mask.sum()),methods={},paired_against_blind={})
        for method in METHODS:
            item['methods'][method]={k:float(records[method+'__'+k][mask].mean()) for k in METRICS}
            before=records['blind__clean_decision_agreement'].astype(bool)
            after=records[method+'__clean_decision_agreement'].astype(bool)
            repair=int((~before&after&mask).sum());damage=int((before&~after&mask).sum())
            item['paired_against_blind'][method]=dict(repairs=repair,damage=damage,net=repair-damage)
        item['estimated_action_forward_nmse']=float(records['estimated_action_forward_nmse'][mask].mean())
        item['estimated_noise_log_mse']=float(records['estimated_noise_log_mse'][mask].mean())
        result[scene]=item
    return result


def worker(args):
    started=time.perf_counter();device='cuda:0';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/'runs'/RUN;old,roles=parent.trial.source(args)
    # The parent is finished; diagnostic weights are its single fixed final30.
    root=args.root/'runs'/parent.RUN
    if read(root/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Completed parent required')
    done=read(root/'rows/PHY_MSat/training/complete.json')
    if (done['code_commit'],done['epoch'],done['updates'])!=(G_COMMIT,30,7410):raise ValueError('Fixed original source parent required')
    if any(done[k] for k in ('G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates')):
        raise ValueError('Original source-only parent provenance required')
    scale=parent.load_scale(args.root);parent.install(scale)
    G=parent.trial.load(root/'rows/PHY_MSat/training/final.pt',parent.ARM,G_COMMIT,device)
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);parent.trial.main_eval.valid_identity(identity)
    snapshots=[(m,{k:v.detach().clone() for k,v in m.state_dict().items()}) for m in (F,G)]
    selected=panel(roles);a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:
        if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source clean FP32 required')
        clean=tensor(a[selected],device)
    finally:a._mmap.close()
    # First neural work: real two-L_s checkpoint smoke and exact baseline replay.
    folder=old/'epochs/01';values={}
    for key in ('received','kernel','coefficients','variance'):
        a=np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
        try:values[key]=tensor(a[selected[:2]],device)
        finally:a._mmap.close()
    with torch.inference_mode():
        h,v=true_trace(values['kernel'],values['coefficients'],values['variance'])
        paths,_,_=factorial(G,values['received'],h,v)
        torch.testing.assert_close(paths['blind'],G(values['received']),atol=0,rtol=0)
        for value in paths.values():
            logits=F.classify_features(F.features(value))
            if value.shape!=(2,2,256) or logits.shape!=(2,6) or not torch.isfinite(value).all() or not torch.isfinite(logits).all():
                raise ValueError('Actual source checkpoint smoke failed')
    write(out/'smoke.json',dict(status='PASS',source_Ls=2,exact_deployment_baseline_replay=True,
        actual_F0=identity,parent_code=G_COMMIT,all_four_paths=True,optimizer_updates=0,target_inputs=0,TX_labels_read=0))
    print('PHY_FACTORIAL_REAL_SOURCE_SMOKE_PASS',flush=True)
    with torch.inference_mode():
        target_features=F.features(clean);decisions=F.classify_features(target_features).argmax(1)
    scale=scale.to(device)
    fields=['epoch','source_index','scene','state','lock','actual_variance','estimated_variance',
        'estimated_action_forward_nmse','estimated_noise_log_mse',*[m+'__'+k for m in METHODS for k in METRICS]]
    records={k:[] for k in fields}
    with torch.inference_mode():
        for epoch in range(1,21):
            folder=old/'epochs'/f'{epoch:02d}';done=read(folder/'complete.json')
            if (done.get('count'),done.get('TX_labels_read'),done.get('target_reads'))!=(63000,0,0):raise ValueError('Completed original source draw required')
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('received','kernel','coefficients','variance')}
            with np.load(folder/'groups.npz',allow_pickle=False) as f:groups={k:f[k][selected] for k in ('scene','state','lock')}
            try:
                shapes={'received':(63000,2,256),'kernel':(63000,2,64),'coefficients':(63000,4,256),'variance':(63000,)}
                if any(a.shape!=shapes[k] or a.dtype!=np.float32 for k,a in arrays.items()):raise ValueError('Original source draw shapes required')
                for offset in range(0,512,64):
                    take=selected[offset:offset+64];x=clean[offset:offset+64];n=len(x)
                    inputs={k:tensor(a[take],device) for k,a in arrays.items()}
                    h,v=true_trace(inputs['kernel'],inputs['coefficients'],inputs['variance'])
                    paths,estimated,noise=factorial(G,inputs['received'],h,v)
                    target=target_features[offset:offset+n];tx=complex_iq(x);expected=h(tx)
                    metrics=dict(estimated_action_forward_nmse=(estimated(tx)-expected).abs().square().mean(1)/expected.abs().square().mean(1).clamp_min(1e-4),
                        estimated_noise_log_mse=(noise.log()-v.log()).square(),actual_variance=inputs['variance'],estimated_variance=noise)
                    for method,value in paths.items():
                        z=F.features(value)
                        metrics.update({method+'__'+k:q for k,q in dict(nmse=(value-x).square().mean((1,2))/x.square().mean((1,2)),
                            feature_cosine_distance=1-torch.nn.functional.cosine_similarity(z,target,dim=1),
                            feature_standardized_mse=((z-target).square()/scale).mean(1),
                            clean_decision_agreement=F.classify_features(z).argmax(1)==decisions[offset:offset+n]).items()})
                        if not torch.isfinite(value).all():raise ValueError('Nonfinite source path; preserve all cases')
                    if any(not torch.isfinite(q).all() for q in metrics.values()):raise ValueError('Nonfinite source metric; never omit cases')
                    for key,value in metrics.items():records[key].extend(value.cpu().tolist())
                    records['epoch'].extend([epoch]*n);records['source_index'].extend(take.tolist())
                    for key in groups:records[key].extend(groups[key][offset:offset+n].tolist())
            finally:
                for a in arrays.values():a._mmap.close()
            print('PHY_FACTORIAL_SOURCE_DRAW_COMPLETE',epoch,512,flush=True)
    for model,before in snapshots:check_F0_state(model,before)
    records={k:np.asarray(v) for k,v in records.items()}
    with (out/'source_records.npz').open('xb') as f:np.savez(f,**records)
    write(out/'results.json',dict(status='SOURCE_FACTORIAL_COMPLETE',run_id=RUN,code_commit=args.commit,parent_code=G_COMMIT,
        parent_run=parent.RUN,F0_identity=identity,count=10240,source_packets=512,source_draws=20,source_Ls=128,source_Us=384,
        summary=summarize(records),G_parameters=8613449,prior_variance=PRIOR_VARIANCE,CG_steps_each=CG_STEPS,noise_floor=MIN_NOISE,
        G_training_updates=0,optimizer_updates=0,model_state_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,
        agreement_is_TX_accuracy=False,oracle_deployable=False,theoretical_ceiling=False,default_DURA_changed=False,
        intervention_also_changes_learned_prior_context=True,source_in_sample=True,seconds=time.perf_counter()-started,goal_complete=False))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty launch owner required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    choices=parent.choices()
    while not choices:time.sleep(5);choices=parent.choices()
    admission=choices[0]
    command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,
        '--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Source factorial diagnostic failed; preserve/no auto retry')
    if read(out/'results.json')['status']!='SOURCE_FACTORIAL_COMPLETE':raise ValueError('Missing complete source results')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual Git code identity required')
    if args.mode=='run':controller(args)
    else:worker(args)


if __name__=='__main__':main()
