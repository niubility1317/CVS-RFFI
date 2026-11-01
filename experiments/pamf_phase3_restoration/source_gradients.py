"""Frozen source gradient geometry; no optimizer, query, labels, or parameter update."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import feature_conditioned_trial as control
from . import feature_strong_trial as strong
from . import multiview_trial as base
from . import variance_trial as varf
from .feature_conditioned import scratch_feature_conditioned,latent_feature_loss
from .full_feature_trial import full_feature_guide
from .objective import PairedObjective
from .source_inverse import panel
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .main_split_eval import valid_identity
from .cost_probe import write

RUN='20261101-diagnostic-Restore-source-gradient-geometry-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.source_gradients'
SPECS={control.ARM:(control.RUN,'c8659b15bb3d316325867b154eede51f1a61e247',control.configuration,.1),
       strong.ARM:(strong.RUN,'3e5416df5b13dfcf9e16995988e988c5494f68b2',strong.configuration,1.)}
OBJECTIVES=('physical','output_F','latent_unit')


def parameter_group(name):
    if name=='position' or name.startswith(('stem.','encoders.','down.','bottleneck.')):return 'encoder'
    if name.startswith('feature_regressor.'):return 'latent_head'
    if name.startswith('conditions.'):return 'FiLM'
    if name.startswith(('up.','decoders.','output.')):return 'decoder'
    raise ValueError('Unclassified G parameter: '+name)


def geometry(gram,weight):
    gram=np.asarray(gram,dtype=np.float64)
    if gram.shape!=(3,3) or not np.isfinite(gram).all() or (np.diag(gram)<0).any():
        raise ValueError('Finite nonnegative-diagonal three-objective gradient Gram required')
    total=np.asarray([1.,1.,weight]);main=np.asarray([1.,1.,0.]);aux=np.asarray([0.,0.,1.])
    square=float(total@gram@total)
    if square < -1e-6*max(1.,float(np.trace(gram))):raise ValueError('Invalid total gradient squared norm')
    norm=float(np.sqrt(max(0.,square)));main_norm=float(np.sqrt(max(0.,main@gram@main)));aux_norm=float(np.sqrt(gram[2,2]))
    cosine=float(main@gram@aux)/(main_norm*aux_norm) if main_norm>0 and aux_norm>0 else None
    return dict(total_gradient_norm=norm,clip1_factor_if_clipped_alone=min(1.,1./(norm+1e-6)),
        main_gradient_norm=main_norm,latent_unit_gradient_norm=aux_norm,main_latent_cosine=cosine,
        main_dot_total=float(main@gram@total),physical_dot_total=float(gram[0]@total),
        output_F_dot_total=float(gram[1]@total),latent_dot_total=float(gram[2]@total))


def gradient_rows(model,reader,observed,clean,target,scale,objective):
    restored,aux=model(observed,return_aux=True)
    physical,_=objective(restored,clean)
    output,_=full_feature_guide(reader,restored,clean,target,scale)
    _,terms=latent_feature_loss(aux['predicted_features'],target,scale)
    losses=[physical.mean(),output.mean(),(terms['latent_cosine_error']+terms['latent_standardized_error']).mean()]
    if any(not torch.isfinite(q) for q in losses):raise ValueError('Nonfinite source objective')
    named=list(model.named_parameters());params=[p for _,p in named]
    gradients=[torch.autograd.grad(loss,params,retain_graph=i<2,allow_unused=True) for i,loss in enumerate(losses)]
    groups={k:[i for i,(n,_) in enumerate(named) if parameter_group(n)==k] for k in ['encoder','latent_head','FiLM','decoder']}
    groups['all']=list(range(len(named)));rows={}
    for group,indices in groups.items():
        vectors=[torch.cat([(gradient[i].detach() if gradient[i] is not None else torch.zeros_like(params[i])).reshape(-1) for i in indices]) for gradient in gradients]
        if any(not torch.isfinite(v).all() for v in vectors):raise ValueError('Nonfinite source derivative')
        gram=torch.stack([torch.stack([torch.dot(a,b) for b in vectors]) for a in vectors]).cpu().tolist()
        rows[group]=dict(parameter_count=sum(params[i].numel() for i in indices),gram=gram,
            by_latent_weight={str(w):geometry(gram,w) for w in [.1,1.]})
    for row in rows.values():
        for weight,value in row['by_latent_weight'].items():
            value['global_clip1_factor']=rows['all']['by_latent_weight'][weight]['clip1_factor_if_clipped_alone']
    if any(p.grad is not None for p in params):raise ValueError('Derivative diagnostic accumulated parameter .grad')
    return rows,[float(v.detach()) for v in losses]


def load_models(root,device):
    result={}
    for arm,(run,commit,configuration,_) in SPECS.items():
        folder=root/'runs'/run
        if read(folder/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Both complete frozen Main parents required')
        done=read(folder/'rows'/arm/'training/complete.json')
        if (done['code_commit'],done['epoch'],done['updates'])!=(commit,30,7410) or any(done[k] for k in
                ['G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates']):
            raise ValueError('Original source-only scratch30 parent required')
        p=torch.load(folder/'rows'/arm/'training/final.pt',map_location='cpu',weights_only=False)
        if (set(p)!={'config','code_commit','epoch','updates','state'} or p['config']!=configuration(arm)
                or (p['code_commit'],p['epoch'],p['updates'])!=(commit,30,7410)):
            raise ValueError('Strict original fixedfinal30 checkpoint required')
        model=scratch_feature_conditioned(2026092701);state=model.state_dict()
        if set(p['state'])!=set(state) or any(v.dtype!=torch.float32 or v.shape!=state[k].shape or not torch.isfinite(v).all() for k,v in p['state'].items()):
            raise ValueError('Exact finite G state required')
        model.load_state_dict(p['state'],strict=True);result[arm]=model.to(device).eval().requires_grad_(False)
    return result


def summarize(records):
    if len(records)!=80 or set(q['arm'] for q in records)!=set(SPECS):raise ValueError('Both G x20draw x2batch required')
    result={}
    for arm in SPECS:
        chosen=[q for q in records if q['arm']==arm]
        if {(q['epoch'],q['offset']) for q in chosen}!={(e,o) for e in range(1,21) for o in [0,256]} or len(chosen)!=40:
            raise ValueError('Unique complete40 source batches per model required')
        result[arm]={}
        for group in ['encoder','latent_head','FiLM','decoder','all']:
            result[arm][group]={}
            for weight in ['0.1','1.0']:
                values=[q['groups'][group]['by_latent_weight'][weight] for q in chosen]
                cos=[q['main_latent_cosine'] for q in values if q['main_latent_cosine'] is not None]
                result[arm][group][weight]=dict(batches=40,cosine_defined_batches=len(cos),
                    mean_main_latent_cosine=float(np.mean(cos)) if cos else None,
                    negative_main_latent_cosine_batches=sum(v<0 for v in cos),
                    mean_total_gradient_norm=float(np.mean([v['total_gradient_norm'] for v in values])),
                    mean_clip1_factor=float(np.mean([v['global_clip1_factor'] for v in values])),
                    clip_active_batches=sum(v['global_clip1_factor']<1 for v in values),
                    main_dot_total_negative_batches=sum(v['main_dot_total']<0 for v in values),
                    physical_dot_total_negative_batches=sum(v['physical_dot_total']<0 for v in values))
    return result


def worker(args):
    started=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/'runs'/RUN;old,roles=base.source(args);selected=panel(roles);models=load_models(args.root,device)
    c=validate_source(read(args.inputs));reader,identity=load_F0(c,device);valid_identity(identity)
    snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in [reader,*models.values()]]
    for model in models.values():model.requires_grad_(True)
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:
        if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source clean required')
        clean=tensor(a[selected],device)
    finally:a._mmap.close()
    scale=varf.load_scale(args.root).to(device);objective=PairedObjective().to(device)
    with torch.no_grad():target=reader.features(clean[:2]).detach()
    for model in models.values():gradient_rows(model,reader,clean[:2],clean[:2],target,scale,objective)
    write(out/'smoke.json',dict(status='PASS',actual_source_Ls=2,frozen_G_models=2,all_three_objective_derivatives=True,
        no_accumulated_parameter_gradients=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('SOURCE_GRADIENT_REAL_SMOKE_PASS',flush=True)
    with torch.no_grad():target=reader.features(clean).detach()
    records=[]
    for epoch in range(1,21):
        folder=old/'epochs'/f'{epoch:02d}';done=read(folder/'complete.json')
        if (done['count'],done['TX_labels_read'],done['target_reads'])!=(63000,0,0):raise ValueError('Complete stored source draw required')
        a=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
        try:
            if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source received required')
            for offset in [0,256]:
                take=selected[offset:offset+256];observed=tensor(a[take],device)
                for arm,model in models.items():
                    groups,losses=gradient_rows(model,reader,observed,clean[offset:offset+256],target[offset:offset+256],scale,objective)
                    records.append(dict(arm=arm,epoch=epoch,offset=offset,count=256,source_indices=take.tolist(),groups=groups,objective_means=losses))
        finally:a._mmap.close()
        print('SOURCE_GRADIENT_DRAW_COMPLETE',epoch,512,2,flush=True)
    for model in models.values():model.requires_grad_(False)
    for model,before in zip([reader,*models.values()],snapshots):check_F0_state(model,before)
    summary=summarize(records)
    write(out/'gradient_records.json',records)
    write(out/'results.json',dict(status='SOURCE_GRADIENT_GEOMETRY_COMPLETE',run_id=RUN,code_commit=args.commit,F0_identity=identity,
        models=list(SPECS),objectives=list(OBJECTIVES),source_packets=512,L_s=128,U_s=384,draws=20,batches=80,batch_size=256,
        gradient_groups=['encoder','latent_head','FiLM','decoder','all'],weights=[.1,1.],summary=summary,
        G_forward_rows=20480,F0_output_feature_rows=20480,F0_clean_feature_rows=512,source2_smoke_extra=True,
        TX_labels_read=0,V_reads=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        parameter_gradients_accumulated=False,temporarily_enabled_G_derivatives_only=True,G_requires_grad_restored_false=True,
        source_in_sample=True,actual_training_step_statistics=False,AdamW_step_prediction=False,theoretical_ceiling=False,
        checkpoint_selection=False,default_DURA_changed=False,seconds=time.perf_counter()-started,goal_complete=False))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty owner required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir();parent=args.root/'runs'/strong.RUN
    write(out/'waiting.json',dict(status='WAIT_PARENT_COMPLETE',parent=strong.RUN,technical_completion_only=True))
    while not (parent/'complete.json').is_file():
        for p in parent.glob('terminal-*.json'):
            if read(p)['exit_code']!=0:raise RuntimeError('Parent technical failure; preserve/no automatic retry')
        time.sleep(5)
    if read(parent/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Complete parent Main required')
    choices=varf.recovery_gpu_choice()
    while not choices:time.sleep(5);choices=varf.recovery_gpu_choice()
    admission=choices[0];command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Source gradient diagnostic failed; preserve/no automatic retry')
    if read(out/'results.json')['status']!='SOURCE_GRADIENT_GEOMETRY_COMPLETE':raise ValueError('Missing complete gradient results')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git code identity required')
    if a.mode=='run':controller(a)
    else:worker(a)


if __name__=='__main__':main()
