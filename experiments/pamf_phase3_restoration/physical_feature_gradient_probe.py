"""Source-only gradient diagnostic for coupling frozen feature loss to channel estimation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as original
from . import multiview_trial as trial
from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .covariance_inverse import gaussian_posterior
from .covariance_model import source_action_loss
from .objective import PairedObjective
from .source_pair import read, validate_source
from .variance_trial import load_scale, recovery_gpu_choice

RUN = '20261108-diagnostic-Restore-PhysicalFeatureGradient-sourceLU-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.physical_feature_gradient_probe'
PARENT_RUN = '20261102-phase1-Restore-CovarianceDetail-E30-s2026092701-r01'
PARENT_COMMIT = '1fcbc3eb3cc16e7577ecc3cbedda99174b4c5cc6'
ARM = original.ARM
SCENES = tuple(trial.main_eval.VIEWS)


def select_rows(roles, scenes):
    if roles.shape != (63000,) or scenes.shape != roles.shape:
        raise ValueError('Exact source roles and scene vector required')
    selected = []
    for scene in SCENES:
        for role in ('L_s', 'U_s'):
            indices = np.flatnonzero((roles == role) & (scenes == scene))[:2]
            if len(indices) != 2: raise ValueError('Two original source records per role and scene required')
            selected.extend((int(i), role, scene) for i in indices)
    if len(selected) != 28 or len({i for i, _, _ in selected}) != 28:
        raise ValueError('Fixed unique 28 source records required')
    return selected


def connected_forward(model, received):
    aux = model.physical.estimate(received)
    base, residual = gaussian_posterior(aux['action_matrix'], received.flatten(1).double(),
        model.physical.prior_mean.double(), model.physical.prior_covariance.double(), aux['variance'].double())
    base = base.float().reshape_as(received)
    context = torch.cat((received, aux['variance'].log()[:, None, None].expand(-1, 1, 256)), 1)
    return model.detail(base, context), aux, residual


def gradient_vector(loss, parameters, retain_graph=True):
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    vectors = [torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
               for p, g in zip(parameters, values)]
    result = torch.cat(vectors)
    if not torch.isfinite(result).all(): raise ValueError('Nonfinite source gradient')
    return result


def pair_statistics(a, b):
    a, b = a.double(), b.double()
    na, nb = a.norm(), b.norm()
    if not torch.isfinite(na) or not torch.isfinite(nb): raise ValueError('Nonfinite gradient norm')
    ratio = float(nb / na) if na > 0 else None
    cosine = float((a @ b) / (na * nb)) if na > 0 and nb > 0 else None
    if ratio is not None and not np.isfinite(ratio): raise ValueError('Nonfinite gradient ratio')
    if cosine is not None and not np.isfinite(cosine): raise ValueError('Nonfinite gradient cosine')
    return dict(first_norm=float(na), second_norm=float(nb), ratio=ratio, cosine=cosine,
                first_zero=bool(na == 0), second_zero=bool(nb == 0))


def load_models(root, device):
    parent = root/'runs'/PARENT_RUN/'rows'/ARM/'training'
    done = read(parent/'complete.json')
    if (done['status'], done['code_commit'], done['epoch'], done['updates']) != ('SOURCE_TRAINED', PARENT_COMMIT, 30, 7410):
        raise ValueError('Original source-only final30 required')
    if any(done[k] for k in ('TX_labels_read','V_reads','OE_reads','target_training_reads')):
        raise ValueError('Original source-only model provenance required')
    state = torch.load(parent/'final.pt', map_location='cpu', weights_only=False)
    if (state['config'], state['code_commit'], state['epoch'], state['updates']) != (original.configuration(ARM), PARENT_COMMIT, 30, 7410):
        raise ValueError('Exact original model and fixed final checkpoint required')
    torch.manual_seed(2026092701)
    scratch = original.factory(ARM)
    original.control.bind_prior(root, scratch.physical)
    trained = original.factory(ARM)
    trained.load_state_dict(state['state'], strict=True)
    for model in (scratch, trained): model.to(device).eval().requires_grad_(True)
    return {'scratch_E0': scratch, 'fixed_source_E30': trained}, done


def worker(args):
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device='cuda:0'; config=validate_source(read(args.inputs)); old,roles=trial.source(args)
    reader,identity=load_F0(config,device); trial.main_eval.valid_identity(identity)
    reader.eval().requires_grad_(False); frozen={k:v.detach().clone() for k,v in reader.state_dict().items()}
    scale=load_scale(args.root).to(device); objective=PairedObjective().to(device)
    models,metadata=load_models(args.root,device)
    if metadata['F0_identity'] != identity: raise ValueError('Same original F0 required')
    model_states={name:{k:v.detach().clone() for k,v in model.state_dict().items()} for name,model in models.items()}
    folder=old/'epochs'/'01'; done=read(folder/'complete.json')
    if (done['count'],done['TX_labels_read'],done['target_reads']) != (63000,0,0): raise ValueError('Source draw contract required')
    with np.load(folder/'groups.npz',allow_pickle=False) as f: selected=select_rows(roles,f['scene'])
    arrays={name:np.load(path,mmap_mode='r',allow_pickle=False) for name,path in {
        'clean':old/'train/clean.npy','received':folder/'received.npy','kernel':folder/'kernel.npy',
        'coefficients':folder/'coefficients.npy','variance':folder/'variance.npy'}.items()}
    out=args.root/'runs'/RUN; rows=[]; start=time.perf_counter()
    try:
        for index,role,scene in selected:
            v={k:tensor(np.asarray(a[index:index+1]).copy(),device) for k,a in arrays.items()}
            with torch.no_grad(): target=reader.features(v['clean'])
            for name,model in models.items():
                with torch.no_grad(): reference=model(v['received'])
                restored,aux,residual=connected_forward(model,v['received'])
                torch.testing.assert_close(restored,reference,rtol=2e-5,atol=2e-6)
                if not torch.isfinite(restored).all() or residual.max()>1e-8: raise ValueError('Invalid source posterior')
                paired,_=objective(restored,v['clean'])
                features=reader.features(restored)
                f_loss=.01*((1-torch.nn.functional.cosine_similarity(features,target,dim=1,eps=1e-8))+
                             ((features-target).square()/scale).mean(1))
                physical,_=source_action_loss(aux,v['kernel'],v['coefficients'],v['variance'])
                heads=tuple(p for h in (model.physical.kernel_head,model.physical.coefficient_head,
                                        model.physical.noise_head) for p in h.parameters())
                a=gradient_vector(physical.mean(),heads); b=gradient_vector(paired.mean(),heads)
                c=gradient_vector(f_loss.mean(),heads,retain_graph=False)
                rows.append(dict(model=name,role=role,scene=scene,source_index=index,
                    physical_iq=pair_statistics(a,b),physical_feature=pair_statistics(a,c),
                    IQ_feature=pair_statistics(b,c),posterior_residual=float(residual.max()),
                    forward_max_abs=float((restored.detach()-reference).abs().max())))
                if len(rows)==2:
                    write(out/'smoke.json',dict(status='PASS',source_rows=1,source_models=2,
                        original_F0_features_only=True,output_matches_detached_path=True,
                        source_physics_and_feature_gradients_finite=True,optimizer_updates=0,
                        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
                    print('REAL_SOURCE_F0_GRADIENT_SMOKE_PASS',flush=True)
    finally:
        for a in arrays.values(): a._mmap.close()
    for name,model in models.items():
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in model_states[name].items()):
            raise ValueError('Diagnostic changed source model state')
    check_F0_state(reader,frozen)
    write(out/'results.json',dict(status='SOURCE_GRADIENT_DIAGNOSTIC_COMPLETE',run_id=RUN,
        source_draw=1,source_records=28,source_roles=['L_s','U_s'],source_scenes=list(SCENES),
        source_models=['scratch_E0','fixed_source_E30'],models_per_record=2,physical_head_parameters_only=True,
        F0_identity=identity,rows=rows,optimizer_updates=0,model_state_updates=0,
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,F0_logits_used=False,
        theoretical_limit=False,source_panel_representative=False,seconds=time.perf_counter()-start))
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',source_only=True,records=28,model_cases=56))
    print('SOURCE_GRADIENT_DIAGNOSTIC_COMPLETE',flush=True)


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CUDA-empty controller required')
    out=args.root/'runs'/RUN;out.mkdir();write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    choices=[]
    while not choices:
        choices=[g for g in recovery_gpu_choice(allowed_gpus=(2,),maximum_processes=2) if g['free_memory_MB']>=12288]
        if not choices:time.sleep(30)
    admission=choices[0];cmd=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,
        '--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'worker.log').open('xb') as log:
        child=subprocess.Popen(cmd,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=child.pid,command=cmd,CWD=os.getcwd(),admission=admission))
    failures=0;reason=None
    while child.poll() is None:
        try:
            observed=resource.compute_pids(admission['GPU'],admission['CUDA_ID']);failures=0
            if child.pid in observed and len(observed)>3:reason='TOTAL_COMPUTE_PID_CAP_EXCEEDED'
        except (RuntimeError,ValueError):
            failures+=1
            if failures>=3:reason='GPU_COUNT_UNVERIFIABLE'
        if reason:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            write(out/'resource_stop.json',dict(pid=child.pid,reason=reason,cap=3,only_own_child_signaled=True))
            break
        time.sleep(2)
    status=child.wait();write(out/'terminal.json',dict(pid=child.pid,exit_code=status))
    if status or reason: raise RuntimeError('Source diagnostic stopped or failed; preserve outputs')
    if read(out/'complete.json')['status']!='ARTIFACTS_COMPLETE': raise ValueError('Incomplete source diagnostic')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--commit',required=True)
    args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Release identity mismatch')
    controller(args) if args.mode=='run' else worker(args)


if __name__=='__main__':main()
