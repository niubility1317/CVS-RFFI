"""CPU source-only equivalence check using the fixed original JointPrior E30 G."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .covariance_detail import clip_branch_gradients
from .covariance_model import source_action_loss
from . import feature_response_trial as feature
from .feature_response_loss import feature_response_loss
from .feature_response_reuse_probe import cache_source_features
from . import joint_prior_trial as parent
from .main_split_eval import valid_identity
from .objective import PairedObjective
from .response_joint_prior import response_loss, selected_perturbation
from .source_inverse import R07
from .source_pair import read, validate_source
from .variance_trial import load_scale

RUN = '20261108-diagnostic-Restore-FeatureReferenceCache-sourceLU-s2026092701-r01'
PARENT = '20261107-phase1-Restore-JointPrior-E30-s2026092701-r01'
PARENT_COMMIT = '4bba9df3bb2c909dc333dce0cf0102cb641b1682'
F0_COMMIT = '3ff8f4b7e85b5608f5b45350aa18081499a7ef1d'


class ReaderCapture:
    def __init__(self, reader):
        self.reader = reader
        self.last = None
        self.inputs = []

    def features(self, value):
        if value.requires_grad:
            value.retain_grad()
            if all(value is not old for old in self.inputs):
                self.inputs.append(value)
        self.last = self.reader.features(value)
        return self.last


def shared_readout(model, capture, received, clean, targets, scale, objective, indices, epoch, trace, cache):
    if epoch % 2 != 1 or not torch.equal(trace['received'], received):
        raise ValueError('Only fixed global-prior odd-epoch source cases allowed')
    base, aux, loss, _ = feature.original.control.joint_readout(model, capture, received, clean, targets,
        scale, objective, source_indices=indices, epoch=epoch, priors=None, assignment=None)
    fbase = capture.features(base)
    delta, choice = selected_perturbation(clean, indices, epoch)
    kernel, main, image, _ = feature.original.known_parameters(trace)
    action = feature.original.dense_action(kernel, main, image)
    changed = feature.original.inject(received, delta, action)
    altered, altered_aux = model.source_readout(changed, prior=None)
    iq_response = response_loss(base, altered, delta)
    fchanged = capture.features(altered)
    changed_source = cache[torch.arange(len(clean), device=clean.device), choice]
    response = feature_response_loss(fbase, fchanged, targets, changed_source, scale)
    aux['relative_solve_residual'] = torch.maximum(aux['relative_solve_residual'], altered_aux['relative_solve_residual'])
    return base, aux, loss + feature.original.RESPONSE_WEIGHT*iq_response + feature.FEATURE_RESPONSE_WEIGHT*response


def comparison(state, reader, clean, trace, indices, target, cache, scale, epoch):
    if epoch not in (1, 3):
        raise ValueError('Fixed odd-epoch four-direction matrix required')
    objective = PairedObjective().to('cpu')
    values = []
    for shared in (False, True):
        model = feature.factory(feature.ARM).to('cpu').train()
        model.load_state_dict(state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        capture = ReaderCapture(reader)
        torch.manual_seed(2026092701)
        if shared:
            base, aux, detail = shared_readout(model, capture, trace['received'], clean, target,
                scale, objective, indices, epoch, trace, cache)
        else:
            base, aux, detail, _ = feature.response_readout(model, capture, trace['received'], clean,
                target, scale, objective, source_indices=indices, epoch=epoch, priors=None,
                assignment=None, source_trace=trace)
        physical, _ = source_action_loss(aux, trace['kernel'], trace['coefficients'], trace['variance'])
        loss = (physical + detail).mean()
        if len(capture.inputs) != 2 or not torch.isfinite(loss) or aux['relative_solve_residual'].max() > 1e-8:
            raise ValueError('Two differentiable F0 inputs and finite valid posterior required')
        loss.backward()
        gradients = {k:None if p.grad is None else p.grad.detach().clone() for k,p in model.named_parameters()}
        input_gradients = [v.grad.detach().clone() for v in capture.inputs]
        clip_branch_gradients(model)
        clipped = {k:None if p.grad is None else p.grad.detach().clone() for k,p in model.named_parameters()}
        optimizer.step()
        updated = {k:p.detach().clone() for k,p in model.named_parameters()}
        values.append((loss.detach(), gradients, input_gradients, updated, clipped))
    a, b = values
    if not torch.isfinite(a[0]) or not torch.isfinite(b[0]):
        raise ValueError('Finite original and shared loss required')
    grad_max = 0.; parameter_max = 0.; present = 0; grad_bad = 0; parameter_bad = 0
    bad_layers = []; worst = None
    for name in a[1]:
        x, y = a[1][name], b[1][name]
        if (x is None) != (y is None):
            raise ValueError('Production parameter gradient presence differs')
        if x is not None:
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('Finite production gradients required')
            grad_max = max(grad_max, float((x-y).abs().max())); present += 1
            grad_bad += int((~torch.isclose(x,y,rtol=1e-5,atol=1e-6)).sum())
        left,right = a[3][name],b[3][name]
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError('Finite production update required')
        diff=(left-right).abs(); d=float(diff.max()); bad=int((~torch.isclose(left,right,rtol=1e-5,atol=1e-6)).sum())
        parameter_bad+=bad
        if bad:bad_layers.append(dict(name=name,notclose=bad,elements=left.numel(),max_abs=d))
        if d>parameter_max:
            parameter_max=d; i=int(diff.flatten().argmax())
            ga,gb=a[4][name].flatten()[i].item(),b[4][name].flatten()[i].item()
            pa,pb=a[1][name].flatten()[i].item(),b[1][name].flatten()[i].item()
            predicted=.001*(ga/(abs(ga)+1e-8)-gb/(abs(gb)+1e-8))
            worst=dict(name=name,flat_index=i,preclip_gradient_original=pa,preclip_gradient_shared=pb,
                clipped_gradient_original=ga,clipped_gradient_shared=gb,AdamW_epsilon=1e-8,
                expected_first_step_update_difference_abs=abs(predicted),observed_parameter_difference_abs=d)
    if not present:raise ValueError('No production G gradients compared')
    input_max=max(float((x-y).abs().max()) for x,y in zip(a[2],b[2]))
    input_bad=sum(int((~torch.isclose(x,y,rtol=1e-5,atol=1e-6)).sum()) for x,y in zip(a[2],b[2]))
    return dict(epoch=epoch,loss_original=float(a[0]),loss_shared=float(b[0]),
        loss_max_abs=float((a[0]-b[0]).abs()),input_gradient_max_abs=input_max,input_gradient_notclose=input_bad,
        parameter_gradient_max_abs=grad_max,parameter_gradient_notclose=grad_bad,compared_parameter_gradients=present,
        updated_parameter_max_abs=parameter_max,updated_parameter_notclose=parameter_bad,
        updated_parameter_outside_tolerance_layers=bad_layers,worst_update=worst,
        original_one_step_tolerance_pass=bool(parameter_bad==0 and grad_bad==0 and input_bad==0 and torch.isclose(a[0],b[0],rtol=1e-5,atol=1e-6)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('CPU-only original code identity required')
    torch.set_num_threads(2); os.nice(10); started=time.perf_counter()
    cfg=validate_source(read(args.inputs)); reader,identity=load_F0(cfg,'cpu'); valid_identity(identity)
    if identity['source_code_commit']!=F0_COMMIT or any(m.training for m in reader.modules()) or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Exact frozen source F0 required')
    fstate={k:v.detach().clone() for k,v in reader.state_dict().items()}
    parent_dir=args.root/'runs'/PARENT/'rows'/'TF_JointPrior'/'training'
    done=read(parent_dir/'complete.json')
    if (done['status'],done['code_commit'],done['epoch'],done['updates'])!=('SOURCE_TRAINED',PARENT_COMMIT,30,7410):
        raise ValueError('Fixed original source-only E30 parent required')
    if any(done[k] for k in ('TX_labels_read','V_reads','OE_reads','target_training_reads','G_initialization_checkpoint_loads')):
        raise ValueError('Source-only parent training provenance required')
    if done['F0_identity']['source_code_commit']!=F0_COMMIT:
        raise ValueError('Parent F0 source identity differs')
    checkpoint=torch.load(parent_dir/'final.pt',map_location='cpu',weights_only=False)
    if (checkpoint['code_commit'],checkpoint['epoch'],checkpoint['updates'],checkpoint['config']) != (
            PARENT_COMMIT,30,7410,parent.configuration(parent.ARM)):
        raise ValueError('Exact preregistered source-only parent state required')
    state=checkpoint['state']
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Finite parent state required')
    source=args.root/'runs'/R07/'cache'; meta=read(source/'complete.json')
    if (meta['status'],meta['train_count'],meta['TX_labels_read'],meta['target_reads']) != (
            'ACTION_SOURCE_CACHE_COMPLETE',63000,0,0):raise ValueError('Original source cache required')
    selected=[0,6301]; roles=np.load(source/'train/roles.npy',allow_pickle=False)
    if [str(roles[i]) for i in selected]!=['L_s','U_s']:raise ValueError('One legal L_s and U_s required')
    def rows(path):
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        try:return tensor(a[selected],'cpu')
        finally:a._mmap.close()
    clean=rows(source/'train/clean.npy'); folder=source/'epochs'/'01'
    trace={k:rows(folder/(k+'.npy')) for k in ('received','kernel','coefficients','variance')}
    indices=torch.tensor(selected,dtype=torch.long);scale=load_scale(args.root)
    with torch.no_grad():target=reader.features(clean).detach()
    cache=cache_source_features(reader,clean,indices)
    feature_max=0.
    with torch.no_grad():
        for epoch in (1,3):
            delta,choice=selected_perturbation(clean,indices,epoch)
            online=reader.features(clean+delta)
            cached=cache[torch.arange(len(clean)),choice]
            torch.testing.assert_close(online,cached,rtol=1e-5,atol=1e-6)
            feature_max=max(feature_max,float((online-cached).abs().max()))
    results=[comparison(state,reader,clean,trace,indices,target,cache,scale,epoch) for epoch in (1,3)]
    check_F0_state(reader,fstate)
    out=args.root/'runs'/RUN;out.mkdir()
    result=dict(status='MEASURED',code_commit=args.commit,F0_source_commit=F0_COMMIT,
        parent_run=PARENT,parent_commit=PARENT_COMMIT,parent_epoch=30,parent_updates=7410,
        production_G='Original JointPrior E30 state in identical FeatureResponse G architecture',
        source_rows=2,source_roles=['L_s','U_s'],source_indices_mod4=[0,1],
        source_epoch_cases=[1,3],covered_directions=4,prior='global source prior only',
        G_test_updates=4,G_saved_or_running_weights_updated=False,optimizer='fresh local AdamW',
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,CPU_threads=2,
        tolerance='rtol1e-5/atol1e-6',feature_max_abs=feature_max,
        optimization='cache_source_reference_only_keep_repeated_restored_F0',
        GPU_equivalence_verified=False,training_speed_measured=False,
        original_one_step_tolerance_pass=all(q['original_one_step_tolerance_pass'] for q in results),
        cases=results,seconds=time.perf_counter()-started)
    with (out/'results.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    with (out/'complete.json').open('x',encoding='utf-8') as f:json.dump({'status':'ARTIFACTS_COMPLETE'},f)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
