"""Fixed source-training-pair probes of an E02 conditional-variance branch."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .cache import tensor, load_F0, check_F0_state
from .conditioned_covariance import conditioned_posterior
from .conditioned_covariance_shared_trial import ARM, configuration, factory
from .covariance_detail import detail_objective
from .cost_probe import write
from .mismatched_prior_training import load_training_priors
from .objective import PairedObjective
from .source_inverse import R07
from .source_pair import read, validate_source
from .variance_trial import load_scale

RUN = '20261107-diagnostic-Restore-conditional-variance-E02-sourceLU-s2026092701-r01'
MODEL_RUN = '20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r02'
MODEL_COMMIT = 'ebb279ff3ad430e6a8242686d816976339444588'
VIEWS = ('clean','practical_high','practical_mid','practical_low_suburban',
         'practical_high_urban','practical_mid_urban','practical_low_urban')
RX = (1,3,4,6,8)
CASES = ('learned','zero','initial_fraction','moderate_fraction')
INITIAL = .01 * float(torch.nn.functional.softplus(torch.tensor(0.)))


def select_source(rx, scene):
    if rx.shape != (63000,) or scene.shape != (63000,) or set(rx.tolist()) != set(RX):
        raise ValueError('Original five-receiver source draw required')
    chosen = []
    for view in VIEWS:
        for receiver in RX:
            indices = np.flatnonzero((scene == view) & (rx == receiver))
            if len(indices) < 2:
                raise ValueError('Two source records per fixed receiver/scene required')
            chosen.extend(indices[:2].tolist())
    return np.asarray(chosen, dtype=np.int64)


def probe(action, observed, mean, covariance, noise, raw, loss_fn):
    """Differentiate diagnostic input variables, never model parameters."""
    diagonal = covariance.diagonal(dim1=-2,dim2=-1).expand(len(observed),-1)
    result = {}
    current = None
    for case in CASES:
        if case == 'learned':
            variable = raw.detach().double().clone().requires_grad_(True)
            fraction = .01 * torch.nn.functional.softplus(variable)
        else:
            value = {'zero':0., 'initial_fraction':INITIAL, 'moderate_fraction':.1}[case]
            variable = torch.full_like(diagonal,value,requires_grad=(case == 'zero'))
            fraction = variable
        restored, residual = conditioned_posterior(action,observed,mean,covariance,noise,diagonal*fraction)
        if residual.max() > 1e-8:
            raise ValueError('Source posterior residual exceeds registered tolerance')
        loss, metrics = loss_fn(restored)
        if loss.shape != (len(observed),) or not torch.isfinite(loss).all():
            raise ValueError('Finite per-source-record objective required')
        values = dict(loss=loss, fraction_mean=fraction.mean(1), **metrics)
        if case in ('learned','zero'):
            gradient, = torch.autograd.grad(loss.sum(), variable)
            if not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite diagnostic local derivative')
            values['gradient_abs_mean'] = gradient.abs().mean(1)
            values['uniform_direction_derivative'] = gradient.sum(1)
            values['negative_coordinate_fraction'] = (gradient < -1e-12).double().mean(1)
        if case == 'learned':
            current = restored.detach()
            values.update(raw_logit_mean=raw.mean(1), raw_logit_min=raw.min(1).values,
                          raw_logit_max=raw.max(1).values, softplus_slope_mean=torch.sigmoid(raw).mean(1))
        result[case] = {key:value.detach().tolist() for key,value in values.items()}
    return result, current


def raw_logits(model, received, mean, covariance, physical):
    mean = mean.expand(len(received),-1)
    diagonal = covariance.diagonal(dim1=-2,dim2=-1).expand(len(received),-1)
    innovation = received.flatten(1).double() - (physical['action_matrix'] @ mean.double()[:,:,None]).squeeze(-1)
    captured = []
    handle = model.detail.output.register_forward_hook(lambda module,args,value: captured.append(value))
    try:
        with torch.no_grad():
            model.detail(received,mean.float().reshape_as(received),diagonal.float().reshape_as(received),
                         innovation.float().reshape_as(received),physical['variance'])
    finally:
        handle.remove()
    return captured[0].flatten(1).detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('Pinned CPU-only release required')
    os.nice(10)
    torch.set_num_threads(2)
    config = validate_source(read(args.inputs))
    cache = args.root/'runs'/R07/'cache'
    done = read(cache/'complete.json')
    if (done['status'],done['train_count'],done['TX_labels_read'],done['target_reads']) != ('ACTION_SOURCE_CACHE_COMPLETE',63000,0,0):
        raise ValueError('Original completed source cache required')
    model = factory(ARM).eval().requires_grad_(False)
    path = args.root/'runs'/MODEL_RUN/'rows'/ARM/'training/recovery/E02.pt'
    saved = torch.load(path,map_location='cpu',weights_only=False)
    if (saved['code_commit'],saved['epoch'],saved['updates'],saved['config']) != (MODEL_COMMIT,2,494,configuration(ARM)):
        raise ValueError('Exact source E02 diagnostic state required')
    model.load_state_dict(saved['state'],strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Finite fixed G state required')
    before = {key:value.clone() for key,value in model.state_dict().items()}
    reader, identity = load_F0(config,'cpu')
    frozen = {key:value.clone() for key,value in reader.state_dict().items()}
    scale = load_scale(args.root)
    paired = PairedObjective()
    priors, assignment = load_training_priors(args.root,'cpu')
    rx = np.load(cache/'train/rx.npy',allow_pickle=False)
    roles = np.load(cache/'train/roles.npy',allow_pickle=False)
    if len(roles) != 63000 or dict(zip(*np.unique(roles,return_counts=True))) != {'L_s':6300,'U_s':56700}:
        raise ValueError('Original source L/U roles required')
    draw = cache/'epochs/02'
    draw_done = read(draw/'complete.json')
    if (draw_done['count'],draw_done['TX_labels_read'],draw_done['target_reads']) != (63000,0,0):
        raise ValueError('Original completed second source draw required')
    with np.load(draw/'groups.npz',allow_pickle=False) as groups:
        scenes = groups['scene'].copy()
    indices = select_source(rx,scenes)
    arrays = []
    for path in (cache/'train/clean.npy',draw/'received.npy'):
        array = np.load(path,mmap_mode='r',allow_pickle=False)
        try:
            if array.shape != (63000,2,256) or array.dtype != np.float32:
                raise ValueError('Original source IQ layout required')
            arrays.append(tensor(array[indices.tolist()],'cpu'))
        finally:
            array._mmap.close()
    clean, received = arrays
    if not all(torch.isfinite(value).all() for value in arrays):
        raise ValueError('Finite selected source IQ required')
    with torch.no_grad():
        features = reader.features(clean)
    out = args.root/'runs'/RUN
    out.mkdir()
    np.savez(out/'reference.npz',source_indices=indices,rx=rx[indices],roles=roles[indices],scene=scenes[indices])

    def batch(take, mode):
        x, y, target = received[take],clean[take],features[take]
        with torch.no_grad():
            physical = model.physical.estimate(x)
        if mode == 'global':
            mean,covariance = model.physical.prior_mean,model.physical.prior_covariance
        else:
            source_indices = torch.tensor(indices[take].tolist(),dtype=torch.long)
            chosen = assignment[source_indices]
            mean,covariance = tuple(value[chosen] for value in priors)
        raw = raw_logits(model,x,mean,covariance,physical)
        def objective(restored):
            return detail_objective(reader,restored.float().reshape_as(x),y,target,scale,paired)
        values, current = probe(physical['action_matrix'].double(),x.flatten(1).double(),mean.double(),
                                covariance.double(),physical['variance'].double(),raw,objective)
        return values,current

    smoke_indices = np.flatnonzero(scenes[indices] != 'clean')[:2].tolist()
    smoke, current = batch(smoke_indices,'global')
    with torch.no_grad():
        actual = model(received[smoke_indices]).flatten(1).double()
    torch.testing.assert_close(current,actual,rtol=2e-5,atol=2e-6)
    check_F0_state(reader,frozen)
    write(out/'smoke.json',dict(status='PASS',actual_source_rows=2,actual_E02_G=True,actual_original_F0=True,
          current_readout_matches_G=True,local_derivatives_finite=True,optimizer_updates=0,
          TX_labels_read=0,V_reads=0,OE_reads=0,target_reads=0))
    print('CONDITIONAL_VARIANCE_SOURCE_SMOKE_PASS',flush=True)
    records = {}
    for mode in ('global','other_receiver'):
        cells = {case:{} for case in CASES}
        for start in range(0,len(indices),4):
            take = list(range(start,min(start+4,len(indices))))
            values,_ = batch(take,mode)
            for case, metrics in values.items():
                for key, value in metrics.items():
                    cells[case].setdefault(key,[]).extend(value)
            print('CONDITIONAL_VARIANCE_PROGRESS',mode,start+len(take),len(indices),flush=True)
        records[mode] = cells
    if any(not torch.equal(value,model.state_dict()[key]) for key,value in before.items()):
        raise ValueError('Frozen G changed during source diagnostic')
    check_F0_state(reader,frozen)
    write(out/'records.json',records)
    summary = {}
    for mode,cells in records.items():
        summary[mode] = {}
        for view in VIEWS:
            take = np.flatnonzero(scenes[indices] == view)
            summary[mode][view] = {case:{key:float(np.asarray(value)[take].mean()) for key,value in metrics.items()}
                                    for case,metrics in cells.items()}
    result = dict(status='SOURCE_CONDITIONAL_VARIANCE_DIAGNOSTIC_COMPLETE',code_commit=args.commit,
        model_run=MODEL_RUN,model_code=MODEL_COMMIT,model_epoch=2,model_updates=494,
        F0_identity=identity,source_records=len(indices),source_rows_per_RX_scene=2,source_draw=2,
        fixed_prior_modes=['global','other_receiver'],fixed_variance_conditions=list(CASES),
        fixed_constant_fractions=[0.,INITIAL,.1],summary=summary,
        optimizer_updates=0,G_and_F0_unchanged=True,TX_labels_read=0,V_reads=0,OE_reads=0,target_reads=0,
        claim_scope='Post-fit source-training-pair mechanism only, no held-out accuracy, TX score, checkpoint selection, or Main ceiling.')
    write(out/'results.json',result)
    write(out/'complete.json',dict(status='COMPLETE',code_commit=args.commit,source_records=len(indices),
                                 fixed_cells=56,TX_labels_read=0,target_reads=0))
    print('CONDITIONAL_VARIANCE_SOURCE_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
