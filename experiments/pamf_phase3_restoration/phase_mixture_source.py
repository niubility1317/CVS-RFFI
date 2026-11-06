"""Frozen source comparison of coherent IQ phase priors with simulator Action."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .coordinate_map_detail import coordinate_prior, operator_in_original_frame
from .cost_probe import write
from .covariance_detail_source_validation import RUN as REFERENCE_RUN
from .covariance_guidance_source import source_metrics
from .covariance_inverse import dense_action, gaussian_posterior
from .covariance_trial import load_prior, PRIOR_COMMIT
from .main_split_eval import valid_identity
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source

RUN = '20261107-diagnostic-Restore-C4-phase-mixture-sourceV-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.phase_mixture_source'
REFERENCE_COMMIT = 'f29b31417ce98a280eac25ea2fa78e0c49a5f0ff'
METHODS = ('original_gaussian', 'uniform_C4_gaussian', 'shared_fixed_mean', 'shared_phase_mixture')
TRACE_KEYS = ('received', 'kernel', 'coefficients', 'variance')


def quarter_turn(vector):
    i, q = vector.chunk(2, dim=-1)
    return torch.cat((-q, i), dim=-1)


def shared_covariance(covariance):
    return .5 * (covariance + operator_in_original_frame(covariance, 1))


def phase_mixture_posterior(action, observed, mean, within_covariance, variance):
    """Exact four-mean Gaussian mixture posterior using a common covariance.

    Whiten y, A mu, A R mu together. The other two projected means are
    their negatives. Gaussian evidence differs only in the mean term.
    """
    ac = action @ within_covariance
    observation_covariance = ac @ action.transpose(-1, -2)
    observation_covariance.diagonal(dim1=-2, dim2=-1).add_(variance[:, None])
    anchors = torch.stack((mean, quarter_turn(mean)), dim=-1)
    projected = action @ anchors
    rhs = torch.cat((observed[:, :, None], projected), dim=-1)
    solved = torch.cholesky_solve(rhs, torch.linalg.cholesky(observation_covariance))
    filtered = (ac.transpose(-1, -2) @ solved)
    base = filtered[:, :, 0]
    innovations = anchors - filtered[:, :, 1:]
    evidence = (projected * solved[:, :, :1]).sum(dim=1)
    energy = (projected * solved[:, :, 1:]).sum(dim=1)
    weights = (torch.cat((evidence, -evidence), dim=1) - .5*energy.repeat(1, 2)).softmax(dim=1)
    contrast = weights[:, :2] - weights[:, 2:]
    mixed = base + (innovations @ contrast[:, :, None]).squeeze(-1)
    return mixed, base + innovations[:, :, 0], weights


def source_indices(rx, day):
    if rx.shape != (1800,) or day.shape != rx.shape or set(rx.tolist()) != {1,3,4,6,8} or set(day.tolist()) != {1,2,3}:
        raise ValueError('Original 1800 source V with five receivers and three days required')
    groups = [np.flatnonzero((rx == r) & (day == d)) for r in (1,3,4,6,8) for d in (1,2,3)]
    if any(len(group) != 120 for group in groups):
        raise ValueError('Original 120 records per source RX/day stratum required')
    return np.concatenate([group[:8] for group in groups])


def read_trace(folder, indices, count):
    meta = read(folder/'complete.json')
    if (meta['count'],meta['TX_labels_read'],meta['target_reads']) != (count,0,0):
        raise ValueError('Original complete source-only simulator trace required')
    values = {}
    for key,shape in zip(TRACE_KEYS, [(count,2,256),(count,2,64),(count,4,256),(count,)]):
        a = np.load(folder/(key+'.npy'), mmap_mode='r', allow_pickle=False)
        try:
            if a.shape != shape or a.dtype != np.float32:
                raise ValueError('Exact R07 source trace layout required')
            values[key] = tensor(a[indices], 'cpu')
        finally:
            a._mmap.close()
    if (values['variance'] <= 0).any() or not all(torch.isfinite(x).all() for x in values.values()):
        raise ValueError('Finite R07 trace with positive variance required')
    return values


def action_inputs(trace):
    h = torch.complex(*trace['kernel'].double().unbind(1))
    coeff = trace['coefficients'].double()
    main = torch.complex(coeff[:,0], coeff[:,1])
    image = torch.complex(coeff[:,2], coeff[:,3])
    return dense_action(h, main, image), trace['received'].flatten(1).double(), trace['variance'].double()


def posterior_readouts(trace, mean, covariance, within, uniform):
    action, observed, variance = action_inputs(trace)
    original, r_original = gaussian_posterior(action, observed, mean, covariance, variance)
    gaussian, r_uniform = gaussian_posterior(action, observed, *uniform, variance)
    if torch.maximum(r_original.max(), r_uniform.max()) > 1e-8:
        raise ValueError('Invalid Gaussian linear solve')
    mixture, anchored, weights = phase_mixture_posterior(action, observed, mean, within, variance)
    outputs = {name: value.float().reshape(-1,2,256)
               for name,value in zip(METHODS, (original,gaussian,anchored,mixture))}
    if not all(torch.isfinite(value).all() for value in (*outputs.values(),weights)):
        raise ValueError('Nonfinite IQ posterior or phase weights')
    return outputs, weights


def worker(args, out):
    start = time.perf_counter()
    F, identity = load_F0(validate_source(read(args.inputs)), 'cpu')
    valid_identity(identity)
    snapshot = {name:value.detach().clone() for name,value in F.state_dict().items()}
    mean, covariance = load_prior(args.root)
    mean = torch.frombuffer(memoryview(mean), dtype=torch.float64)
    covariance = torch.frombuffer(memoryview(covariance), dtype=torch.float64).reshape(512,512)
    within = shared_covariance(covariance)
    uniform = coordinate_prior(mean, covariance)
    old = args.root/'runs'/R07/'cache'
    roles = np.load(old/'train/roles.npy', allow_pickle=False)
    take = np.flatnonzero(roles == 'L_s')[:2]
    if len(take) != 2:
        raise ValueError('Two original source L_s smoke records required')
    # First neural work uses real source records and the actual frozen reader.
    with torch.inference_mode():
        trace = read_trace(old/'epochs/01', take, 63000)
        outputs, weights = posterior_readouts(trace, mean, covariance, within, uniform)
        for iq in outputs.values():
            logits = F.classify_features(F.features(iq))
            if logits.shape != (2,6) or not torch.isfinite(logits).all():
                raise ValueError('Actual original F0 CPU smoke failed')
        a, y, variance = action_inputs(trace)
        rotated, _, rw = phase_mixture_posterior(operator_in_original_frame(a,1), quarter_turn(y),
                                                    mean, within, variance)
        torch.testing.assert_close(rotated.float(), quarter_turn(outputs['shared_phase_mixture'].flatten(1)),
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(rw, weights.roll(1,dims=1), rtol=1e-9, atol=1e-10)
    write(out/'smoke.json', dict(status='PASS', actual_Ls=2, actual_original_F0=True, methods=list(METHODS),
        F0_rows=8, quarter_turn_IQ_and_weights_checked=True, CPU_only=True, TX_labels_read=0, target_inputs=0))
    print('SOURCE_PHASE_MIXTURE_REAL_SMOKE_PASS', flush=True)
    reference = args.root/'runs'/REFERENCE_RUN
    meta = read(reference/'predictions_complete.json')
    if (meta['status'],meta['code_commit'],meta['source_V_packets'],meta['TX_labels_read'],meta['target_inputs']) != (
        'SOURCE_V_PREDICTIONS_COMPLETE',REFERENCE_COMMIT,1800,0,0) or meta['F0_identity'] != identity:
        raise ValueError('Exact source V clean reference and frozen F0 required')
    with np.load(reference/'references.npz',allow_pickle=False) as f:
        take = source_indices(f['rx'], f['day'])
        if not np.array_equal(f['ids'],np.load(old/'validation/physical_ids.npy',allow_pickle=False)):
            raise ValueError('Exact saved original source V order required')
        ref = {key:f[key][take] for key in ['ids','rx','day','clean','features','logits']}
        ref['scale'] = f['scale']
    with (out/'reference.npz').open('xb') as f:
        np.savez(f, **ref, source_index=take)
    clean, target, clean_logits, scale = [tensor(ref[k],'cpu') for k in ['clean','features','logits','scale']]
    summary = []
    with torch.inference_mode():
        for scene in SCENES:
            trace = read_trace(old/'validation_actions'/scene, take, 1800)
            records = {method:[] for method in METHODS}
            weights = []
            for offset in range(0,len(take),16):
                view = slice(offset,offset+16)
                outputs, phase = posterior_readouts({k:v[view] for k,v in trace.items()},mean,covariance,within,uniform)
                weights.extend(phase.tolist())
                for name, iq in outputs.items():
                    features = F.features(iq)
                    logits = F.classify_features(features)
                    metrics = source_metrics(iq,clean[view],features,logits,target[view],clean_logits[view],scale)
                    values = dict(metrics,iq=iq,features=features,logits=logits)
                    if not all(torch.isfinite(value).all() for value in values.values()):
                        raise ValueError('Nonfinite source metrics')
                    records[name].append({k:v.tolist() for k,v in values.items()})
            arrays = {name+'__'+key:np.concatenate([batch[key] for batch in batches],axis=0)
                      for name,batches in records.items() for key in batches[0]}
            weights = np.asarray(weights)
            with (out/(scene+'.npz')).open('xb') as f:
                np.savez(f, **arrays, phase_weights=weights)
            baseline = arrays['uniform_C4_gaussian__clean_decision_agreement'].astype(bool)
            for name in METHODS:
                agreement = arrays[name+'__clean_decision_agreement'].astype(bool)
                summary.append(dict(scene=scene,method=name,count=len(take),
                    iq_nmse=float(arrays[name+'__iq_nmse'].mean()),
                    feature_cosine_distance=float(arrays[name+'__feature_cosine_distance'].mean()),
                    Fclean_agreement=float(agreement.mean()),
                    vs_uniform_repairs=int((~baseline & agreement).sum()), vs_uniform_damage=int((baseline & ~agreement).sum()),
                    nominal_phase_mass=float(weights[:,0].mean()), maximum_phase_mass=float(weights.max(1).mean())))
            print('SOURCE_PHASE_MIXTURE_VIEW_COMPLETE',scene,len(take),flush=True)
    check_F0_state(F,snapshot)
    write(out/'results.json', dict(status='SOURCE_PHASE_MIXTURE_COMPLETE',run_id=RUN,code_commit=args.commit,
        prior_code_commit=PRIOR_COMMIT,reference_code_commit=REFERENCE_COMMIT,methods=list(METHODS),views=list(SCENES),
        records=summary,source_V_records=120,source_V_views=840,F0_new_rows=3360,F0_smoke_rows=8,F0_clean_rows_reused=120,
        source_true_simulator_Action=True,source_true_simulator_scalar_noise=True,
        CPU_threads=2,TX_labels_read=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,
        model_state_updates=0,new_deployment_candidate=False,theoretical_ceiling=False,seconds=time.perf_counter()-start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('CUDA-empty source diagnostic with actual code identity required')
    torch.set_num_threads(2)
    out = args.root/'runs'/RUN
    out.mkdir()
    write(out/'launch.json', dict(pid=os.getpid(),code_commit=args.commit,CPU_threads=2,CUDA_empty=True))
    worker(args,out)
    write(out/'complete.json', dict(status='ARTIFACTS_COMPLETE',TX_labels_read=0,target_inputs=0,optimizer_updates=0))


if __name__ == '__main__':
    main()
