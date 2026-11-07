"""Compare moment-matched IQ priors on the original frozen source panel."""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .covariance_guidance_source import source_metrics
from .covariance_inverse import dense_action, gaussian_posterior
from .covariance_parameter_source import G_COMMIT, load_model
from .main_split_eval import valid_identity
from .phase_mixture_source import REFERENCE_COMMIT, read_trace
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source
from .structured_noise_source import known_parameters, load_reference
from .waveform_mixture_prior import fit_source_modes, mixture_posterior


RUN = '20261107-diagnostic-Restore-waveform-mixture-sourceV-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.waveform_mixture_source'
METHODS = ('blind_gaussian', 'blind_mixture', 'known_gaussian', 'known_mixture')


def fit_prior(cache, out, mean, covariance):
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    if roles.shape != (63000,) or (roles == 'L_s').sum() != 6300 or (roles == 'U_s').sum() != 56700:
        raise ValueError('Only original source L_s/U_s fit rows are legal')
    iq = np.load(cache/'train/clean.npy', mmap_mode='r', allow_pickle=False)
    try:
        if iq.shape != (63000, 2, 256) or iq.dtype != np.float32:
            raise ValueError('Exact source clean IQ required')
        offsets, within, weights = fit_source_modes(iq.reshape(63000, 512), mean, covariance)
    finally:
        iq._mmap.close()
    between = (offsets.T * weights) @ offsets
    mean_error = float(np.max(np.abs(weights @ offsets)))
    covariance_error = float(np.max(np.abs(within + between - covariance)))
    if max(mean_error, covariance_error) > 1e-12:
        raise ValueError('Mixture must match original Gaussian moments')
    with (out/'waveform_prior.npz').open('xb') as file:
        np.savez(file, offsets=offsets, within=within, weights=weights)
    prior_report = dict(source_Ls=6300, source_Us=56700, whitened_PCs=16, lloyd_steps=8,
        occupancy=[round(w * len(roles)) for w in weights], mean_max_error=mean_error,
        covariance_max_error=covariance_error, between_trace_fraction=float(between.trace()/covariance.trace()),
        within_min_eigenvalue=float(np.linalg.eigvalsh(within)[0]))
    prior = tuple(torch.frombuffer(memoryview(a), dtype=torch.float64).reshape(a.shape)
                  for a in (within, offsets, weights))
    return prior, prior_report, np.flatnonzero(roles == 'L_s')[:2]


def readouts(G, trace, prior):
    observed = trace['received'].flatten(1).double()
    mean, covariance = G.prior_mean.double(), G.prior_covariance.double()
    est = G.estimate(trace['received'], dense=False)
    blind = tuple(est[name].to(torch.complex128) for name in ('kernel', 'main', 'image'))
    blind += (est['variance'].double(),)
    outputs = {}
    for prefix, (kernel, main, image, variance) in [('blind', blind), ('known', known_parameters(trace))]:
        action = dense_action(kernel, main, image)
        gaussian, residual = gaussian_posterior(action, observed, mean, covariance, variance)
        mixed, probabilities = mixture_posterior(action, observed, mean, *prior, variance)
        if residual.max() > 1e-8 or not torch.isfinite(mixed).all() or not torch.isfinite(probabilities).all():
            raise ValueError('Invalid source mixture comparison')
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones_like(variance), rtol=0, atol=1e-12)
        outputs[prefix+'_gaussian'] = dict(iq=gaussian.float().reshape_as(trace['received']))
        outputs[prefix+'_mixture'] = dict(iq=mixed.float().reshape_as(trace['received']), probabilities=probabilities)
    return outputs


def worker(args, out):
    start = time.perf_counter()
    G = load_model(args.root, 'cpu')
    cache = args.root/'runs'/R07/'cache'
    prior, prior_report, source2 = fit_prior(cache, out, np.array(G.prior_mean.tolist(), dtype=np.float64),
        np.array(G.prior_covariance.tolist(), dtype=np.float64))
    F, identity = load_F0(validate_source(read(args.inputs)), 'cpu')
    valid_identity(identity)
    models = (G, F)
    snapshots = [{name: value.detach().clone() for name, value in model.state_dict().items()} for model in models]
    with torch.inference_mode():
        trace = read_trace(cache/'epochs/01', source2, 63000)
        outputs = readouts(G, trace, prior)
        torch.testing.assert_close(outputs['blind_gaussian']['iq'], G(trace['received']), rtol=0, atol=0)
        for value in outputs.values():
            logits = F.classify_features(F.features(value['iq']))
            if logits.shape != (2, 6) or not torch.isfinite(logits).all():
                raise ValueError('Original actual F0 source CPU smoke failed')
    write(out/'smoke.json', dict(status='PASS', actual_Ls=2, actual_F0=True, actual_original_BF=True,
        gaussian_exactly_matches_original_BF=True, methods=list(METHODS),
        F0_rows=8, CPU_only=True, TX_labels_read=0, target_inputs=0, source_prior=prior_report))
    print('SOURCE_WAVEFORM_MIXTURE_REAL_SMOKE_PASS', flush=True)

    ref, take = load_reference(args.root, identity, cache)
    with (out/'reference.npz').open('xb') as file:
        np.savez(file, **ref, source_index=take)
    clean, target, clean_logits, scale = [tensor(ref[k], 'cpu') for k in ('clean','features','logits','scale')]
    summaries = []
    with torch.inference_mode():
        for scene in SCENES:
            trace = read_trace(cache/'validation_actions'/scene, take, 1800)
            records = {}
            for start_row in range(0, len(take), 16):
                view = slice(start_row, start_row+16)
                outputs = readouts(G, {key: value[view] for key, value in trace.items()}, prior)
                for name, values in outputs.items():
                    iq = values['iq']
                    features = F.features(iq)
                    logits = F.classify_features(features)
                    metrics = source_metrics(iq, clean[view], features, logits, target[view], clean_logits[view], scale)
                    for key, value in dict(values, **metrics, logits=logits).items():
                        if not torch.isfinite(value).all():
                            raise ValueError('Nonfinite source mixture readout')
                        records.setdefault(name+'__'+key, []).extend(value.tolist())
            arrays = {key: np.asarray(value) for key, value in records.items()}
            with (out/(scene+'.npz')).open('xb') as file:
                np.savez(file, **arrays)
            for name in METHODS:
                agreement = arrays[name+'__clean_decision_agreement'].astype(bool)
                scalar = arrays[name.split('_')[0]+'_gaussian__clean_decision_agreement'].astype(bool)
                row = dict(scene=scene, method=name, count=len(take), Fclean_correct=int(agreement.sum()),
                    repairs=int((~scalar & agreement).sum()), damage=int((scalar & ~agreement).sum()),
                    iq_nmse=float(arrays[name+'__iq_nmse'].mean()),
                    feature_cosine_distance=float(arrays[name+'__feature_cosine_distance'].mean()))
                if name.endswith('_mixture'):
                    row['mean_max_mode_probability'] = float(arrays[name+'__probabilities'].max(axis=1).mean())
                summaries.append(row)
            print('SOURCE_WAVEFORM_MIXTURE_VIEW_COMPLETE', scene, len(take), flush=True)
    for model, before in zip(models, snapshots):
        check_F0_state(model, before)
    write(out/'results.json', dict(status='SOURCE_WAVEFORM_MIXTURE_COMPLETE', run_id=RUN, code_commit=args.commit,
        F0_identity=identity, frozen_BF_code=G_COMMIT, reference_code=REFERENCE_COMMIT,
        methods=list(METHODS), views=list(SCENES), source_prior=prior_report, records=summaries,
        source_V_records=120, source_V_views=840, F0_new_rows=3360, F0_smoke_rows=8,
        F0_clean_rows_reused=120, TX_labels_read=0, OE_reads=0, novel_reads=0, target_inputs=0,
        optimizer_updates=0, model_state_updates=0, CPU_threads=2, new_deployment_candidate=False,
        theoretical_ceiling=False, seconds=time.perf_counter()-start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('CUDA-empty source diagnostic with actual code identity required')
    torch.set_num_threads(2)
    out = args.root/'runs'/RUN
    out.mkdir()
    write(out/'launch.json', dict(pid=os.getpid(), code_commit=args.commit, CPU_threads=2, CUDA_empty=True))
    worker(args, out)
    write(out/'complete.json', dict(status='ARTIFACTS_COMPLETE', TX_labels_read=0, target_inputs=0, optimizer_updates=0))


if __name__ == '__main__':
    main()
