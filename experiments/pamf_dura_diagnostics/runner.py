"""Diagnose the frozen DURA source inverse problem, without an F or truth reader.

Oracle substitutions here consume simulated SOURCE action parameters. They are
counterfactual diagnostics, not deployable models or rigorous accuracy bounds.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map

RUN = '20261030-diagnostic-DURA-source-inverse-s2026092701-r02'
PARENT = '20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69'
ACTION = '20261019-phase1-restoration-learned-action-source-s2026092701-r07'


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_indices(roles):
    if roles.shape != (63000,) or (roles == 'L_s').sum() != 6300 or (roles == 'U_s').sum() != 56700:
        raise ValueError('Original L6300/U56700 source roles required')
    # Fixed position sampling; no IQ, TX labels or evaluation scores select data.
    return np.concatenate([np.flatnonzero(roles == role)[np.linspace(0, count-1, n, dtype=int)]
                           for role, count, n in [('L_s', 6300, 128), ('U_s', 56700, 384)]])


def tensor(value, device):
    # The frozen N607 runtime is Torch2.1 + NumPy2; its C bridge is unavailable.
    # Match the existing R46 direct-buffer route, for IQ and action arrays.
    contiguous = np.array(value, dtype=np.float32, copy=True, order='C')
    return torch.frombuffer(memoryview(contiguous), dtype=torch.float32).reshape(contiguous.shape).to(device)


def numpy_values(value):
    return np.asarray(value.detach().cpu().tolist(), dtype=np.float32)


def action_from_arrays(kernel, coefficients):
    return Action(complex_iq(kernel), torch.complex(coefficients[:, 0], coefficients[:, 1]),
                  torch.complex(coefficients[:, 2], coefficients[:, 3]))


def waveform_metrics(value, clean):
    if value.shape != clean.shape or value.shape[1:] != (2, 256):
        raise ValueError('Complete aligned source IQ required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-4)
    difference = value-clean
    metrics = {'iq_nmse': difference.square().mean((1, 2))/energy,
               'prefix232_nmse': difference[..., :232].square().mean((1, 2))/energy,
               'tail24_nmse': difference[..., 232:].square().mean((1, 2))/energy}
    reference = torch.fft.fft(complex_iq(clean), norm='ortho')
    error = (torch.fft.fft(complex_iq(value), norm='ortho')-reference).abs().square()
    power = reference.abs().square()
    order = power.argsort(-1)
    for index in range(4):
        selected = order[:, index*64:(index+1)*64]
        # Each denominator has a floor tied to full-window power; weak bins do
        # not acquire infinite relative weight. This is scoring, not a loss.
        denominator = power.gather(1, selected).mean(1).clamp_min(1e-4)
        metrics[f'fft_q{index+1}_relative_error'] = error.gather(1, selected).mean(1)/denominator
    if any(not torch.isfinite(v).all() for v in metrics.values()):
        raise FloatingPointError('Nonfinite source diagnostic')
    return metrics


def last_prior(model, received, context, previous):
    base, action, native = context
    observed = complex_iq(received)
    innovation = pair(action.adjoint(observed-action(complex_iq(previous))))
    innovation = innovation/(innovation.square().mean((1, 2), keepdim=True)+1e-6).sqrt()
    logvar = native.log().clamp(-16, 4)[:, None, None].expand(-1, 1, 256)/8
    return base+SCALE*model.prior[-1](torch.cat((received, base, previous, innovation, logvar), 1))


@torch.no_grad()
def diagnose_batch(model, clean, received, actual_action, actual_variance, probe_generator):
    context = model.estimate_context(received)
    stages = model.stage_outputs(received, context)[0]
    base, action, variance = context
    prior = last_prior(model, received, context, stages[0])
    observed, start = complex_iq(received), complex_iq(stages[0])
    restored = pair(weighted_map(action, observed, complex_iq(prior), variance, steps=2, initial=start))
    if not torch.equal(restored, stages[-1]):
        raise ValueError('Native last-stage reconstruction must be exact')
    candidates = dict(received=received, coarse=base, stage1=stages[0], native=stages[-1], prior=prior)
    candidates['learned_CG16'] = pair(weighted_map(action, observed, complex_iq(prior), variance, steps=16, initial=start))
    candidates['oracle_variance_CG16'] = pair(weighted_map(action, observed, complex_iq(prior), actual_variance, steps=16, initial=start))
    candidates['oracle_action_CG16'] = pair(weighted_map(actual_action, observed, complex_iq(prior), variance, steps=16, initial=start))
    candidates['oracle_both_CG16'] = pair(weighted_map(actual_action, observed, complex_iq(prior), actual_variance, steps=16, initial=start))
    # These two deliberately use clean as a prior to quantify consistency bias.
    # Their outputs cannot leave the diagnostic or be called legal predictions.
    for name, operator, noise in [('clean_prior_learned', action, variance),
                                  ('clean_prior_oracle', actual_action, actual_variance)]:
        candidates[name] = pair(weighted_map(operator, observed, complex_iq(clean), noise,
                                            steps=16, initial=complex_iq(clean)))
    metrics = {f'{name}/{key}': value for name, estimate in candidates.items()
               for key, value in waveform_metrics(estimate, clean).items()}
    x = complex_iq(clean)
    actual_noise = observed-actual_action(x)
    learned_noise = observed-action(x)
    metrics['operator/clean_replay_nmse'] = (action(x)-actual_action(x)).abs().square().mean(1)/actual_action(x).abs().square().mean(1).clamp_min(1e-4)
    metrics['operator/actual_residual_power'] = actual_noise.abs().square().mean(1)/2
    metrics['operator/learned_residual_power'] = learned_noise.abs().square().mean(1)/2
    metrics['operator/estimated_variance'] = variance
    metrics['operator/actual_variance'] = actual_variance
    metrics['operator/log_variance_ratio'] = (variance/actual_variance).log()
    for probe in range(4):
        values = torch.randn(clean.shape, generator=probe_generator, device=clean.device, dtype=clean.dtype)
        z = complex_iq(values)
        response = actual_action(z)
        metrics[f'operator/probe{probe}_nmse'] = (action(z)-response).abs().square().mean(1)/response.abs().square().mean(1).clamp_min(1e-4)
    return {key: numpy_values(value) for key, value in metrics.items()}


def summarize(values, selected):
    return {key: dict(mean=float(v[selected].mean()), median=float(np.median(v[selected])),
                      p90=float(np.quantile(v[selected], .9)), p95=float(np.quantile(v[selected], .95)))
            for key, v in values.items()}


def run(args):
    from experiments.pamf_r65_four_route.runtime import load_checkpoint
    if args.output.name != RUN or args.output.exists():
        raise ValueError('New immutable diagnostic output root required')
    if args.checkpoint.parts[-4:] != (PARENT, 'stage2', 'native_map_prior', 'final.pt'):
        raise ValueError('Only the frozen DURA v1 checkpoint is allowed')
    if args.action_run.name != ACTION:
        raise ValueError('Only original source action cache is allowed')
    cache = args.action_run/'cache'
    with (cache/'complete.json').open(encoding='utf-8') as stream:
        cache_record = json.load(stream)
    if (cache_record.get('status') != 'ACTION_SOURCE_CACHE_COMPLETE'
            or cache_record.get('train_count') != 63000
            or cache_record.get('epochs') != 20
            or cache_record.get('TX_labels_read') != 0 or cache_record.get('target_reads') != 0):
        raise ValueError('Matching source-only action cache completion required')
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    indices = source_indices(roles)
    clean = np.load(cache/'train/clean.npy', mmap_mode='r', allow_pickle=False)
    if clean.shape != (63000, 2, 256) or clean.dtype != np.float32:
        raise ValueError('Original FP32 source IQ required')
    model, payload = load_checkpoint(args.checkpoint, args.device)
    if (payload['arm'], payload['epoch'], payload['updates']) != ('stage2', 20, 4940):
        raise ValueError('Frozen source-final DURA checkpoint required')
    model.eval().requires_grad_(False)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    args.output.mkdir(parents=True)
    write_json(args.output/'launch.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
               commit=args.commit, physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
               owner='codex/root/dura-source-inverse', source_only=True))
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(2026103001)
    # First step: actual checkpoint + two real L_s inputs, no query/F/truth.
    with torch.inference_mode():
        smoke = model(tensor(clean[indices[:2]], device))
        if smoke.shape != (2, 2, 256) or not torch.isfinite(smoke).all():
            raise ValueError('Actual source checkpoint smoke failed')
    print(json.dumps(dict(event='SOURCE_SMOKE_PASS', real_Ls=2, F_reads=0, target_reads=0)), flush=True)
    all_summary = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for epoch in (1, 20):
            folder = cache/'epochs'/f'{epoch:02d}'
            names = ('received', 'kernel', 'coefficients', 'variance')
            arrays = {key: np.load(folder/(key+'.npy'), mmap_mode='r', allow_pickle=False) for key in names}
            expected = dict(received=(63000, 2, 256), kernel=(63000, 2, 64),
                            coefficients=(63000, 4, 256), variance=(63000,))
            if any(value.shape != expected[key] or value.dtype != np.float32 for key, value in arrays.items()):
                raise ValueError('Original full matching source action arrays required')
            with np.load(folder/'groups.npz', allow_pickle=False) as groups:
                labels = {key: groups[key][indices] for key in ('scene', 'state', 'lock')}
            accumulated = {}
            for offset in range(0, len(indices), 32):
                take = indices[offset:offset+32]
                values = {key: tensor(value[take], device) for key, value in arrays.items()}
                operator = action_from_arrays(values['kernel'], values['coefficients'])
                batch = diagnose_batch(model, tensor(clean[take], device), values['received'],
                                       operator, values['variance'], generator)
                for key, value in batch.items():
                    accumulated.setdefault(key, []).append(value)
                print(json.dumps(dict(event='SOURCE_DIAGNOSTIC_BATCH', epoch=epoch,
                                      count=min(offset+32, len(indices)), seconds=time.perf_counter()-started)), flush=True)
            values = {key: np.concatenate(value) for key, value in accumulated.items()}
            if any(v.shape != (512,) or not np.isfinite(v).all() for v in values.values()):
                raise FloatingPointError('Incomplete or nonfinite source diagnostics')
            np.savez(args.output/f'epoch{epoch:02d}.npz', source_positions=indices, roles=roles[indices],
                     **labels, **values)
            groups = {'all': np.ones(512, dtype=bool)}
            for key, group_values in dict(role=roles[indices], **labels).items():
                for label in np.unique(group_values):
                    groups[f'{key}:{label}'] = group_values == label
            all_summary[str(epoch)] = {key: dict(count=int(mask.sum()), metrics=summarize(values, mask))
                                       for key, mask in groups.items()}
    if any(not torch.equal(v.detach().cpu(), before[k]) for k, v in model.state_dict().items()):
        raise ValueError('Frozen DURA state changed')
    write_json(args.output/'results.json', dict(status='ANALYZED_SOURCE_ONLY', summary=all_summary,
               unique_source_records=512, L_s=128, U_s=384, epochs=[1, 20], source_smoke_Ls=2,
               training_updates=0, TX_labels_read=0, F_reads=0, V_reads=0, target_reads=0,
               model_unchanged=True, seconds=time.perf_counter()-started,
               claim='Training-pool counterfactual diagnostics only; oracle substitutions are not deployable, accuracy bounds, or identity evidence'))
    write_json(args.output/'complete.json', dict(status='ANALYZED_SOURCE_ONLY', goal_complete=False,
               prediction_artifacts=0, model_updates=0, TX_labels_read=0, F_reads=0, target_reads=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'action-run', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args)


if __name__ == '__main__':
    main()
