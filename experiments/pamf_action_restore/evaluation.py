"""Freeze every source prediction before a separate CPU process opens V labels."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .operators import Action, soft_map
from .models import SCALE
from .runtime import load_checkpoint
from experiments.pamf_fcrossrx_readout.frozen import load_reader, REFERENCE_RUN
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
from experiments.pamf_restoration.data import SCENES, write_json
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_restoration.runner import array_tensor, tensor_array
from experiments.pamf_restoration.frozen import classification_statistics

VIEWS = ('clean', *SCENES)
READOUTS = ('identity', 'coarse', 'learned_operator')


def teacher_action(arrays, index, device):
    kernel = complex_iq(array_tensor(arrays['kernel'][index], device))
    coefficients = array_tensor(arrays['coefficients'][index], device)
    return Action(kernel, torch.complex(coefficients[:, 0], coefficients[:, 1]),
                  torch.complex(coefficients[:, 2], coefficients[:, 3]))


def probe_error(estimate, target, probes):
    errors = []
    for probe in probes:
        truth = target(probe)
        errors.append((estimate(probe)-truth).abs().square().mean(-1)/truth.abs().square().mean(-1).clamp_min(1e-4))
    return torch.stack(errors).mean(0)


@torch.no_grad()
def predict(args):
    reference = args.output/'cache/validation'
    ids, _, _ = source_reference(reference, args.source_cache)
    row = args.output/'learned_operator'
    model, payload = load_checkpoint(row/'final.pt', 'cuda:0')
    if payload['epoch'] != 20 or payload['updates'] != 4940:
        raise ValueError('Fixed final action checkpoint required')
    model.requires_grad_(False)
    reader = load_reader(args.reader_folder, args.source_cache, 'cuda:0').requires_grad_(False).eval()
    folder = row/'prediction'
    folder.mkdir()
    clean = np.load(reference/'clean.npy', mmap_mode='r', allow_pickle=False)
    generator = torch.Generator(device='cuda:0').manual_seed(2026101913)
    started = time.perf_counter()
    for view in VIEWS:
        observed = np.load(reference/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        arrays = {key: np.load(args.output/'cache/validation_actions'/view/(key+'.npy'), mmap_mode='r', allow_pickle=False)
                  for key in ('kernel', 'coefficients')}
        decisions = {key: [] for key in READOUTS}
        outputs, bases, action_errors, noise_variances = [], [], [], []
        for offset in range(0, len(ids), 64):
            index = slice(offset, offset+64)
            y = array_tensor(observed[index], 'cuda:0')
            base, action, variance = model.estimate(y)
            restored = pair(soft_map(action, complex_iq(y), complex_iq(base), variance, SCALE**2, 6))
            if not torch.isfinite(restored).all():
                raise FloatingPointError('Nonfinite learned-action output')
            for name, value in (('identity', y), ('coarse', base), ('learned_operator', restored)):
                logits = reader(value)
                if logits.shape != (len(y), 6) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid original default-head predictions')
                decisions[name].extend(logits.argmax(1).tolist())
            probes = [torch.complex(*torch.randn((2, len(y), 256), device='cuda:0', generator=generator))/2**.5 for _ in range(4)]
            error = probe_error(action, teacher_action(arrays, index, 'cuda:0'), probes)
            action_errors.append(tensor_array(error)); noise_variances.append(tensor_array(variance))
            outputs.append(tensor_array(restored)); bases.append(tensor_array(base))
        with (folder/(view+'.npz')).open('xb') as stream:
            np.savez_compressed(stream, ids=ids, restored=np.concatenate(outputs), coarse_iq=np.concatenate(bases),
                probe_action_error=np.concatenate(action_errors), estimated_real_noise_variance=np.concatenate(noise_variances),
                **{key: np.asarray(value, np.uint8) for key, value in decisions.items()})
        print(json.dumps(dict(event='SOURCE_VIEW_COMPLETE', view=view, count=len(ids), TX_labels_read=0)), flush=True)
    write_json(folder/'complete.json', dict(status='SOURCE_PREDICTIONS_COMPLETE', views=list(VIEWS),
        readouts=list(READOUTS), count=len(ids), TX_labels_read=0, target_reads=0, state_updates=0,
        original_F_run=REFERENCE_RUN, head_fit_updates=0, seconds=time.perf_counter()-started))
    y = array_tensor(np.load(reference/'practical_low_urban.npy', allow_pickle=False)[:16], 'cuda:0')
    for _ in range(2):
        model(y[:1])
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(16):
        torch.cuda.synchronize(); start = time.perf_counter()
        value = model(y[i:i+1])
        torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite batch1 prediction')
    write_json(row/'batch1_profile.json', dict(samples=16, warmup_calls=2, median_ms=float(np.median(times)),
        p90_ms=float(np.quantile(times, .9)), includes_coarse=True, CG_steps=6,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), source_only=True))


def prediction_arrays(folder, ids):
    complete = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    expected = dict(status='SOURCE_PREDICTIONS_COMPLETE', views=list(VIEWS), readouts=list(READOUTS),
        count=len(ids), TX_labels_read=0, target_reads=0, state_updates=0, head_fit_updates=0)
    if any(complete.get(k) != v for k, v in expected.items()):
        raise ValueError('Complete fixed source predictions required before TX labels')
    result = {}
    for view in VIEWS:
        with np.load(folder/(view+'.npz'), allow_pickle=False) as archive:
            if not np.array_equal(archive['ids'], ids):
                raise ValueError('Source prediction physical IDs differ')
            result[view] = {key: archive[key].copy() for key in (*READOUTS, 'restored', 'coarse_iq', 'probe_action_error', 'estimated_real_noise_variance')}
        for name in READOUTS:
            array = result[view][name]
            if array.shape != (len(ids),) or array.dtype.kind not in 'iu' or (array < 0).any() or (array >= 6).any():
                raise ValueError('Invalid source default-head decisions')
        for name in ('restored', 'coarse_iq'):
            array = result[view][name]
            if array.shape != (len(ids), 2, 256) or not np.isfinite(array).all():
                raise ValueError('Invalid source IQ prediction coverage')
        for name in ('probe_action_error', 'estimated_real_noise_variance'):
            array = result[view][name]
            if (array.shape != (len(ids),) or not np.isfinite(array).all() or (array < 0).any() or
                    (name == 'estimated_real_noise_variance' and (array == 0).any())):
                raise ValueError('Invalid source action diagnostics')
    return result


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    reference, row = args.output/'cache/validation', args.output/'learned_operator'
    ids, source, index = source_reference(reference, args.source_cache)
    predictions = prediction_arrays(row/'prediction', ids)
    # This is the only TX-label open, after all seven prediction files validate.
    labels = np.load(args.source_cache/'V/labels.npy', allow_pickle=False)[index]
    clean = np.load(reference/'clean.npy', mmap_mode='r', allow_pickle=False)
    records, waveform = [], {}
    denominator = np.maximum((clean**2).mean((1, 2)), 1e-4)
    for view in VIEWS:
        data = predictions[view]
        observed = np.load(reference/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        waveform[view] = {}
        for name, value in (('identity', observed), ('coarse', data['coarse_iq']), ('learned_operator', data['restored'])):
            error = (value-clean)**2
            spectrum = np.fft.fft(value[:, 0]+1j*value[:, 1], norm='ortho')
            target = np.fft.fft(clean[:, 0]+1j*clean[:, 1], norm='ortho')
            waveform[view][name] = dict(iq_nmse=float(np.mean(error.mean((1, 2))/denominator)),
                prefix232_nmse=float(np.mean(error[:, :, :232].mean((1, 2))/np.maximum((clean[:, :, :232]**2).mean((1, 2)), 1e-4))),
                tail24_nmse=float(np.mean(error[:, :, -24:].mean((1, 2))/np.maximum((clean[:, :, -24:]**2).mean((1, 2)), 1e-4))),
                fft_log_power_L1=float(np.mean(abs(np.log(abs(spectrum)**2+1e-3)-np.log(abs(target)**2+1e-3)))))
            groups = [('overall', 'all', np.ones(len(ids), bool))]
            for dimension, values in (('TX', labels), ('RX', source.rx[index]), ('day', source.day[index])):
                groups += [(dimension, str(v), values == v) for v in np.unique(values)]
            for dimension, value_id, mask in groups:
                stat = classification_statistics(labels[mask], data[name][mask], data['identity'][mask])
                stat['iq_nmse'] = float(np.mean(error.mean((1, 2))[mask]/denominator[mask]))
                records.append(dict(view=view, readout=name, dimension=dimension, value=value_id, **stat))
        waveform[view]['probe_action_error'] = float(data['probe_action_error'].mean())
        true_variance = np.load(args.output/'cache/validation_actions'/view/'variance.npy', allow_pickle=False)
        waveform[view]['log_noise_variance_MSE'] = float(np.mean((np.log(data['estimated_real_noise_variance'])-np.log(true_variance))**2))
    means = {name: float(np.mean([waveform[view][name]['iq_nmse'] for view in SCENES])) for name in READOUTS}
    gain = 1-means['learned_operator']/means['coarse']
    feasible = gain >= .05 and waveform['clean']['learned_operator']['iq_nmse'] <= .02
    write_json(row/'summary.json', dict(status='ANALYZED', source_only=True, count=len(ids), views=waveform,
        records=records, original_F_run=REFERENCE_RUN, original_default_head=True, truth_last=True,
        mean_six_scene_NMSE=means, gain_vs_coarse=gain, waveform_feasible=bool(feasible),
        no_identity_qualification_from_waveform=True, target_reads=0, head_fit_updates=0, goal_complete=False))
