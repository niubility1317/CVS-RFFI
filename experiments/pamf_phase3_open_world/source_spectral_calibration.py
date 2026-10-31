"""Source-only temporal falsification of label-free receiver spectral calibration.

All source models have already seen these RXs and physical source-L records.
This module neither opens target data nor establishes unseen-RX performance.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch
from . import star_dura_best_run as best
from . import star_representation_cache as cache
from . import identity_source_diagnostic as diagnostic

RUN_ID = '20261031-phase3-source-spectral-calibration-s2026092701-r01'
STATUS = 'SOURCE_SPECTRAL_CALIBRATION_COMPLETE'
SCHEMA = 'pamf_phase3_source_spectral_calibration_v1'
RX = (1, 3, 4, 6, 8)
METHODS = ('IDENTITY', 'RECEIVER_PSD')
FIELDS = {'run_id', 'code_commit', 'backend_config', 'IQ_root', 'output_root', 'batch_size'}
SMOOTH_WIDTH = 17
GAIN_MIN, GAIN_MAX = .5, 2.
_read, _write = best._read, best._write


def validate_config(c):
    if (not isinstance(c, dict) or set(c) != FIELDS or c.get('run_id') != RUN_ID
            or re.fullmatch('[0-9a-f]{40}', str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size'] != 128):
        raise ValueError('Exact source-only configuration; no target/query/truth inputs')
    b = c['backend_config']
    if (not isinstance(b, dict) or set(b) != {'train_config', 'dura_checkpoint', 'dura_release'}
            or b['train_config'].get('run_id') != '20261030-phase3-star-signal-open-s2026092701-r03'
            or b['train_config'].get('code_commit') != 'dc01a402582f93e452dbaa3ef59a9fbb44151818'
            or b['train_config'].get('arm') != 'E_SIGNAL'):
        raise ValueError('Original pure-source r03/F0/DURA backend required')
    if c['IQ_root'] != b['train_config'].get('IQ_root'):
        raise ValueError('Original source-role cache required')
    out = Path(c['output_root']).resolve()
    inputs = [c['IQ_root'], b['train_config']['output_root'], b['train_config']['native_config']['source_output'], b['dura_checkpoint']]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in inputs):
        raise ValueError('Immutable output must be separate from existing inputs')
    return c


def _IQ(x):
    x = np.asarray(x)
    if x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (2, 256) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty FP32 [N,2,256] cached DURA P1 required')
    return x


def mean_spectrum(x):
    """Unit-mean per-record periodograms; no TX labels or second input RMS."""
    x = _IQ(x)
    f = np.fft.fft(x[:, 0].astype(np.float64) + 1j * x[:, 1].astype(np.float64), axis=1)
    power = np.abs(f) ** 2
    mass = power.mean(1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError('Zero-energy calibration records cannot identify a spectrum')
    p = (power / mass).mean(0)
    p = sum(np.roll(p, k) for k in range(-8, 9)) / SMOOTH_WIDTH
    p = np.maximum(p, 1e-8)
    return p / p.mean()


def fit(x, rx, day):
    """Only each RX's early day and other RXs' early spectra enter state."""
    x = _IQ(x)
    rx, day = np.asarray(rx), np.asarray(day)
    if (rx.shape != (len(x),) or day.shape != rx.shape or rx.dtype != np.int64 or day.dtype != np.int64
            or not np.isin(rx, RX).all() or not np.isin(day, (1, 2, 3)).all()):
        raise ValueError('Existing source RX/day metadata only')
    spectra = []
    for r in RX:
        mask = (rx == r) & (day == 1)
        if not mask.any():
            raise ValueError('Every source RX needs early calibration records')
        spectra.append(mean_spectrum(x[mask]))
    spectra = np.asarray(spectra, np.float64)
    references = []
    for i in range(len(RX)):
        ref = np.exp(np.log(spectra[np.arange(len(RX)) != i]).mean(0))
        references.append(ref / ref.mean())
    references = np.asarray(references, np.float64)
    gains = np.clip(np.sqrt(references / spectra), GAIN_MIN, GAIN_MAX)
    return dict(rx=np.asarray(RX, np.int64), early_spectra=spectra, references=references, gains=gains)


def apply(x, gain):
    """Fixed circular zero-phase amplitude filter; independently per record."""
    x = _IQ(x)
    gain = np.asarray(gain)
    if gain.dtype != np.float64 or gain.shape != (256,) or not np.isfinite(gain).all() or np.any((gain < GAIN_MIN) | (gain > GAIN_MAX)):
        raise ValueError('Frozen bounded FP64 spectral gains required')
    f = np.fft.fft(x[:, 0].astype(np.float64) + 1j * x[:, 1].astype(np.float64), axis=1)
    z = np.fft.ifft(f * gain[None], axis=1)
    result = np.stack((z.real, z.imag), axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite calibrated input')
    return result


def _encode(b, x, device):
    x = _IQ(x)
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        tensor = torch.tensor(x.tolist(), dtype=torch.float32, device=device)
        raw = b['reader'].features(tensor)
        logits = b['reader'].classify_features(raw)
        f = torch.nn.functional.normalize(raw, dim=1)
        z = b['model'](tensor)
    a = {k: np.asarray(v.cpu().tolist(), np.float32) for k, v in dict(logits=logits, F0_unit=f, r03_unit=z).items()}
    if a['logits'].shape != (len(x), 6) or any(a[k].shape != (len(x), 160) for k in ('F0_unit', 'r03_unit')) or not all(np.isfinite(v).all() for v in a.values()):
        raise ValueError('Actual original default logits and two finite unit embeddings required')
    return a


def geometry(x, y, rx):
    """Label-side description only, after every source prediction is fixed."""
    x = diagnostic._unit(x)
    cells = np.asarray([diagnostic._unit(x[(y == c) & (rx == r)].mean(0)) for c in range(6) for r in RX])
    distance = np.clip(1 - cells @ cells.T, 0, 2)
    labels = np.repeat(np.arange(6), len(RX))
    same = (labels[:, None] == labels[None]) & np.triu(np.ones(distance.shape, bool), 1)
    centers = np.asarray([diagnostic._unit(x[y == c].mean(0)) for c in range(6)])
    between = np.clip(1 - centers @ centers.T, 0, 2)[np.triu_indices(6, 1)]
    return dict(same_TX_cross_RX_centroid_distance=float(distance[same].mean()), between_TX_centroid_distance=float(between.mean()))


def summarize(rows, meta):
    y, rx, day = (meta[k] for k in ('y', 'rx_i', 'day_i'))
    later = day != 1
    result = {}
    for j, name in enumerate(METHODS):
        logits = rows['logits'][:, j]
        result[name] = dict(early=diagnostic._classification(logits[~later], y[~later]), later=diagnostic._classification(logits[later], y[later]),
            by_RX={str(r): diagnostic._classification(logits[later & (rx == r)], y[later & (rx == r)]) for r in RX},
            by_day={str(d): diagnostic._classification(logits[day == d], y[day == d]) for d in (1, 2, 3)},
            geometry={k: geometry(rows[k][later, j], y[later], rx[later]) for k in ('F0_unit', 'r03_unit')})
    before, after = (result[k] for k in METHODS)
    protected = all(before['by_RX'][str(r)]['macro_accuracy'] - after['by_RX'][str(r)]['macro_accuracy'] <= .02 + 1e-12 for r in RX)
    a, b = before['geometry']['r03_unit'], after['geometry']['r03_unit']
    reduced = b['same_TX_cross_RX_centroid_distance'] <= .9 * a['same_TX_cross_RX_centroid_distance']
    retained = b['between_TX_centroid_distance'] >= .95 * a['between_TX_centroid_distance']
    old, new = rows['logits'][:, 0].argmax(1), rows['logits'][:, 1].argmax(1)
    return dict(methods=result, paired_later=dict(changed=int((later & (old != new)).sum()), correct_gained=int((later & (old != y) & (new == y)).sum()),
        correct_lost=int((later & (old == y) & (new != y)).sum())), source_identity_protected=bool(protected), source_RX_distance_reduced_10pct=bool(reduced),
        source_between_TX_retained_95pct=bool(retained), source_screen_passed=bool(protected and reduced and retained),
        scientific_scope='Previously trained source L temporal diagnostic; not independent unseen-RX/unknown validation', A_K=None, C_K=None, R_U=None, H_score=None,
        unknown_or_registration_conclusion=None, target_access_authorized_by_this_result=False, full_goal_status='ACTIVE')


def run(c, device='cuda:0'):
    validate_config(c)
    out = Path(c['output_root'])
    if out.exists():
        raise FileExistsError('New immutable source diagnostic root required')
    b = best._backend(c['backend_config'], device)
    models = [b['reader'], b['model'], b['dura'][0]]
    for model in models:
        model.eval().requires_grad_(False)
    snapshots = [diagnostic._snapshot(m) for m in models]
    def check():
        for m, s in zip(models, snapshots):
            diagnostic._check(m, s)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    smoke = best._smoke(b, device)
    check()
    if smoke.get('status') != 'PASS':
        raise ValueError('Actual checkpoint no-query smoke failed')
    out.mkdir(parents=True)
    _write(out / 'resolved_config.json', c)
    _write(out / 'smoke.json', smoke)
    a = cache.load_role(Path(c['IQ_root']) / 'L', 'L')
    try:
        m = a['manifest']
        if m['count'] != 6300 or m['parent_F0_identity'] != b['reader_identity'] or m['dura_identity'] != b['dura_identity'] or not m['source_only']:
            raise ValueError('Matching unchanged source L/F0/DURA handles required')
        meta = a['metadata']
        state = fit(a['restored'], meta['rx_i'], meta['day_i'])
        best._save(out / 'state.npz', state)
        all_rows = []
        for start in range(0, 6300, 128):
            end = min(start + 128, 6300)
            original = np.asarray(a['restored'][start:end])
            changed = np.empty_like(original)
            for i, r in enumerate(RX):
                mask = meta['rx_i'][start:end] == r
                if mask.any():
                    changed[mask] = apply(original[mask], state['gains'][i])
            arms = [_encode(b, values, device) for values in (original, changed)]
            all_rows.append({k: np.stack([v[k] for v in arms], axis=1) for k in arms[0]})
            check()
            print('SOURCE_SPECTRAL_PROGRESS', end, 6300, flush=True)
        rows = {k: np.concatenate([v[k] for v in all_rows]) for k in all_rows[0]}
        best._save(out / 'predictions.npz', dict(ids=a['ids'].copy(), **rows))
        _write(out / 'predictions_complete.json', dict(status='PREDICTIONS_FROZEN', method_names=METHODS, count=6300, label_evaluation_started=False))
        summary = summarize(rows, meta)
        summary.update(run_id=RUN_ID, code_commit=c['code_commit'], method_names=METHODS, gain_min=float(state['gains'].min()), gain_max=float(state['gains'].max()),
            original_F0_identity=b['reader_identity'], original_r03_identity=b['model_identity'], dura_identity=b['dura_identity'], source_manifest=m)
        check()
        _write(out / 'summary.json', summary)
        done = dict(schema=SCHEMA, status=STATUS, run_id=RUN_ID, code_commit=c['code_commit'], config=c, source_screen_passed=summary['source_screen_passed'],
            source_L_reads=6300, calibration_records=2100, later_diagnostic_records=4200, receiver_states=5, reference_states=5, calibration_TX_labels_used=False,
            target_U_reads=0, target_Q_reads=0, source_V_reads=0, source_NV_reads=0, source_OE_reads=0, original_U_s_reads=0, private_truth_reads=0,
            real_F_forward_rows=12600, real_signal_forward_rows=12600, real_G_forward_rows=0, synthetic_F_forward_rows=2, synthetic_signal_forward_rows=2, synthetic_G_forward_rows=2,
            optimizer_updates=0, new_neural_updates=0, threshold_updates=0, original_models_unchanged=True, all_predictions_before_evaluation=True,
            source_only_inputs=True, source_only_model_ancestors=True, input_path='cached_DURA_P1_then_frozen_receiver_PSD',
            filter_kind='256-point circular complex FFT, real positive gain, no output RMS or learned phase',
            scientific_verdict='SOURCE_SCREEN_PASS_REQUIRES_TARGET_CONFIRMATION' if summary['source_screen_passed'] else 'SOURCE_SCREEN_FAILED', full_goal_status='ACTIVE')
        _write(out / 'complete.json', done)
        print(STATUS, flush=True)
        return done
    finally:
        cache.close_role(a)


def main():
    print(json.dumps(dict(event='SOURCE_SPECTRAL_PROCESS', pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv)), flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run(_read(args.config), args.device)


if __name__ == '__main__':
    main()
