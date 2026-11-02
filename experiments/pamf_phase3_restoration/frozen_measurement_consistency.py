"""Read-only CPU diagnosis of saved IQ under the source cached Action proxy."""
import argparse
import json
from pathlib import Path

import numpy as np


RUN = '20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01'
PRODUCER = 'f29b31417ce98a280eac25ea2fa78e0c49a5f0ff'
R07 = '20261019-phase1-restoration-learned-action-source-s2026092701-r07'
METHODS = ['received', 'physical', 'guided', 'detail']
SCENES = ['clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban']
EPSILON = 1e-9


def apply_cached_action(iq, kernel, main, image):
    x = np.asarray(iq, np.float64)
    h, m, image = [np.asarray(v, np.complex128) for v in [kernel, main, image]]
    if (x.ndim != 3 or x.shape[1] != 2 or h.ndim != 2 or len(h) != len(x)
            or m.shape != (len(x), x.shape[2]) or image.shape != m.shape
            or not 1 <= h.shape[1] <= x.shape[2]):
        raise ValueError('Matched IQ/FIR/main/image source records required')
    n, k = x.shape[2], h.shape[1]
    z = x[:, 0]+1j*x[:, 1]
    padded = np.pad(z, ((0, 0), (k-1, 0)), mode='reflect')
    fft_length = 1 << (n+2*k-3).bit_length()
    filtered = np.fft.ifft(np.fft.fft(padded, n=fft_length)*np.fft.fft(h, n=fft_length), axis=1)[:, k-1:k-1+n]
    y = m*filtered+image*filtered.conj()
    return np.stack((y.real, y.imag), 1)


def measurement_metrics(iq, clean, received, kernel, main, image, variance):
    x, target, y = [np.asarray(v, np.float64) for v in [iq, clean, received]]
    variance = np.asarray(variance, np.float64)
    if target.shape != x.shape or y.shape != x.shape or variance.shape != (len(x),) or (variance <= 0).any():
        raise ValueError('Paired clean/received IQ and positive cached scalar variance required')
    projected = apply_cached_action(x, kernel, main, image)
    mse = ((projected-y)**2).mean((1, 2))
    return dict(iq_nmse=((x-target)**2).mean((1, 2))/np.maximum((target**2).mean((1, 2)), 1e-4),
                measurement_mse=mse, measurement_over_cached_noise_variance=mse/variance)


def load_completed_folder(root):
    folder = Path(root)/'runs'/RUN
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    if read(folder/'complete.json').get('status') != 'ARTIFACTS_COMPLETE':
        raise ValueError('Completed frozen source V required before saved IQ access')
    done = read(folder/'predictions_complete.json')
    expected = dict(status='SOURCE_V_PREDICTIONS_COMPLETE', code_commit=PRODUCER,
                    methods=METHODS, views=SCENES, source_V_packets=1800, count=12600,
                    TX_labels_read=0, target_inputs=0, optimizer_updates=0, model_state_updates=0)
    if any(done.get(key) != value for key, value in expected.items()):
        raise ValueError('Exact frozen source V producer required before saved IQ access')
    return folder


def analyze(root, code_commit):
    folder = load_completed_folder(root)
    old = Path(root)/'runs'/R07/'cache'
    with np.load(folder/'references.npz', allow_pickle=False) as f:
        ref = {key: f[key].copy() for key in ['clean', 'ids', 'rx', 'day', 'logits']}
    if ref['clean'].shape != (1800, 2, 256) or not np.array_equal(ref['ids'], np.load(old/'validation/physical_ids.npy', allow_pickle=False)):
        raise ValueError('Same original frozen source V physical order required')
    clean = ref['clean'].astype(np.float64)
    clean_decision = ref['logits'].argmax(1)
    masks = [('overall', 'ALL', np.ones(1800, bool))]
    for dimension, labels in [('RX', ref['rx']), ('day', ref['day']),
                              ('RXday', np.asarray([str(rx)+'/'+str(day) for rx, day in zip(ref['rx'], ref['day'])]))]:
        masks.extend((dimension, str(value), labels == value) for value in np.unique(labels))
    assert len(masks) == 24
    rows, conditional = [], []
    for scene in SCENES:
        trace = old/'validation_actions'/scene
        done = json.loads((trace/'complete.json').read_text())
        if (done['count'], done['TX_labels_read'], done['target_reads']) != (1800, 0, 0):
            raise ValueError('Existing completed source V Action trace required')
        h = np.load(trace/'kernel.npy', allow_pickle=False).astype(np.float64)
        c = np.load(trace/'coefficients.npy', allow_pickle=False).astype(np.float64)
        variance = np.load(trace/'variance.npy', allow_pickle=False).astype(np.float64)
        kernel, main, image = h[:, 0]+1j*h[:, 1], c[:, 0]+1j*c[:, 1], c[:, 2]+1j*c[:, 3]
        with np.load(folder/(scene+'.npz'), allow_pickle=False) as f:
            if not np.array_equal(f['ids'], ref['ids']):
                raise ValueError('Saved IQ physical order differs')
            iq = {method: f[method+'__iq'].astype(np.float64) for method in METHODS}
            decisions = {method: f[method+'__prediction'].copy() for method in METHODS}
        iq['clean_reference'] = clean
        decisions['clean_reference'] = clean_decision
        values = {method: measurement_metrics(x, clean, iq['received'], kernel, main, image, variance) for method, x in iq.items()}
        oracle = values['clean_reference']['measurement_over_cached_noise_variance']
        for method, metrics in values.items():
            metrics['measurement_excess_over_clean_reference'] = metrics['measurement_over_cached_noise_variance']-oracle
            metrics['abs_measurement_gap_from_clean_reference'] = abs(metrics['measurement_excess_over_clean_reference'])
            metrics['Fclean_decision_agreement'] = (decisions[method] == clean_decision).astype(float)
            if any(not np.isfinite(v).all() for v in metrics.values()):
                raise ValueError('Finite complete source metrics required')
            before = decisions['physical'] == clean_decision
            after = decisions[method] == clean_decision
            for dimension, stratum, mask in masks:
                rows.append(dict(scene=scene, method=method, dimension=dimension, stratum=stratum, count=int(mask.sum()),
                    metrics={key:float(v[mask].mean()) for key, v in metrics.items()},
                    measurement_ratio_median=float(np.median(metrics['measurement_over_cached_noise_variance'][mask])),
                    Fclean_repairs_vs_physical=int((~before&after&mask).sum()), Fclean_damage_vs_physical=int((before&~after&mask).sum())))
        d, p = values['detail'], values['physical']
        wave = d['iq_nmse']-p['iq_nmse']
        measurement = d['measurement_over_cached_noise_variance']-p['measurement_over_cached_noise_variance']
        gap = d['abs_measurement_gap_from_clean_reference']-p['abs_measurement_gap_from_clean_reference']
        states = lambda delta: {'smaller':delta < -EPSILON, 'tied':abs(delta) <= EPSILON, 'larger':delta > EPSILON}
        before, after = decisions['physical'] == clean_decision, decisions['detail'] == clean_decision
        events = {'Fclean_repair':~before&after, 'Fclean_damage':before&~after, 'Fclean_unchanged':before == after}
        for dimension, stratum, mask in masks:
            cells = {w:{m:{event:int((mask&wm&mm&em).sum()) for event, em in events.items()}
                        for m, mm in states(measurement).items()} for w, wm in states(wave).items()}
            gap_cells = {g:{event:int((mask&gm&em).sum()) for event, em in events.items()} for g, gm in states(gap).items()}
            assert sum(sum(sum(e.values()) for e in m.values()) for m in cells.values()) == int(mask.sum())
            assert sum(sum(e.values()) for e in gap_cells.values()) == int(mask.sum())
            conditional.append(dict(scene=scene, dimension=dimension, stratum=stratum, count=int(mask.sum()),
                                    detail_vs_physical_waveform_and_measurement=cells,
                                    detail_vs_physical_abs_gap_from_clean_measurement=gap_cells))
    assert len(rows) == 840 and len(conditional) == 168
    return dict(status='FROZEN_SOURCE_MEASUREMENT_COMPLETE', run_id=RUN, code_commit=code_commit,
        source_prediction_code_commit=PRODUCER, cached_action_source_run=R07, source_V_packets=1800,
        records=rows, conditional=conditional, epsilon=EPSILON,
        saved_IQ_records_read=52200, saved_IQ_arrays_read=29, saved_decision_records_read=52200,
        original_IQ_dataset_reads=0, cached_source_action_records_read=12600,
        new_neural_forwards=0, new_TX_label_reads=0, target_inputs=0, optimizer_updates=0,
        cached_scalar_variance_is_full_noise_covariance=False, cached_action_is_hardware_Jacobian=False,
        clean_reference_is_deployable=False, goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.commit), ensure_ascii=False, allow_nan=False))
