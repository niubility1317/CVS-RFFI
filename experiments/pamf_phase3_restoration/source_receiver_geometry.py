"""CPU-only descriptive geometry of already frozen source-V feature arrays."""
import argparse
import json
from pathlib import Path

import numpy as np


RUN = '20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01'
PRODUCER = 'f29b31417ce98a280eac25ea2fa78e0c49a5f0ff'
METHODS = ['received', 'physical', 'guided', 'detail']
SCENES = ['clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban']


def summarize_geometry(features, reference, labels):
    z, clean = np.asarray(features, np.float64), np.asarray(reference, np.float64)
    labels = np.asarray(labels)
    if z.ndim != 2 or clean.shape != z.shape or labels.shape != (len(z),) or not len(z):
        raise ValueError('Matched per-record features, clean reference and groups required')
    mean, target_mean = z.mean(0), clean.mean(0)
    error = z-clean
    error_mean = error.mean(0)
    between = within = error_between = error_within = clean_between = alignment = pattern_error = 0.
    groups = []
    for value in np.unique(labels):
        mask = labels == value
        a, target, e = z[mask], clean[mask], error[mask]
        mu, target_mu, e_mu = a.mean(0), target.mean(0), e.mean(0)
        weight = len(a)/len(z)
        pattern, clean_pattern = mu-mean, target_mu-target_mean
        between += weight*float(np.mean(pattern**2))
        within += weight*float(np.mean((a-mu)**2))
        error_between += weight*float(np.mean((e_mu-error_mean)**2))
        error_within += weight*float(np.mean((e-e_mu)**2))
        clean_between += weight*float(np.mean(clean_pattern**2))
        alignment += weight*float(np.mean(pattern*clean_pattern))
        pattern_error += weight*float(np.mean((pattern-clean_pattern)**2))
        groups.append(dict(group=str(value), count=len(a), paired_feature_mse=float(np.mean(e**2)),
                           paired_mean_bias_energy=float(np.mean(e_mu**2)),
                           feature_variance=float(np.mean((a-mu)**2))))
    total = float(np.mean((z-mean)**2))
    mse, bias = float(np.mean(error**2)), float(np.mean(error_mean**2))
    discrepancies = [abs(total-between-within), abs(mse-bias-error_between-error_within),
                     abs(pattern_error-error_between)]
    if max(discrepancies) > 1e-10*max(1., total, mse):
        raise ArithmeticError('Variance and paired-error decomposition disagree')
    return dict(count=len(z), dimensions=z.shape[1], total_variance=total,
                between_group_variance=between, within_group_variance=within,
                between_group_fraction=None if total == 0 else between/total,
                paired_feature_mse=mse, paired_error_global_bias=bias,
                paired_error_between_group=error_between, paired_error_within_group=error_within,
                paired_error_between_group_fraction=None if mse == 0 else error_between/mse,
                clean_between_group_variance=clean_between,
                clean_group_pattern_alignment_gain=None if clean_between == 0 else alignment/clean_between,
                clean_group_pattern_error=pattern_error, max_identity_error=max(discrepancies), groups=groups)


def load_completed_reference(root):
    folder = Path(root)/'runs'/RUN
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    if read(folder/'complete.json').get('status') != 'ARTIFACTS_COMPLETE':
        raise ValueError('Completed frozen source V required before feature access')
    done = read(folder/'predictions_complete.json')
    expected = dict(status='SOURCE_V_PREDICTIONS_COMPLETE', code_commit=PRODUCER,
                    methods=METHODS, views=SCENES, source_V_packets=1800, count=12600,
                    TX_labels_read=0, target_inputs=0, optimizer_updates=0, model_state_updates=0)
    if any(done.get(k) != value for k, value in expected.items()):
        raise ValueError('Exact frozen source V producer required before feature access')
    scored = read(folder/'results.json')
    if not scored.get('truth_last') or scored.get('code_commit') != PRODUCER:
        raise ValueError('Completed original source V scoring required')
    with np.load(folder/'references.npz', allow_pickle=False) as f:
        # NPZ keys are lazy: no IQ, logits or identity labels are loaded here.
        ref = {key: f[key].copy() for key in ['features', 'scale', 'ids', 'rx', 'day']}
    if ref['features'].shape != (1800, 160) or ref['scale'].shape != (160,) or (ref['scale'] <= 0).any():
        raise ValueError('Existing fixed source F0 features and source variance scale required')
    return folder, ref


def analyze(root, code_commit):
    folder, ref = load_completed_reference(root)
    clean = ref['features'].astype(np.float64)/np.sqrt(ref['scale'].astype(np.float64))
    labels = {'RX': ref['rx'], 'day': ref['day'],
              'RXday': np.asarray([str(rx)+'/'+str(day) for rx, day in zip(ref['rx'], ref['day'])])}
    if [len(np.unique(labels[k])) for k in ['RX', 'day', 'RXday']] != [5, 3, 15]:
        raise ValueError('Original five source RX and three days required')
    if sorted(np.unique(labels['RXday'], return_counts=True)[1].tolist()) != [120]*15:
        raise ValueError('Matched original source RX/day panel required')
    rows, group_rows = [], []
    def append(scene, method, z):
        for dimension, group in labels.items():
            q = summarize_geometry(z, clean, group)
            for part in q.pop('groups'):
                group_rows.append(dict(scene=scene, method=method, dimension=dimension, **part))
            rows.append(dict(scene=scene, method=method, dimension=dimension, **q))
    append('shared_clean_reference', 'clean_reference', clean)
    for scene in SCENES:
        with np.load(folder/(scene+'.npz'), allow_pickle=False) as f:
            if not np.array_equal(f['ids'], ref['ids']):
                raise ValueError('Saved source feature physical order differs')
            for method in METHODS:
                z = f[method+'__features'].astype(np.float64)
                if z.shape != clean.shape or not np.isfinite(z).all():
                    raise ValueError('Matched finite saved source features required')
                append(scene, method, z/np.sqrt(ref['scale'].astype(np.float64)))
    assert len(rows) == 87 and len(group_rows) == 667
    return dict(status='FROZEN_SOURCE_FEATURE_GEOMETRY_COMPLETE', run_id=RUN,
                code_commit=code_commit, source_prediction_code_commit=PRODUCER,
                G_code_commit='1fcbc3eb3cc16e7577ecc3cbedda99174b4c5cc6',
                F0_code_commit='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d',
                source_V_packets=1800, saved_feature_rows_read=52200, summary_rows=rows, group_rows=group_rows,
                max_decomposition_error=max(q['max_identity_error'] for q in rows),
                normalization='Original fixed LsUs source variance only; no new V scale fit',
                descriptive_group_means_only=True, group_centroids_saved=False,
                new_neural_forwards=0, new_IQ_array_reads=0, new_TX_label_reads=0,
                target_inputs=0, optimizer_updates=0, new_classifier_fit_updates=0,
                goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.commit), ensure_ascii=False, allow_nan=False))
