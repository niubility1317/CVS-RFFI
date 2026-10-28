"""Recalibrate the parent's frozen source readouts using known-source risk only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import source_open_run as parent
from .source_risk_calibration import calibrate_known, METHODS


PARENT_RUN = '20261029-phase3-source-open-readout-s2026092701-r01'


def _known_labels(export_root, known):
    with np.load(Path(export_root) / parent.FEATURE_REFS['source_V'], allow_pickle=False) as stored:
        ids, labels = stored['ids'].copy(), stored['y'].copy()
    if (not np.array_equal(ids, known['ids']) or labels.shape != (len(ids),)
            or labels.dtype != np.int64 or set(labels.tolist()) != set(range(6))):
        raise ValueError('Known source V labels must align exactly with its fixed feature IDs')
    return labels


def predict(parent_run_root, output):
    output = parent._new(output)
    root = Path(parent_run_root)
    if root.name != PARENT_RUN:
        raise ValueError('Only the fixed analyzed full-six-class source-readout parent may be reused')
    export_root, fit_root = root / 'export', root / 'fit'
    marker = parent._export_marker(export_root)
    identity = parent._identity(marker)
    parent._frozen_fit(fit_root, identity)
    geometry, readouts = parent._load_aggregates(fit_root)
    output.mkdir(parents=True, exist_ok=False)
    labeled = parent._load_features(export_root, 'source_L', False)
    smoke = parent._score({key: value[:2] for key, value in labeled.items()}, geometry, readouts)
    if smoke.shape != (2, len(METHODS)) or not np.isfinite(smoke).all():
        raise ValueError('The actual frozen parent readouts must pass two-source-L feature smoke')
    parent._write(output / 'smoke.json', dict(status='PASS', actual_frozen_readouts=True,
        source_L_smoke_samples=2, source_L_feature_rows_loaded=len(labeled['ids']),
        source_L_labels_read=False, source_V_reads=0, novel_validation_reads=0,
        encoder_updates=0, identity_head_updates=0, readout_updates=0, target_reads=0,
        method_names=list(METHODS), knownness_shape=list(smoke.shape)))
    del labeled
    print('SOURCE_RISK_PARENT_SMOKE_PASS', 2, len(METHODS), flush=True)
    known = parent._load_features(export_root, 'source_V', False)
    knownness = parent._score(known, geometry, readouts)
    closed_known = known['logits'].argmax(axis=1).astype(np.int64)
    labels = _known_labels(export_root, known)
    risk = calibrate_known(known['ids'], labels, closed_known, knownness, known['rx_i'])
    thresholds = risk['thresholds']
    calibration = output / 'calibration'; calibration.mkdir()
    with (calibration / 'thresholds.npy').open('xb') as stream:
        np.save(stream, thresholds, allow_pickle=False)
    summary = {key: value for key, value in risk.items() if key != 'thresholds'}
    summary.update(status='SOURCE_KNOWN_RISK_THRESHOLDS_FROZEN', source_only=True, target_reads=0,
        source_V_labels_read=True, source_known_label_calibration=True,
        unknown_calibration_scores_read=0, unknown_calibration_labels_read=0, novel_validation_reads=0,
        encoder_updates=0, identity_head_updates=0, readout_updates=0, thresholds_ref='thresholds.npy',
        independent_confirmation=False, source_novel_evaluation_reused=True)
    parent._write(calibration / 'complete.json', summary)
    print('SOURCE_KNOWN_RISK_THRESHOLDS_FROZEN', len(known['ids']), .04, .015, flush=True)
    # No novel validation feature or score has been opened before this marker.
    novel = parent._load_features(export_root, 'novel_validation', False)
    if set(known['ids'].tolist()) & set(novel['ids'].tolist()):
        raise ValueError('Known-source and novel-source validation IDs overlap')
    novelness = parent._score(novel, geometry, readouts)
    ids = np.r_[known['ids'], novel['ids']]
    values = np.concatenate((knownness, novelness))
    closed = np.r_[closed_known, novel['logits'].argmax(axis=1)].astype(np.int64)
    accepted = values >= thresholds
    order = np.argsort(ids, kind='stable')
    parent._npz(output / 'predictions.npz', dict(ids=ids[order], method_names=np.asarray(METHODS),
        closed_prediction=closed[order], knownness=values[order], accepted=accepted[order],
        prediction=np.where(accepted, closed[:, None], -1).astype(np.int64)[order],
        rx_i=np.r_[known['rx_i'], novel['rx_i']][order], day_i=np.r_[known['day_i'], novel['day_i']][order]))
    result = dict(status='ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', source_only=True, target_reads=0,
        truth_read=False, truth_read_scope='unknown/private truth only; known source labels are used for risk calibration',
        query_fit=False, source_V_labels_read=True, source_known_label_calibration=True,
        unknown_calibration_scores_read=0, unknown_calibration_labels_read=0,
        known_count=len(known['ids']), novel_count=len(novel['ids']), count=len(ids), method_names=list(METHODS),
        encoder_updates=0, identity_head_updates=0, readout_updates=0, OE_truth_reads=0,
        limits=risk['limits'], constraints_passed=risk['constraints_passed'],
        calibration_ref='calibration/complete.json', thresholds_ref='calibration/thresholds.npy', predictions_ref='predictions.npz',
        export_identity=identity, parent_run_root=str(root), export_root=str(export_root), fit_root=str(fit_root),
        independent_confirmation=False, source_novel_evaluation_reused=True,
        risk_scope='empirical risk on the same known-source calibration samples',
        evidence_scope='repeated source development evaluation; not independent confirmation or formal target evidence')
    parent._write(output / 'complete.json', result)
    print('ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', len(ids), len(METHODS), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    command = commands.add_parser('predict')
    command.add_argument('--parent-run-root', required=True); command.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = predict(args.parent_run_root, args.output)
    print(json.dumps(dict(status=result['status'], count=result['count'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
