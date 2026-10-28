"""Truth-last frozen E51/readout scoring against the original E50 baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np

from . import source_embedding_scoring as embedding
from .source_open_run import METHODS


STATUS = 'SCORED_SOURCE_EMBEDDING_READOUT_COMPLETE'
ENCODER_RUN = '20261029-phase3-source-embedding-increment-s2026092701-r02'
ENCODER_COMMIT = '28d53ea9b38c19f642c59e31f178a3f65f40894d'


def _encoder_identity(marker):
    """Match the actual fixed r02 arm/checkpoint; no new artifact/hash chain."""
    identity, arm = marker.get('encoder_state_identity'), marker['arm']
    if (not isinstance(identity, dict) or identity.get('run_id') != ENCODER_RUN
            or identity.get('row_id') != arm + '-s2026092701' or identity.get('arm') != arm
            or identity.get('code_commit') != ENCODER_COMMIT
            or not isinstance(identity.get('checkpoint'), str)):
        raise ValueError('Truth closed: fixed r02 encoder row identity differs')
    parts = PurePosixPath(identity['checkpoint'].replace('\\', '/')).parts
    if parts[-5:] != (ENCODER_RUN, 'rows', arm, 'train', 'final.pt'):
        raise ValueError('Truth closed: encoder checkpoint does not belong to the fixed r02 arm')
    return identity


def _predictions(path):
    try:
        marker = embedding._read_json(Path(path).parent / 'complete.json')
    except (OSError, ValueError) as exc:
        raise ValueError('Truth closed: completed source embedding/readout predictions required') from exc
    if (not isinstance(marker, dict) or marker.get('status') != 'SOURCE_EMBEDDING_READOUT_PREDICTIONS_COMPLETE'
            or marker.get('arm') not in embedding.ARMS or marker.get('source_only') is not True
            or type(marker.get('target_reads')) is not int or marker['target_reads'] != 0
            or marker.get('unknown_truth_read') is not False or marker.get('query_fit') is not False
            or marker.get('source_V_labels_read') is not True
            or marker.get('source_known_label_calibration') is not True
            or marker.get('default_head_replaced') is not False
            or marker.get('method_names') != list(METHODS)
            or not isinstance(marker.get('parent_export_identity'), dict) or not marker['parent_export_identity']):
        raise ValueError('Truth closed: seven-method source embedding/readout contract required')
    for key, value in dict(encoder_updates=222, identity_head_updates=222,
                          new_encoder_updates=0, new_identity_head_updates=0,
                          readout_steps_per_model=200).items():
        if type(marker.get(key)) is not int or marker[key] != value:
            raise ValueError('Truth closed: historical versus new update count differs: ' + key)
    _encoder_identity(marker)
    for key in ('known_count', 'novel_count', 'count'):
        if type(marker.get(key)) is not int or marker[key] <= 0:
            raise ValueError('Truth closed: positive integer prediction role counts required')
    if marker['count'] != marker['known_count'] + marker['novel_count']:
        raise ValueError('Truth closed: completed role counts differ')
    pred = embedding._load(path, embedding.PREDICTION_KEYS)
    embedding._strings(pred['ids'], 'prediction IDs', unique=True)
    names = pred['method_names']
    if names.ndim != 1 or names.dtype.kind != 'U' or names.tolist() != list(METHODS):
        raise ValueError('Truth closed: fixed seven-method names/order required')
    n, m = len(pred['ids']), len(METHODS)
    if n != marker['count']:
        raise ValueError('Truth closed: frozen prediction count differs')
    for key in ('closed_prediction', 'baseline_closed_prediction'):
        embedding._integer(pred[key], (n,), key, allowed=np.arange(6))
    embedding._integer(pred['rx_i'], (n,), 'source receiver', allowed=embedding.SOURCE_RECEIVERS)
    embedding._integer(pred['day_i'], (n,), 'source day', allowed=embedding.SOURCE_DAYS)
    embedding._integer(pred['prediction'], (n, m), 'prediction', allowed=np.arange(-1, 6))
    values, accepted = pred['knownness'], pred['accepted']
    if values.shape != (n, m) or values.dtype.kind != 'f' or not np.isfinite(values).all():
        raise ValueError('Truth closed: finite floating knownness [N,7] required')
    if accepted.shape != (n, m) or accepted.dtype.kind != 'b':
        raise ValueError('Truth closed: boolean accepted [N,7] required')
    if not np.array_equal(pred['prediction'], np.where(accepted, pred['closed_prediction'][:, None], -1)):
        raise ValueError('Truth closed: accepted predictions must use the unchanged complete E51 head')
    return pred, marker


def score(predictions, known_features, novel_truth, output):
    """Evaluate one frozen-encoder arm; only predeclared MLP16 is promotable."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Source embedding/readout score output must be a new directory')
    pred, marker = _predictions(predictions)
    # Reuse the independent original-V-logits baseline check. Its first label IO
    # follows every prediction check; novel truth follows baseline verification.
    y, label_to_tx = embedding._truth(known_features, novel_truth, pred, marker)
    methods = {}
    for j, name in enumerate(METHODS):
        def metrics(mask):
            return embedding._group(y[mask], pred['closed_prediction'][mask],
                pred['baseline_closed_prediction'][mask], pred['knownness'][mask, j],
                pred['accepted'][mask, j], label_to_tx)
        overall = metrics(np.ones(len(y), dtype=bool))
        receivers = {str(rx): metrics(pred['rx_i'] == rx) for rx in embedding.SOURCE_RECEIVERS}
        overall_goals = embedding._checks(overall)
        receiver_goals = {rx: embedding._checks(value) for rx, value in receivers.items()}
        passed = overall_goals['passed'] and all(z['passed'] for z in receiver_goals.values())
        promotable = name == 'mlp16_oe'
        methods[name] = dict(overall=overall, by_receiver=receivers,
            by_known_class={str(row['label']): dict(row) for row in overall['known_per_class']},
            by_unknown_transmitter={row['transmitter']: dict(row) for row in overall['unknown_per_class']},
            goals=dict(overall=overall_goals, by_receiver=receiver_goals, passed=passed),
            diagnostic_passed=passed, promotable_candidate=promotable, eligible=promotable and passed)
    eligible = ['mlp16_oe'] if methods['mlp16_oe']['eligible'] else []
    selected = 'mlp16_oe' if eligible else None
    next_step = 'SOURCE_EMBEDDING_READOUT_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION' if selected else 'NO_PROMOTION'
    result = dict(schema='pamf_phase3_source_embedding_readout_scores_v1', status=STATUS,
        arm=marker['arm'], source_only=True, target_reads=0, truth_last=True, unknown_truth_read=True,
        query_fit=False, source_known_label_calibration=True, original_parent_baseline_verified=True,
        encoder_row_identity_verified=True, encoder_state_identity=marker['encoder_state_identity'],
        parent_export_identity=marker['parent_export_identity'], encoder_updates=222, identity_head_updates=222,
        new_encoder_updates=0, new_identity_head_updates=0, readout_steps_per_model=200,
        update_scope='222 historical r02 encoder/head updates; this run encoder/head updates0, each new rejection readout200',
        default_head_replaced=False, independent_confirmation=False, source_novel_evaluation_reused=True,
        evidence_scope='source_only_not_target', known_count=marker['known_count'], novel_count=marker['novel_count'],
        count=len(y), novel_transmitter_count=len(label_to_tx),
        novel_label_to_transmitter={str(k): v for k, v in label_to_tx.items()},
        source_receivers=list(embedding.SOURCE_RECEIVERS), source_days=list(embedding.SOURCE_DAYS),
        method_names=list(METHODS), methods=methods, eligible_methods=eligible, selected_method=selected,
        next_step=next_step, final_phase3_complete=False, domain_adaptation_performed=False,
        anonymous_registration_performed=False,
        metric_definitions=dict(known_macro='Equal mean over six old known labels; rejected predictions count as incorrect',
            unknown_macro='Equal mean over private true TX; every accepted unknown is false acceptance',
            baseline='Original E50 F6 default logits argmax independently checked on every known ID',
            new_closed='Frozen r02 E51 complete default head, shared by all seven methods',
            encoder_closed_loss='Original baseline macro accuracy minus E51 closed macro accuracy',
            total_open_loss='Original baseline macro accuracy minus open macro accuracy',
            rejection_loss='E51 closed macro accuracy minus open macro accuracy',
            protection='Overall and each source RX: encoder loss<=.005, totalopen loss<=.02, FRR<=.05, unknown rejection>=.95',
            selection='Only MLP16 per fixed encoder arm; linear/static diagnostic. Controller must finish both rows, then prefer supervised_only if both MLPs pass',
            scope='Repeated source novel development with empirical known-V risk calibration; not independent target confirmation'))
    output.mkdir(parents=True, exist_ok=False)
    embedding._write(output / 'results.json', result)
    embedding._write(output / 'complete.json', dict(status=STATUS, source_only=True, target_reads=0,
        truth_last=True, arm=marker['arm'], encoder_updates=222, identity_head_updates=222,
        new_encoder_updates=0, new_identity_head_updates=0, readout_steps_per_model=200,
        original_parent_baseline_verified=True, encoder_row_identity_verified=True,
        evidence_scope=result['evidence_scope'], eligible_methods=eligible, selected_method=selected,
        next_step=next_step, final_phase3_complete=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions', 'known-features', 'novel-truth', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    result = score(args.predictions, args.known_features, args.novel_truth, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'arm', 'eligible_methods', 'selected_method', 'next_step')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
