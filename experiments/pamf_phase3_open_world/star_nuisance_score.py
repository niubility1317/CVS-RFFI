"""Independent truth-last source nuisance evaluation; no fitting or feedback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import star_nuisance_source as source
from .metrics import open_set_metrics

SCHEMA = 'pamf_phase3_star_nuisance_scores_v1'
STATUS = 'STAR_NUISANCE_SCORES_COMPLETE'
SCENES, RECEIVERS = source.SCENES, source.RECEIVERS
UNKNOWN_TX = ('19-15', '19-16', '7-1')  # Scorer-only required source-heldout roster.


def _ids(value, n):
    return (value.shape == (n,) and value.dtype.kind == 'U' and len(set(value.tolist())) == n
            and not np.any(np.char.strip(value) == ''))


def _prediction(root):
    root = Path(root); marker = source._read(root/'complete.json')
    if (marker.get('status') != source.STATUS or marker.get('schema') != source.SCHEMA or marker.get('run_id') != source.RUN_ID
            or marker.get('candidate_names') != list(source.CANDIDATES) or marker.get('path_names') != list(source.PATHS)
            or marker.get('variant_names') != list(source.VARIANTS) or marker.get('method_names') != list(source.METHODS)
            or marker.get('primary_candidate') != source.PRIMARY or marker.get('source_only') is not True
            or any(marker.get(k) != 0 for k in ('target_reads', 'source_novel_truth_reads', 'source_U_cache_reads', 'new_neural_updates'))
            or any(marker.get(k) is not True for k in ('model_parameters_buffers_unchanged', 'L_aggregates_unchanged_during_V',
                                                      'all_thresholds_frozen_before_novel'))
            or marker.get('baseline') != 'same-input same-path original F0 complete head'):
        raise ValueError('All twelve source-only frozen predictions must complete before private truth is read')
    calibration = source._read(root/'source/calibration/complete.json')
    if (calibration.get('status') != source.CALIBRATION_STATUS or calibration.get('schema') != source.SCHEMA
            or calibration.get('run_id') != source.RUN_ID or calibration.get('candidate_names') != list(source.CANDIDATES)
            or calibration.get('group_count') != 42 or calibration.get('source_only') is not True
            or calibration.get('target_reads') != 0 or calibration.get('unknown_inputs_used') is not False
            or calibration.get('source_V_metric_fit') is not False or calibration.get('source_V_sigma_fit') is not False
            or set(calibration.get('methods', {})) != set(source.CANDIDATES)):
        raise ValueError('Matching known-only frozen global threshold report required')
    thresholds = np.load(root/'source/calibration/thresholds.npy', allow_pickle=False)
    if thresholds.shape != (12,) or thresholds.dtype != np.float64 or not np.isfinite(thresholds).all():
        raise ValueError('Twelve finite source-frozen thresholds required')
    for j, name in enumerate(source.CANDIDATES):
        if calibration['methods'][name].get('threshold') != float(thresholds[j]):
            raise ValueError('Frozen threshold array and source report disagree')
    rows = []
    for role, fields, count_key in (('known', source.KNOWN_FIELDS, 'known_count'), ('novel', source.NOVEL_FIELDS, 'novel_count')):
        n = marker.get(count_key)
        if type(n) is not int or n <= 0: raise ValueError('Positive physical prediction counts required')
        with np.load(root/(role+'_predictions.npz'), allow_pickle=False) as z:
            if set(z.files) != fields: raise ValueError('Exact prediction fields required; private novel TX/scene forbidden')
            data = {k: z[k].copy() for k in z.files}
        if (not _ids(data['ids'], n) or data['scores'].shape != (n,12) or data['scores'].dtype != np.float64
                or not np.isfinite(data['scores']).all() or data['closed'].shape != (n,2) or data['closed'].dtype != np.int64
                or np.any((data['closed'] < 0) | (data['closed'] >= 6))):
            raise ValueError('Unique aligned IDs, FP64 scores and complete six-class head predictions required')
        for key, roster in (('rx_i', RECEIVERS), ('day_i', (1,2,3))):
            if data[key].shape != (n,) or data[key].dtype != np.int64 or not np.isin(data[key], roster).all():
                raise ValueError('Only aligned original source RX/day metadata is allowed')
        if role == 'known':
            if (data['y'].shape != (n,) or data['y'].dtype != np.int64 or np.any((data['y']<0)|(data['y']>=6))
                    or data['scenario'].shape != (n,) or data['scenario'].dtype.kind != 'U'
                    or not np.isin(data['scenario'], SCENES).all() or data['baseline_closed'].shape != (n,2)
                    or data['baseline_closed'].dtype != np.int64 or not np.array_equal(data['baseline_closed'], data['closed'])):
                raise ValueError('Legal known V labels/scenes and same-path original F0 baseline required')
        rows.append(data)
    if set(rows[0]['ids'].tolist()) & set(rows[1]['ids'].tolist()):
        raise ValueError('Known V and heldout novel physical IDs must be separate')
    return marker, calibration, thresholds, rows[0], rows[1]


def _truth(path, novel):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'ids', 'tx', 'rx_i', 'day_i', 'scenario'}:
            raise ValueError('Independent source novel truth fields differ')
        data = {k: z[k].copy() for k in z.files}
    n = len(novel['ids'])
    if not _ids(data['ids'], n) or set(data['ids'].tolist()) != set(novel['ids'].tolist()):
        raise ValueError('Truth IDs must uniquely cover exactly the completed novel predictions')
    lookup = {sid:i for i,sid in enumerate(data['ids'].tolist())}
    order = np.asarray([lookup[sid] for sid in novel['ids'].tolist()], np.int64)
    if any(value.shape != (n,) for value in data.values()): raise ValueError('One truth value per opaque prediction ID required')
    data = {key:value[order] for key,value in data.items()}
    if (data['tx'].dtype.kind != 'U' or not np.isin(data['tx'], UNKNOWN_TX).all()
            or data['scenario'].dtype.kind != 'U' or not np.isin(data['scenario'], SCENES).all()
            or any(data[k].dtype != np.int64 or not np.array_equal(data[k], novel[k]) for k in ('rx_i','day_i'))):
        raise ValueError('Source-heldout TX/scenes and public RX/day alignment required')
    return data


def _protected(row):
    return (row['coverage']['known_complete'] and row['known_macro_frr'] is not None
            and row['known_macro_frr'] <= .05+1e-12 and row['original_closed_minus_new_open'] is not None
            and row['original_closed_minus_new_open'] <= .02+1e-12)


def _gate(groups, calibrated):
    required = ['overall'] + ['rx/'+str(r) for r in RECEIVERS] + ['scene/'+s for s in SCENES]
    known = all(_protected(row) for row in groups.values())
    unknown = all(groups[name]['coverage']['unknown_complete'] and groups[name]['unknown_macro_rejection'] is not None
                  and groups[name]['unknown_macro_rejection'] >= .9-1e-12 for name in required)
    return dict(known_42_groups_passed=known, source_known_calibration_passed=calibrated,
        unknown_overall_five_RX_six_scenes_90_passed=unknown, unknown_RX_scene_diagnostic_only=True,
        passed=known and calibrated and unknown)


def score(source_root, novel_truth, output):
    output = Path(output)
    if output.exists(): raise FileExistsError('A new immutable source nuisance score root is required')
    marker, calibration, thresholds, known, novel = _prediction(source_root)
    # Sole private truth access; all candidate predictions and thresholds are complete.
    truth = _truth(novel_truth, novel)
    known_masks = dict(source.previous._masks(known['rx_i'], known['scenario']))
    unknown_masks = dict(source.previous._masks(novel['rx_i'], truth['scenario']))
    candidates = {}
    for j, name in enumerate(source.CANDIDATES):
        pi = source.PATHS.index(name.rsplit('::',1)[1]); threshold = thresholds[j]
        def calculate(kmask, umask, roster=UNKNOWN_TX):
            return open_set_metrics(known['y'][kmask], known['closed'][kmask,pi], known['scores'][kmask,j],
                truth['tx'][umask], novel['scores'][umask,j], threshold, tuple(range(6)), roster,
                baseline_prediction=known['baseline_closed'][kmask,pi])
        groups = {key:calculate(kmask, unknown_masks[key]) for key,kmask in known_masks.items()}
        candidates[name] = dict(groups=groups, overall=groups['overall'], threshold=float(threshold),
            by_unknown_TX={tx:calculate(known_masks['overall'], truth['tx']==tx, (tx,)) for tx in UNKNOWN_TX},
            goals=_gate(groups, calibration['methods'][name].get('constraints_passed') is True),
            diagnostic_only=name != source.PRIMARY)
    passed = candidates[source.PRIMARY]['goals']['passed']
    result = dict(status=STATUS, schema=SCHEMA, run_id=source.RUN_ID, code_commit=marker['code_commit'],
        source_only=True, target_reads=0, private_source_novel_truth_read=True, truth_last=True,
        source_gate_passed=passed, primary_candidate=source.PRIMARY,
        eligible_candidates=[source.PRIMARY] if passed else [],
        next_step='SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION' if passed else 'NO_PROMOTION',
        candidates=candidates, known_count=len(known['ids']), novel_count=len(novel['ids']),
        limits=dict(source_unknown_macro_rejection_min=.9, known_macro_frr_max=.05, original_F0_open_loss_max=.02),
        required_unknown_groups='overall + 5 RX + 6 scenes', known_protection_groups=42,
        unknown_RX_scene_groups_diagnostic_only=True, curve_averaging='TX equal weight with intact score ties',
        calibration_scope='same source known V empirical risk, repeated source-heldout novel evaluation',
        formal_target_accessed=False, final_phase3_complete=False,
        primary_rule='rxday::RMD::P1 fixed before observations; ablations cannot replace it')
    output.mkdir(parents=True, exist_ok=False)
    source._write(output/'results.json', result)
    source._write(output/'complete.json', {k:v for k,v in result.items() if k != 'candidates'})
    print(STATUS, result['next_step'], flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True); parser.add_argument('--novel-truth', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    done = score(args.source_root, args.novel_truth, args.output)
    print(json.dumps(dict(status=done['status'], next_step=done['next_step'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
