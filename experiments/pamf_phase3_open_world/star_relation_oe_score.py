"""Separate truth-last source and fixed cross-RX scores for the relation test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import star_relation_oe_eval as evaluation
from . import star_consensus_score as common
from .metrics import open_set_metrics
from .scoring import _truth as target_truth_load
from .star_ova_score import _ci
from .star_representation_score import cross_rx_retrieval

RUN_ID = evaluation.RUN_ID
SCHEMA = 'pamf_phase3_star_relation_oe_scores_v1'
STATUS = 'STAR_RELATION_OE_SCORES_COMPLETE'
_read, _write = common._read, common._write


def _prediction(root):
    root = Path(root); m = _read(root/'complete.json')
    if (m.get('schema') != evaluation.SCHEMA or m.get('status') != evaluation.STATUS or m.get('run_id') != RUN_ID
            or m.get('arm') not in evaluation.ARMS or m.get('method') != evaluation.PRIMARY_METHOD
            or m.get('path') != 'P1' or m.get('target_RX') != '24-1'
            or any(m.get(k) != n for k, n in dict(known_count=27000, novel_count=3150, target_count=864,
                target_U_reads=0, source_truth_reads=0, target_truth_reads=0, eval_neural_updates=0).items())
            or m.get('target_Q_fit') is not False or m.get('source_V_neural_fit') is not False
            or m.get('source_V_threshold_fit') is not True or m.get('formal_target_accessed') is not False
            or m.get('all_new_IQ_satellite') is not True or m.get('frozen_models_verified') is not True
            or m.get('thresholds_frozen_before_novel_and_target') is not True
            or m.get('source_unknown_success_target_entry_gate') is not False
            or m.get('source_hard_constraint_groups') != list(evaluation.MAIN_GROUPS)):
        raise ValueError('All source and cross-RX predictions must be complete and frozen before either truth')
    evaluation.validate_config(m['config'])
    if any(m['config'][k] != m[k] for k in ('run_id', 'arm', 'code_commit')):
        raise ValueError('Actual prediction/config identities differ')
    identity = m['model_identity']
    if (any(identity.get(k) != m[k] for k in ('run_id', 'arm', 'code_commit'))
            or identity.get('model_parameters') != 232993 or identity.get('updates') != 2220
            or identity.get('cumulative_updates') != 13320 or identity.get('identity_head_replaced') is not False):
        raise ValueError('Actual updated complete native default-head identity required')
    tau = np.load(root/'threshold.npy', allow_pickle=False)
    cal = _read(root/'calibration/complete.json')
    if (tau.shape != (1,) or tau.dtype != np.float64 or not np.isfinite(tau).all()
            or float(tau[0]) != m['threshold'] or cal.get('threshold') != m['threshold']
            or cal.get('status') != evaluation.CALIBRATION_STATUS
            or any(cal.get(k) != m[k] for k in ('run_id', 'code_commit', 'arm'))):
        raise ValueError('Matching scalar source-only threshold and calibration required')
    rows = {}
    for role, fields, n in (('known', evaluation.KNOWN_FIELDS, 27000), ('novel', evaluation.NOVEL_FIELDS, 3150),
            ('target', evaluation.TARGET_FIELDS, 864)):
        with np.load(root/(role+'_predictions.npz'), allow_pickle=False) as z:
            if set(z.files) != fields:
                raise ValueError('Only exact public predictions; target/source novel truth forbidden')
            a = {k: z[k].copy() for k in z.files}
        if (not common._ids(a['ids'], n) or a['default_logits'].shape != (n, 6) or a['default_logits'].dtype != np.float32
                or not np.isfinite(a['default_logits']).all() or a['closed'].shape != (n,) or a['closed'].dtype != np.int64
                or not np.array_equal(a['closed'], a['default_logits'].argmax(1))
                or a['scores'].shape != (n,) or a['scores'].dtype != np.float64 or not np.isfinite(a['scores']).all()
                or not np.allclose(a['scores'], evaluation.entropy_score(a['default_logits'])-tau[0], rtol=0., atol=1e-12)):
            raise ValueError('Reconstructable finite default-head entropy margins and identities required')
        if role in ('known', 'target'):
            b = a['baseline_closed']
            if b.shape != (n,) or b.dtype != np.int64 or np.any((b < 0) | (b >= 6)):
                raise ValueError('Original F0 same-P1 baseline required')
        if role != 'target':
            for k, roster in (('rx_i', common.RECEIVERS), ('day_i', (1, 2, 3))):
                if a[k].shape != (n,) or a[k].dtype != np.int64 or not np.isin(a[k], roster).all():
                    raise ValueError('Aligned source public RX/day metadata required')
        if role == 'known':
            rebuilt_tau, rebuilt = evaluation.calibrate_known(a['ids'], a['y'], a['closed'], a['baseline_closed'],
                evaluation.entropy_score(a['default_logits']), a['rx_i'], a['scenario'])
            if rebuilt_tau != float(tau[0]) or any(rebuilt[k] != cal[k] for k in rebuilt):
                raise ValueError('Full known-only maximum legal seven-group threshold must reconstruct')
        if role == 'novel':
            for k in ('candidate_fused_unit', 'baseline_fused_unit'):
                u = a[k]
                if (u.shape != (n, 160) or u.dtype != np.float32 or not np.isfinite(u).all()
                        or not np.allclose(np.linalg.norm(u.astype(np.float64), axis=1), 1., rtol=0., atol=2e-6)):
                    raise ValueError('Finite paired source unit features required')
        rows[role] = a
    if any(set(rows[a]['ids'].tolist()) & set(rows[b]['ids'].tolist()) for a, b in (('known', 'novel'), ('known', 'target'), ('novel', 'target'))):
        raise ValueError('Source roles and cross-RX target must be disjoint')
    return m, cal, rows


def _known_pass(row):
    return bool(row['coverage']['known_complete'] and row['known_macro_acceptance'] is not None
        and row['known_macro_acceptance'] >= .9-1e-12 and row['original_closed_minus_new_open'] is not None
        and row['original_closed_minus_new_open'] <= .02+1e-12)


def _full_pass(row):
    return bool(_known_pass(row) and row['coverage']['unknown_complete']
        and row['unknown_macro_rejection'] is not None and row['unknown_macro_rejection'] >= .9-1e-12)


def _metrics(y, closed, scores, uy, uscores, baseline, known_roster=tuple(range(6)), unknown_roster=common.UNKNOWN_TX):
    row = open_set_metrics(y, closed, scores, uy, uscores, 0., known_roster, unknown_roster, baseline_prediction=baseline)
    rates = [np.mean(closed[y == c] == c) if np.any(y == c) else None for c in known_roster]
    current = None if any(v is None for v in rates) else float(np.mean(rates))
    row.update(candidate_macro_closed_accuracy=current,
        original_F0_minus_candidate_closed=None if current is None or row['original_F0_macro_closed_accuracy'] is None else row['original_F0_macro_closed_accuracy']-current)
    return row


def score(prediction_root, novel_truth, target_truth, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('New immutable independent score root required')
    m, cal, rows = _prediction(prediction_root)
    known, novel, target = (rows[k] for k in ('known', 'novel', 'target'))
    # All source + target arrays and known calibration verified before truth.
    nt = common._truth(novel_truth, novel)
    qt = target_truth_load(target_truth, target['ids'])
    if (set(qt['y'].tolist()) != set(range(9)) or not np.isin(qt['scenario'], common.SCENES).all()
            or len(set(qt['receiver'].tolist())) != 1):
        raise ValueError('Full fixed source-six/three-unknown G10 target scoring roster required')
    km = dict(common._masks(known['rx_i'], known['scenario']))
    nm = dict(common._masks(novel['rx_i'], nt['scenario']))
    source_groups = {k: _metrics(known['y'][v], known['closed'][v], known['scores'][v], nt['tx'][nm[k]],
        novel['scores'][nm[k]], known['baseline_closed'][v]) for k, v in km.items()}
    source_main = source_groups['overall']
    source = dict(overall=source_main, groups=source_groups,
        source_known_seven_groups_passed=all(_known_pass(source_groups[k]) for k in evaluation.MAIN_GROUPS),
        source_seven_groups_joint_diagnostic_passed=all(_full_pass(source_groups[k]) for k in evaluation.MAIN_GROUPS),
        source_RX_seen=True, source_novel_historically_exposed=True, target_entry_gate=False)
    retrieval = {key: cross_rx_retrieval(novel[field], novel['ids'], novel['rx_i'], nt['tx'], common.UNKNOWN_TX)
        for key, field in (('candidate', 'candidate_fused_unit'), ('original_F0', 'baseline_fused_unit'))}
    source['cross_RX_1NN'] = retrieval
    source['structure_proxy_is_not_A_N'] = True
    y = qt['y']; k = y < 6; u = ~k
    def target_metric(mask, kr=tuple(range(6)), ur=(6, 7, 8)):
        return _metrics(y[mask & k], target['closed'][mask & k], target['scores'][mask & k],
            y[mask & u], target['scores'][mask & u], target['baseline_closed'][mask & k], kr, ur)
    qm = {'overall': np.ones(len(y), bool)} | {'scene/'+s: qt['scenario'] == s for s in common.SCENES}
    qgroups = {key: target_metric(mask) for key, mask in qm.items()}
    qmain = qgroups['overall']; passed = all(_full_pass(row) for row in qgroups.values())
    eligible = bool(passed and m['arm'] == evaluation.PRIMARY_ARM)
    target_result = dict(overall=qmain, groups=qgroups, joint_seven_groups_passed=passed,
        by_known_TX={str(c): target_metric((y == c) | u, (c,)) for c in range(6)},
        by_unknown_TX={str(c): target_metric(k | (y == c), ur=(c,)) for c in (6, 7, 8)},
        confidence_intervals=dict(A_K_per_class=_ci(qmain['known_per_class'], 'correct_accepted'),
            R_U_per_class=_ci(qmain['unknown_per_class'], 'rejected'),
            scope='per-class Wilson, descriptive one development RX; not cross-RX population uncertainty'),
        receiver_indices=sorted(set(int(v) for v in qt['receiver'])), source_target_RX_disjoint=True,
        historically_exposed_development_RX=True, new_blind_confirmation=False)
    result = dict(schema=SCHEMA, status=STATUS, run_id=RUN_ID, code_commit=m['code_commit'], arm=m['arm'],
        primary_arm=evaluation.PRIMARY_ARM, primary_method=evaluation.PRIMARY_METHOD, source=source, target=target_result,
        primary_summary=qmain, eligible_primary=eligible,
        next_step='ELIGIBLE_NEXT_UDA_FALSIFICATION' if eligible else 'NO_PROMOTION',
        source_unknown_success_target_entry_gate=False, truth_last=True, formal_target_accessed=False,
        target_RX='24-1', target_U_fit=False, target_Q_fit=False, target_DA=False, target_REG=False,
        A_N=None, H_OW=None, full_goal_status='ACTIVE', final_phase3_complete=False,
        costs={k: m[k] for k in ('candidate_real_P1_forward_rows', 'candidate_synthetic_forward_rows',
            'original_F0_real_P1_forward_rows', 'G_real_forward_rows', 'G_synthetic_forward_rows')},
        neural_parameters=499944, auxiliary_neural_parameters=0,
        primary_rule='Fixed relation_oe/entropy_global; overall and six target scenes use CK90/RU90/original-F0 loss2pp; no control replacement or source90 entry gate')
    output.mkdir(parents=True)
    _write(output/'results.json', result)
    _write(output/'complete.json', {k: v for k, v in result.items() if k not in ('source', 'target')})
    print(STATUS, m['arm'], result['next_step'], flush=True)
    return result


def main():
    print(json.dumps(dict(event='STAR_RELATION_OE_SCORE_PROCESS', pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv)), flush=True)
    p = argparse.ArgumentParser(); p.add_argument('--prediction-root', required=True); p.add_argument('--novel-truth', required=True)
    p.add_argument('--target-truth', required=True); p.add_argument('--output', required=True); a = p.parse_args()
    score(a.prediction_root, a.novel_truth, a.target_truth, a.output)


if __name__ == '__main__':
    main()
