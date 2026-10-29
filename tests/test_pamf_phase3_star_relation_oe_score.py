import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiments.pamf_phase3_open_world import star_relation_oe_eval as e
from experiments.pamf_phase3_open_world import star_relation_oe_score as s


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root, arm='relation_oe'):
    root.mkdir(); code = 'a'*40
    y = np.repeat(np.arange(6), 4500).astype(np.int64)
    rx = np.tile(e.common.RECEIVERS, 5400).astype(np.int64)
    days = np.tile([1, 2, 3], 9000).astype(np.int64)
    scenes = np.tile(np.asarray(e.common.SCENES), 4500)
    ids = np.asarray(['known-'+str(i) for i in range(27000)])
    z = np.zeros((27000, 6), np.float32); z[np.arange(27000), y] = 8.
    tau, cal = e.calibrate_known(ids, y, y, y, e.entropy_score(z), rx, scenes)
    cal.update(schema=e.SCHEMA, run_id=e.RUN_ID, code_commit=code, arm=arm)
    write_json(root/'calibration/complete.json', cal)
    np.save(root/'threshold.npy', np.asarray([tau], np.float64))
    np.savez(root/'known_predictions.npz', ids=ids, y=y, rx_i=rx, day_i=days, scenario=scenes,
        default_logits=z, closed=y, baseline_closed=y, scores=e.entropy_score(z)-tau)
    nids = np.asarray(['novel-'+str(i) for i in range(3150)])
    # Source unknown intentionally accepted; the cross-RX entry is not gated
    # by source RU. All public scores still reconstruct exactly.
    nz = np.tile(z[0], (3150, 1)); nc = nz.argmax(1).astype(np.int64)
    nrx = np.tile(e.common.RECEIVERS, 630).astype(np.int64)
    ndays = np.tile([1, 2, 3], 1050).astype(np.int64)
    unit = np.full((3150, 160), 1/np.sqrt(160), np.float32)
    np.savez(root/'novel_predictions.npz', ids=nids, rx_i=nrx, day_i=ndays, default_logits=nz,
        closed=nc, scores=e.entropy_score(nz)-tau, candidate_fused_unit=unit, baseline_fused_unit=unit)
    nt = root.parent/'source_truth.npz'
    np.savez(nt, ids=nids, tx=np.repeat(e.common.UNKNOWN_TX, 1050), rx_i=nrx, day_i=ndays,
        scenario=np.tile(np.asarray(e.common.SCENES), 525))
    qy = np.repeat(np.arange(9), 96).astype(np.int64)
    qids = np.asarray(['target-'+str(i) for i in range(864)])
    qz = np.zeros((864, 6), np.float32)
    qz[np.arange(576), qy[:576]] = 8.
    qc = qz.argmax(1).astype(np.int64)
    np.savez(root/'target_predictions.npz', ids=qids, default_logits=qz, closed=qc,
        baseline_closed=qc, scores=e.entropy_score(qz)-tau)
    qt = root.parent/'target_truth.npz'
    np.savez(qt, ids=qids, y=qy, receiver=np.full(864, 9, np.int64), scenario=np.tile(np.asarray(e.common.SCENES), 144))
    ident = dict(run_id=e.RUN_ID, code_commit=code, arm=arm, model_parameters=232993, updates=2220,
        cumulative_updates=13320, identity_head_replaced=False)
    marker = dict(schema=e.SCHEMA, status=e.STATUS, run_id=e.RUN_ID, code_commit=code, arm=arm,
        method=e.PRIMARY_METHOD, path='P1', target_RX='24-1', known_count=27000, novel_count=3150, target_count=864,
        target_U_reads=0, source_truth_reads=0, target_truth_reads=0, eval_neural_updates=0,
        target_Q_fit=False, source_V_neural_fit=False, source_V_threshold_fit=True, formal_target_accessed=False,
        all_new_IQ_satellite=True, frozen_models_verified=True, thresholds_frozen_before_novel_and_target=True,
        source_unknown_success_target_entry_gate=False, source_hard_constraint_groups=list(e.MAIN_GROUPS),
        config=dict(run_id=e.RUN_ID, code_commit=code, arm=arm), model_identity=ident, threshold=tau,
        candidate_real_P1_forward_rows=31014, candidate_synthetic_forward_rows=4,
        original_F0_real_P1_forward_rows=4014, G_real_forward_rows=864, G_synthetic_forward_rows=2)
    write_json(root/'complete.json', marker)
    return nt, qt


def test_source_failure_does_not_block_fixed_cross_rx_falsification(tmp_path):
    root = tmp_path/'predict'; nt, qt = fixture(root)
    with patch.object(e, 'validate_config'), patch.object(s, 'cross_rx_retrieval', return_value={'macro_hit_rate': .8}):
        r = s.score(root, nt, qt, tmp_path/'score')
    assert r['source']['overall']['unknown_macro_rejection'] == 0.
    assert not r['source']['source_seven_groups_joint_diagnostic_passed']
    assert r['target']['overall']['unknown_macro_rejection'] == 1.
    assert r['eligible_primary'] and not r['final_phase3_complete'] and r['full_goal_status'] == 'ACTIVE'
    assert r['A_N'] is None and r['target_DA'] is False and r['target_REG'] is False


def test_corrupt_target_blocks_both_private_truth_reads(tmp_path):
    root = tmp_path/'predict'; nt, qt = fixture(root)
    with np.load(root/'target_predictions.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    data['scores'][0] += .01
    np.savez(root/'target_predictions.npz', **data)
    with patch.object(e, 'validate_config'), patch.object(s.common, '_truth') as source_truth, patch.object(s, 'target_truth_load') as target_truth:
        with pytest.raises(ValueError, match='entropy margins'):
            s.score(root, nt, qt, tmp_path/'score')
        source_truth.assert_not_called(); target_truth.assert_not_called()


def test_control_cannot_replace_primary_and_completed_output_cannot_overwrite(tmp_path):
    root = tmp_path/'predict'; nt, qt = fixture(root, 'relation_control')
    with patch.object(e, 'validate_config'), patch.object(s, 'cross_rx_retrieval', return_value={'macro_hit_rate': .8}):
        r = s.score(root, nt, qt, tmp_path/'score')
        assert not r['eligible_primary'] and r['next_step'] == 'NO_PROMOTION'
        with pytest.raises(FileExistsError):
            s.score(root, nt, qt, tmp_path/'score')
