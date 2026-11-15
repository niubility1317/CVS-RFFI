import json
import pytest
from experiments.pamf_repl_more_recovery import plan


def test_recovery_matrix_keeps_fixed_rows_and_safe_reuse():
    rows = plan.candidate_rows()
    assert len(rows) == 10
    assert len(plan.REUSE_E50) == 5
    assert plan.RECOVER_E30 in {r['row_id'] for r in rows}
    assert plan.RECOVER_E30 not in plan.REUSE_E50
    assert plan.REUSE_E50 <= {r['row_id'] for r in rows}


def test_reused_source_must_match_row_contract_and_be_target_blind(tmp_path):
    row = next(r for r in plan.candidate_rows() if r['row_id'] in plan.REUSE_E50)
    root = tmp_path/'runs'/plan.RUN/'F'/row['row_id']/'source'
    contract = tmp_path/'contract.json'; contract.write_text('{"classes":[0,1]}')
    c = dict(row_id=row['row_id'], run_id=plan.RUN, tuning_row=row,
             tuning_recipe=plan.RECIPES[row['recipe']], output_root=str(root),
             model_seed=row['model_seed'], protocol_id=row['protocol_id'], source_contract=str(contract),
             hard_pl={}, hard_pl_mixstyle={}, hard_pl_gradient_guard={}, hard_pl_joint_contrast={},
             hard_pl_receiver_extrapolation={}, source_cache='cache', source_rxs=[1], source_days=[1],
             paper_row={'family':'factorial'}, shared_prefix='E20')
    source = tmp_path/'runs'/plan.PREVIOUS_RUN/'F'/row['row_id']/'source'; source.mkdir(parents=True)
    prior = dict(c, run_id=plan.PREVIOUS_RUN)
    done = dict(status='SOURCE_FROZEN',epoch=50,updates=11100,target_access=False,U_truth_read=False,
                head_fit_updates=0,config=prior)
    (source/'final_ssdg.pth').write_bytes(b'checkpoint')
    (source/'complete.json').write_text(json.dumps(done))
    assert plan.reused_e50(c) == source
    done['target_access'] = True
    (source/'complete.json').write_text(json.dumps(done))
    with pytest.raises(ValueError,match='permissions'):
        plan.reused_e50(c)
