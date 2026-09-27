from copy import deepcopy
import pytest
from cvsrffi.xuc_fusion.ir_experiments import build_matrices, validate_matrix, validate_convergence_for_launch


def test_all_methods_native_DR_enabled():
    for doc in build_matrices()[:2]:
        assert all(i['row']['joint']['native_dr'] for i in doc['rows'])


def test_equal_physical_exposure():
    doc=build_matrices()[0];validate_matrix(doc)
    broken=deepcopy(doc);broken['rows'][0]['row']['joint']['unlabeled_batch']=128
    with pytest.raises(ValueError):validate_matrix(broken)


def test_no_target_selection():
    doc=build_matrices()[0]
    assert doc['checkpoint']['selection']=='source_V_only'
    assert doc['target_policy'].endswith('independent_truth_scorer')


def test_budget_audit_not_convergence():
    performance,_,confirmation=build_matrices()
    assert performance['accepted_main_steps']==44400 and not performance['scientific_convergence_claim']
    with pytest.raises(ValueError):validate_convergence_for_launch(confirmation)


def test_unapproved_runs_not_launched():
    assert all(d['launch'] is False for d in build_matrices())


def test_encoder_off_keeps_head_supervision():
    rows=[i['row'] for i in build_matrices()[0]['rows'] if i['label']=='IR_ENCODER_OFF']
    for row in rows:
        assert not row['response']['encoder_adversary'] and row['response']['unlabeled_head']
        assert row['joint']['outer_adv_weight']>0 and not row['joint']['disable_adversarial_head_loss']
