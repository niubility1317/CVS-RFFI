import pytest
import torch
import inspect

from experiments.mopc_adsb_repro.run import correct, regularization, require_author_training
from experiments.pamf_phase2_practical.external_native import _registration_config
from experiments.pamf_phase2_practical.mopc_stabilization import policy
from experiments.pamf_phase2_practical.satellite_native_c import validate_scope
from experiments.pamf_phase2_practical.mopc_identity import require_author_mopc_row
from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import RegistrationConfig
from experiments.pamf_phase2_practical.vendor_native.paper_reproduction.mopc_hr_non_exemplar_cil_sei.algorithm import correct_old_prototypes


def test_legacy_paper_mopc_entrypoints_are_retired():
    from comparison_suite.registration import _mopc_register
    from paper_reproduction.cvs_aligned.adv3b02_ci_heads import _fit_mopc
    from paper_reproduction.cvs_aligned.class_incremental import _train_mopc
    from paper_reproduction.cvs_aligned.adv3b02_paper_full_ci import fit_mopc_hr_paper_full
    for fn in (_mopc_register, _fit_mopc, _train_mopc, fit_mopc_hr_paper_full):
        required = {name: None for name, arg in inspect.signature(fn).parameters.items()
                    if arg.default is inspect.Parameter.empty}
        with pytest.raises(ValueError, match='retired'):
            fn(**required)


def test_registration_defaults_to_author_behavior():
    assert RegistrationConfig().mopc_semantics == 'public_code'
    assert RegistrationConfig().mopc_max_grad_norm is None
    assert _registration_config({}, 'mopc_hr', 1)['mopc_semantics'] == 'public_code'


def test_other_mopc_semantics_and_clipping_are_rejected():
    with pytest.raises(ValueError):
        RegistrationConfig(mopc_semantics='paper')
    with pytest.raises(ValueError):
        RegistrationConfig(mopc_max_grad_norm=1000)
    with pytest.raises(ValueError):
        _registration_config({}, 'mopc_paper', 1)
    with pytest.raises(ValueError):
        policy({'profile': 'mopc_hr', 'mopc_stabilization': 'clip1000_v1'}, 'native_cil')


def test_satellite_scope_accepts_only_unclipped_mopc():
    job = dict(profile='mopc_hr', capsule='capsules/P01/MIX01/1-1', seed=2026092701,
               design='mix3_v1', satellite_c_points=[[5, 5], [5, 10], [10, 5], [10, 10]],
               satellite_a0_ks=[])
    validate_scope(job)
    with pytest.raises(ValueError):
        validate_scope(dict(job, mopc_stabilization='clip1000_v1'))


def test_adsb_runner_rejects_removed_paper_formula():
    model = torch.nn.Linear(2, 2)
    old = torch.tensor([[1., 2.]])
    with pytest.raises(ValueError):
        regularization(model, model, 'paper')
    with pytest.raises(ValueError):
        correct(old, old, old, 'paper')


def test_scoring_rejects_unverified_or_clipped_mopc_identity():
    row = {'stage':'C', 'arm':'mopc_hr', 'resources':{'native_resources':{'mopc_semantics':'public_code'}}}
    require_author_mopc_row(row)
    with pytest.raises(ValueError): require_author_mopc_row({'stage':'C', 'arm':'mopc_hr', 'resources':{}})
    with pytest.raises(ValueError): require_author_mopc_row({'stage':'C', 'arm':'mopc_hr', 'resources':{'native_resources':{'mopc_semantics':'paper'}}})
    with pytest.raises(ValueError): require_author_mopc_row({'stage':'C', 'arm':'mopc_hr', 'resources':{'native_resources':{'mopc_semantics':'public_code', 'gradient_clip_max_norm':1000}}})


def test_prototype_correction_defaults_to_author_and_rejects_paper():
    old = torch.tensor([[1., 2.], [2., 1.]])
    before = torch.tensor([[2., 3.], [4., -1.]])
    after = before + 1
    actual = correct_old_prototypes(old, before, after)
    expected = .97 * old + .03 * (torch.softmax(old @ before.T, 1) @ (after-before))
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError): correct_old_prototypes(old, before, after, similarity_mode='paper_cosine')


def test_prediction_requires_author_training_identity(tmp_path):
    import json
    marker = tmp_path / 'train_complete.json'
    marker.write_text(json.dumps({'version':'paper'}))
    with pytest.raises(ValueError): require_author_training(tmp_path)
    marker.write_text(json.dumps({'version':'author'}))
    require_author_training(tmp_path)


def test_independent_scoring_checks_mopc_before_truth_or_output(tmp_path):
    import json
    from experiments.pamf_phase2_practical import clean_score, episodic_score
    invalid = {'stage':'C', 'arm':'mopc_hr', 'resources':{}}
    item = {'job':{}, 'rt':{}, 'folder':tmp_path, 'payload':{'rows':[invalid]}}
    with pytest.raises(ValueError, match='provenance'):
        clean_score.score_job(item, tmp_path/'clean-output')
    pred = tmp_path/'pred'; pred.mkdir()
    (pred/'complete.json').write_text(json.dumps({'status':'PREDICTIONS_COMPLETE','truth_read':False}))
    (pred/'predictions.json').write_text(json.dumps({'rows':[invalid]}))
    with pytest.raises(ValueError, match='provenance'):
        episodic_score.run({}, {'prediction':str(pred), 'output':str(tmp_path/'episode-output')})
    assert not (tmp_path/'clean-output').exists()
    assert not (tmp_path/'episode-output').exists()
