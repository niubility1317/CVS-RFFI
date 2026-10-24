import copy
import sys
from types import SimpleNamespace
import pytest
from experiments.pamf_ssl_research import source_optimizer_counterfactual as subject


def plan():
    return dict(kind=subject.KIND, comparisons=subject.COMPARISONS.copy(),
                seeds=[2026092701, 2026092702, 2026092703], source_epoch=20,
                steps_per_seed=1, L_batch=128, U_batch=256,
                optimizer_state='fresh_moments_first_new_native10_step',
                V_reads=0, target_reads=0, U_truth_reads=0, G_training_reads=0)


def test_accepts_only_declared_complete_source_plan():
    subject.validate_plan(plan())


@pytest.mark.parametrize('key,value', [('target_reads', 1), ('U_truth_reads', 1), ('V_reads', 1),
                                     ('G_training_reads', 1), ('source_epoch', 30),
                                     ('steps_per_seed', 2), ('optimizer_state', 'inherited'),
                                     ('seeds', [2026092701]), ('comparisons', ['L_ONLY'])])
def test_rejects_scope_changes(key, value):
    value_plan = copy.deepcopy(plan())
    value_plan[key] = value
    with pytest.raises(ValueError):
        subject.validate_plan(value_plan)


def test_installed_observer_restores_originals_on_error():
    original_facts = subject.source.gradient_facts
    original_class = subject.source.BranchDiagnostic
    with pytest.raises(RuntimeError):
        with subject.installed_counterfactual():
            assert subject.source.gradient_facts is not original_facts
            assert subject.source.BranchDiagnostic is not original_class
            raise RuntimeError('deliberate observation failure')
    assert subject.source.gradient_facts is original_facts
    assert subject.source.BranchDiagnostic is original_class


def test_disposable_steps_are_counted_separately_from_untouched_originals(monkeypatch):
    facts = dict(labeled_only_optimizer=dict(shadow_parameter_gradient_tensors=1), comparisons={
        method: dict(optimizer=dict(shadow_parameter_gradient_tensors=1), relations={
            't2': {'L_full': dict(gradient=dict(cosine=.5), optimizer_extra_descent=dict(cosine=-.5))}})
        for method in subject.COMPARISONS[1:]})
    measured = dict(rows=[dict(model_seed=seed, source_branch_facts=dict(optimizer_counterfactual=copy.deepcopy(facts)))
                          for seed in plan()['seeds']], optimizer_steps=0, parameter_updates=0,
                    parameter_grad_writes=0, persisted_buffer_updates=0)
    monkeypatch.setattr(subject.source, 'collect', lambda *args: copy.deepcopy(measured))
    result = subject.collect({}, plan(), 'test-commit')
    assert result['optimizer_steps'] == result['shadow_optimizer_steps'] == 12
    assert result['parameter_updates'] == result['parameter_grad_writes'] == 12
    assert not any(result['original_model_update_counts'].values())
    assert result['new_formal_training_updates'] == 0
    assert len(result['nonnegative_gradient_but_negative_optimizer_extra_descent']) == 9


def test_current_native_config_needs_no_historical_combination_field(monkeypatch):
    args = SimpleNamespace(epochs=10, max_grad_norm=5., phase1_lr_schedule='cosine',
                           lr=2e-4, phase1_lr_min=1e-6, weight_decay=1e-4)
    monkeypatch.setitem(sys.modules, 'experiments.cric_dura_comparison.f_train',
                        SimpleNamespace(arguments=lambda config, device: args))
    monkeypatch.setitem(sys.modules, 'experiments.cvs_phase1_repair.design',
                        SimpleNamespace(cosine_lr=lambda lr, epoch, epochs, minimum: lr if epoch == 1 and epochs == 10 else 0.))
    observed = subject._first_step_options(dict(segment_epochs=10, epoch_offset=20))
    assert observed == dict(base_lr=2e-4, weight_decay=1e-4)
