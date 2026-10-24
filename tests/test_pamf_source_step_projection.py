import copy
import pytest
from experiments.pamf_ssl_research import source_step_projection as subject


def plan():
    return dict(kind=subject.KIND, source_rules=subject.RULES.copy(), seeds=[2026092701, 2026092702, 2026092703],
                source_epoch=20, steps_per_seed=1, L_batch=128, U_batch=256,
                V_reads=0, target_reads=0, U_truth_reads=0, G_training_reads=0)


@pytest.mark.parametrize('key,value', [('kind', 'different'), ('seeds', [2026092701]),
                                     ('source_epoch', 30), ('steps_per_seed', 2),
                                     ('target_reads', 1), ('V_reads', 1), ('U_truth_reads', 1), ('G_training_reads', 1)])
def test_rejects_changed_source_scope(key, value):
    changed = plan()
    changed[key] = value
    with pytest.raises(ValueError):
        subject.validate_plan(changed)


def test_source_scientific_rule_is_fixed():
    changed = copy.deepcopy(plan())
    changed['source_rules']['minimum_whole_increment_norm_retention_per_seed'] = 0.
    with pytest.raises(ValueError):
        subject.validate_plan(changed)


def test_restores_all_observers_on_error():
    facts = subject.source.gradient_facts
    klass = subject.source.BranchDiagnostic
    step = subject.optimizer_source.shadow_step
    build = subject.shadow.build
    with pytest.raises(RuntimeError):
        with subject.installed_counterfactual():
            assert subject.source.gradient_facts is not facts
            assert subject.optimizer_source.shadow_step is not step
            raise RuntimeError('intentional failure')
    assert subject.source.gradient_facts is facts
    assert subject.source.BranchDiagnostic is klass
    assert subject.optimizer_source.shadow_step is step
    assert subject.shadow.build is build
