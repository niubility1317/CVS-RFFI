from types import SimpleNamespace

from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, restorer, trial


class ArchivedW24:
    def stages(self, y, quality, rms, *, robust=False, condition=()):
        return y, quality, rms, robust, condition

    def propose(self, y, quality, rms, edges):
        return self.stages(y, quality, rms, condition=(edges,))


def test_robust_arm_only_changes_the_native_solver_weighting():
    archive = SimpleNamespace(DuraW24=ArchivedW24)
    control = restorer(archive, 'robust_control')
    robust = restorer(archive, 'robust_map')
    assert control.propose('iq', 'quality', 'rms', 'edges') == ('iq', 'quality', 'rms', False, ('edges',))
    assert robust.propose('iq', 'quality', 'rms', 'edges') == ('iq', 'quality', 'rms', True, ('edges',))


def test_robust_trial_stays_full_and_fit_receives_no_reference_or_target():
    spec = trial('robust')
    assert spec.arms == ('robust_control', 'robust_map')
    assert spec.steps == 3600
    assert spec.source_modes[-2:] == ('robust_control_full', 'robust_map_full')
    assert set(cases('robust')) == {f'{FULL}-{i}' for i in range(3)}
    runtime = dict(source_root='/source', run_root='/new', classifier_root='/F', native_target_root='/target')
    assert fit_inputs(runtime, 'robust_map') == dict(source_root='/source', run_root='/new/robust_map')
