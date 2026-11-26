from experiments.pamf_lora_main.w24_formal import FULL, cases, evaluation_modes
from experiments.pamf_lora_main.w24_third_prior import OUTER_BASELINE, fit_inputs, trial


def test_half_primary_has_exactly_three_full_shards_and_a_source_parent():
    candidate = trial('half-primary')
    assert candidate.steps == 1800
    assert candidate.source_modes[-1] == OUTER_BASELINE
    assert set(cases('half-primary')) == {f'{FULL}-{i}' for i in range(3)}
    assert evaluation_modes(FULL, True, third_prior='half-primary')[-2:] == candidate.source_modes[2:4]
    paths = fit_inputs(dict(source_root='/source', source_parent='/frozen', run_root='/run',
                            target_root='/target', classifier_root='/f'), candidate.arms[1])
    assert set(paths) == {'source_root', 'source_parent', 'run_root'}
