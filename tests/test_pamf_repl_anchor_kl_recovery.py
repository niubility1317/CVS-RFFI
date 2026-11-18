import pytest

from experiments.cric_dura_comparison.io import write
from experiments.pamf_repl_anchor_kl_recovery import plan
from experiments.pamf_repl_anchor_kl_recovery.runner import verify_frozen


def test_recovery_disallows_changed_recipe_or_target_access(tmp_path):
    row = plan.candidate_rows()[0]
    keys = ('row_id', 'tuning_row', 'tuning_recipe', 'model_seed', 'source_contract',
            'source_cache', 'protocol_id', 'shared_prefix', 'paper_row')
    expected = dict.fromkeys(keys)
    expected.update(row_id=row['row_id'], tuning_recipe=plan.RECIPES[row['recipe']])
    done = dict(status='SOURCE_FROZEN', epoch=50, updates=11100, target_access=False,
                U_truth_read=False, config=dict(expected, run_id=plan.PREVIOUS_RUN))
    (tmp_path / 'final_ssdg.pth').touch()
    for epoch, calls in ((30, 2220), (50, 4440)):
        folder = tmp_path / f'segment{epoch}'
        folder.mkdir()
        write(folder / 'repl_state.json', dict(calls=calls, U_truth_read=False))
        (folder / 'U_pseudo_observations.jsonl.gz').touch()
    verify_frozen(tmp_path, done, expected)
    with pytest.raises(ValueError, match='exact source-only'):
        verify_frozen(tmp_path, dict(done, target_access=True), expected)
    with pytest.raises(ValueError, match='exact source-only'):
        verify_frozen(tmp_path, done, dict(expected, tuning_recipe={'kl_factor': 0.}))


def test_recovery_keeps_matrix_and_excludes_third_worker_and_gpu_seven():
    assert plan.MAX_GPU_PROCESSES == 2 and plan.GPUS == tuple(range(7))
    assert plan.RUN != plan.PREVIOUS_RUN and len(plan.candidate_rows()) == 28
