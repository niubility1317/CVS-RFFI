import pytest

from experiments.pamf_phase2_practical.group2_schedule import tasks, closed_prediction


def test_scheduler_builds_each_b_once_per_seed_capsule_and_rejects_duplicates(tmp_path):
    rt = {'run_root':str(tmp_path), 'profiles':['dadda_sda', 'mrior_sda'],
          'methods':['mopc_hr','orthogonal','lora'], 'model_seeds':[1,2],
          'capsules':['capsules/P01/MIX01/1-1'], 'K':[1,5], 'N':[2,20]}
    all_tasks = tasks(rt)
    assert len(all_tasks) == 4
    assert len({t['key'] for t in all_tasks}) == 4
    assert all(len(t['points']) == 4 and t['methods'] == rt['methods'] for t in all_tasks)
    with pytest.raises(ValueError):
        tasks(dict(rt, capsules=rt['capsules'] * 2))


def test_absent_prediction_closure_fails_closed(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        closed_prediction({'output':str(tmp_path)}, expected=1)


def test_full_dispatch_rejects_incomplete_or_changed_comparison_matrix():
    from experiments.pamf_phase2_practical.group2_schedule import validate_runtime
    with pytest.raises(ValueError, match='Incomplete'):
        validate_runtime({'profiles':['dadda_sda'], 'methods':['mopc_hr','orthogonal','lora'],
                          'model_seeds':[2026092701], 'K':[1,5,10,20], 'N':[2,5,10,20],
                          'target_receivers':['1-1'], 'capsules':[]})
