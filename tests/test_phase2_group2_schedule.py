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


def test_smoke_uses_only_allowed_gpu_with_enough_free_memory(monkeypatch):
    from experiments.pamf_phase2_practical import group2_schedule
    free = {'1':1000, '2':7000, '3':12000}
    monkeypatch.setattr(group2_schedule, '_gpu_free', lambda gpu: free[gpu])
    assert group2_schedule._smoke_gpu({'gpu_limits':{'1':2, '2':2}, 'minimum_free_MiB':6000}) == '2'
    assert group2_schedule._smoke_gpu({'gpu_limits':{'1':2}, 'minimum_free_MiB':6000}) is None


def test_runtime_requires_all_frozen_source_release_handles():
    from experiments.pamf_phase2_practical.group2_schedule import validate_runtime
    profiles=['protonet_cda','mrior_sda','dadda_sda','twostage','feature_separation_hu2024','poster_ft','radionet_ada_knn']
    rx=['1-1','14-7','2-1','20-1','7-14','7-7','8-8']
    episodes=[f'MIX{i:02d}' for i in range(1,9)]+[f'{s}{i:02d}' for s in ('LS','MU','LU') for i in range(1,6)]
    capsules=[f'capsules/{p}/{e}/{r}' for p in ('P01','P02','P03') for e in episodes for r in rx]
    rt={'profiles':profiles,'methods':['mopc_hr','orthogonal','lora'],'model_seeds':[2026092701,2026092702,2026092703,2026092704],
        'K':[1,5,10,20], 'N':[2,5,10,20], 'target_receivers':rx, 'capsules':capsules,
        'gpu_limits':{'0':2}, 'score_processes':4, 'minimum_free_MiB':6000}
    with pytest.raises(ValueError,match='frozen source'):
        validate_runtime(rt)
    rt['frozen_sources']={key:'/source/'+key for key in ('F','AF','BASE','FOLD','G')}
    validate_runtime(rt)
