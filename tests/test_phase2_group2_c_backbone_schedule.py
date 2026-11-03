import pytest

from experiments.pamf_phase2_practical.group2_c_backbone_schedule import tasks, validate_runtime


def runtime():
    receivers = ['1-1', '14-7', '2-1', '20-1', '7-14', '7-7', '8-8']
    episodes = [f'MIX{i:02d}' for i in range(1, 9)] + [f'{scene}{i:02d}' for scene in ('LS', 'MU', 'LU') for i in range(1, 6)]
    return dict(b_methods=['protonet_cda', 'mrior_sda', 'dadda_sda', 'twostage_sda', 'radionet_ada_knn'],
        c_methods=['mopc_hr', 'orthogonal', 'lora_triplet_knn'], model_seeds=[2026092701, 2026092702, 2026092703, 2026092704],
        target_receivers=receivers, capsules=[f'capsules/{pool}/{episode}/{rx}'
            for pool in ('P01', 'P02', 'P03') for episode in episodes for rx in receivers],
        K=[1, 5, 10, 20], N=[2, 5, 10, 20], run_root='/tmp/nonexistent-fixture',
        gpu_limits={str(i): 2 for i in range(8)}, minimum_free_MiB=6000, score_processes=4,
        c_source_roots={'mopc_hr': '/source/mopc', 'orthogonal': '/source/native', 'lora_triplet_knn': '/source/native'})


def test_tasks_cover_all_bound_pairs_without_duplicate_task_keys():
    rt = runtime()
    validate_runtime(rt)
    rows = tasks(rt)
    assert len(rows) == 5*3*4*483
    assert len({r['key'] for r in rows}) == len(rows)
    assert sum(len(r['points']) for r in rows) == 5*3*(672*16+1260)
    assert all(len(r['points']) == (16 if '/MIX' in r['capsule'] else 1) for r in rows)


def test_matrix_and_gpu_limits_fail_closed():
    rt = runtime()
    rt['b_methods'].append('issl_author_behavior')
    with pytest.raises(ValueError):
        validate_runtime(rt)
    rt = runtime()
    rt['gpu_limits']['7'] = 3
    with pytest.raises(ValueError):
        validate_runtime(rt)


def test_fixed_single_seed_pilot_is_explicit_and_does_not_expand():
    rt = runtime()
    rt['scope'] = 'pilot'
    rt['model_seeds'] = [2026092701]
    rt['capsules'] = ['capsules/P01/MIX01/1-1']
    validate_runtime(rt)
    rows = tasks(rt)
    assert len(rows) == 15
    assert all(row['points'] == [(5, 20)] for row in rows)
