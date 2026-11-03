from pathlib import Path
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


def test_poster_pilot_is_separate_and_has_only_three_pairs():
    rt = runtime()
    rt.update(scope='poster_pilot', b_methods=['poster_ft'], model_seeds=[2026092701],
              capsules=['capsules/P01/MIX01/1-1'])
    validate_runtime(rt)
    assert len(tasks(rt)) == 3


def test_hu_pilot_requires_source_auxiliary_checkpoint_root():
    rt = runtime()
    rt.update(scope='hu_pilot', b_methods=['feature_separation_hu2024'], model_seeds=[2026092701],
              capsules=['capsules/P01/MIX01/1-1'])
    with pytest.raises(ValueError, match='Hu source'):
        validate_runtime(rt)
    rt['hu_aux_run_root'] = '/source/aux'
    validate_runtime(rt)
    assert len(tasks(rt)) == 3


def test_full_seven_b_matrix_selects_seed_bound_aux_and_pilot_reuse():
    rt = runtime()
    rt.update(scope='full7', b_methods=rt['b_methods']+['poster_ft', 'feature_separation_hu2024'],
        hu_aux_roots_by_seed={str(seed): '/aux/one' if seed == 2026092701 else '/aux/more'
            for seed in rt['model_seeds']},
        pilot_roots={'base': '/pilot/base', 'poster_ft': '/pilot/poster',
                     'feature_separation_hu2024': '/pilot/hu'})
    validate_runtime(rt)
    rows = tasks(rt)
    assert len(rows) == 7*3*4*483
    selected = [job for job in rows if 'reuse_prediction' in job]
    assert len(selected) == 21
    assert all(job['seed'] == 2026092701 and job['capsule'] == 'capsules/P01/MIX01/1-1'
               and (5, 20) in job['points'] for job in selected)
    assert all(Path('/pilot/hu') in Path(job['reuse_prediction']).parents for job in selected
               if job['b_method'] == 'feature_separation_hu2024')
    del rt['hu_aux_roots_by_seed']['2026092704']
    with pytest.raises(ValueError, match='Hu source'):
        validate_runtime(rt)


def test_full_seven_b_rejects_missing_pilot_results_plan():
    rt = runtime()
    rt.update(scope='full7', b_methods=rt['b_methods']+['poster_ft', 'feature_separation_hu2024'],
        hu_aux_roots_by_seed={str(seed): '/aux' for seed in rt['model_seeds']})
    with pytest.raises(ValueError, match='pilot'):
        validate_runtime(rt)
