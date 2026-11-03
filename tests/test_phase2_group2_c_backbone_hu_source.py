import pytest

from experiments.pamf_phase2_practical.group2_c_backbone_hu_source import tasks, validate_runtime


def runtime():
    return dict(c_methods=['mopc_hr', 'orthogonal', 'lora_triplet_knn'], model_seeds=[2026092701],
        source_epochs=200, source_batch=256, gpu_limits={'0': 1, '1': 1, '2': 1},
        minimum_free_MiB=6000, run_root='/tmp/source-aux', source_cache='/source/cache',
        source_contract='/source/contract', c_source_roots={'mopc_hr': '/mopc',
            'orthogonal': '/native', 'lora_triplet_knn': '/native'})


def test_source_only_matrix_and_fixed_budget():
    rt = runtime()
    validate_runtime(rt)
    rows = tasks(rt)
    assert len(rows) == 3 and len({row['key'] for row in rows}) == 3
    assert {r['c_method'] for r in rows} == set(rt['c_methods'])


def test_source_training_rejects_target_handles_and_changed_budget():
    rt = runtime()
    rt['data_root'] = '/target'
    with pytest.raises(ValueError, match='target'):
        validate_runtime(rt)
    rt = runtime()
    rt['source_epochs'] = 1
    with pytest.raises(ValueError, match='budget'):
        validate_runtime(rt)
