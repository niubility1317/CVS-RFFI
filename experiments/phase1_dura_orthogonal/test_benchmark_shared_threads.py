from experiments.phase1_dura_orthogonal.benchmark_shared_threads import MODELS, compare_encoded


def test_compares_each_model_and_each_batch():
    common = [[b'a', b'b', b'c'], [b'd', b'e', b'f']]
    by_run = {run: [common[0][i], common[1][i]] for i, (_, run, _) in enumerate(MODELS)}
    assert compare_encoded(common, by_run) == 0
    by_run[MODELS[1][1]][1] = b'changed'
    assert compare_encoded(common, by_run) == 1
