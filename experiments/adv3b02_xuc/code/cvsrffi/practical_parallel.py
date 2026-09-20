"""Bounded process pool for independent records, never shared channel streams."""
import atexit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np

_pools = {}


def _compute(task):
    array, cfg, kwargs = task
    from leo_practical import apply_leo_practical_channel_batch
    return apply_leo_practical_channel_batch(array, cfg, **kwargs)


def shutdown():
    for pool in _pools.values():
        pool.shutdown(wait=True)
    _pools.clear()


atexit.register(shutdown)


def parallel_batch(array, cfg, *, workers=0, **kwargs):
    if workers <= 1 or len(array) <= 1:
        return _compute((array, cfg, kwargs))
    ids, sessions = kwargs['sample_ids'], kwargs['session_ids']
    if len(ids) != len(array) or len(sessions) != len(array) or len(set(map(str, ids))) != len(array):
        raise ValueError('Invalid global physical ID alignment')
    workers = int(workers)
    if workers not in _pools:
        _pools[workers] = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'))
    chunks = np.array_split(np.arange(len(array)), min(workers, len(array)))
    tasks = []
    for indices in chunks:
        local = dict(kwargs, sample_ids=[ids[i] for i in indices],
                     session_ids=[sessions[i] for i in indices])
        tasks.append((array[indices], cfg, local))
    results = list(_pools[workers].map(_compute, tasks))
    return (np.concatenate([r[0] for r in results]),
            [record for r in results for record in r[1]] if kwargs.get('return_meta', True) else None,
            np.concatenate([r[2] for r in results]))
