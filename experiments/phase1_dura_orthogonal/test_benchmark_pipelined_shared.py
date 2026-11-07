import queue
from types import SimpleNamespace

import pytest
import torch

from experiments.phase1_dura_orthogonal import benchmark_pipelined_shared as bench
from experiments.phase1_dura_orthogonal.parallel_source_benchmark import encode


def test_tensor_payload_preserves_values_and_is_independent():
    original = torch.randn(8, 2, 256)
    payload = encode((original,))[0]
    decoded = bench.decode_tensor(payload)
    assert torch.equal(decoded, original)
    decoded[0, 0, 0] += 1
    assert not torch.equal(decoded, original)


def test_collect_pipeline_orders_and_rejects_duplicate(monkeypatch):
    monkeypatch.setattr(bench, 'MODELS', (('a', 'run_a', 'c'), ('b', 'run_b', 'c')))
    p = [SimpleNamespace(exitcode=None)]
    q = queue.Queue()
    for run, batch in (('run_b', 1), ('run_a', 0), ('run_b', 0), ('run_a', 1)):
        q.put(('OK', run, batch, run+str(batch)))
    assert bench.collect_pipeline(q, p, 2) == [['run_a0', 'run_b0'], ['run_a1', 'run_b1']]
    q.put(('OK', 'run_a', 0, 'first'))
    q.put(('OK', 'run_a', 0, 'duplicate'))
    with pytest.raises(ValueError, match='exactly once'):
        bench.collect_pipeline(q, p, 2)
