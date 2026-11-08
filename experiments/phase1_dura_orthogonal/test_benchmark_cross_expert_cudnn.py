import torch
from experiments.phase1_dura_orthogonal import benchmark_cross_expert_cudnn as bench


def test_condition_selects_requested_autotuning(monkeypatch):
    calls=[]
    monkeypatch.setattr(bench,'train',lambda fields,resident,steps: calls.append((torch.backends.cudnn.benchmark,resident,steps)))
    before=torch.backends.cudnn.benchmark
    try:
        bench.run_condition(('x',),False,2)
        bench.run_condition(('x',),True,4)
        assert calls==[(False,False,2),(True,False,4)]
    finally:
        torch.backends.cudnn.benchmark=before
