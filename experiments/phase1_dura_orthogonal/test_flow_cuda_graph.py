import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import pytest
import torch

from experiments.phase1_dura_orthogonal.conditional_iq_flow import ConditionalIQFlow, flow_matching_loss
from experiments.phase1_dura_orthogonal.flow_cuda_graph import FlowGraphStep
from experiments.phase1_dura_orthogonal.benchmark_flow_cuda_graph import graph_train
from experiments.phase1_dura_orthogonal.benchmark_flow_sync import exact, train


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires GPU')
def test_graph_replays_exact_training_and_blocks_nonfinite():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    eager = ConditionalIQFlow(width=8, steps=4).cuda()
    graph = ConditionalIQFlow(width=8, steps=4).cuda()
    graph.load_state_dict(eager.state_dict())
    opt_a = torch.optim.AdamW(eager.parameters(), lr=.0003, weight_decay=.001)
    opt_b = torch.optim.AdamW(graph.parameters(), lr=.0003, weight_decay=.001)
    generator = torch.Generator().manual_seed(42)
    fields = [torch.randn(4, 2, 256, generator=generator).cuda() for _ in range(3)]
    fields += [torch.full((4,), .1, device='cuda'), torch.randn(4, 2, 256, generator=generator).cuda(),
               torch.rand(4, generator=generator).cuda()]
    runner = FlowGraphStep(graph, opt_b, fields)
    for _ in range(3):
        loss_a = flow_matching_loss(eager, *fields, validate_time=False)
        opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(eager.parameters(), 1., error_if_nonfinite=True)
        opt_a.step()
        loss_b = runner.step(fields)
        assert torch.equal(loss_a.detach(), loss_b)
        assert all(torch.equal(a, b) for a, b in zip(eager.parameters(), graph.parameters()))
    before = [p.detach().clone() for p in graph.parameters()]
    broken = list(fields)
    broken[4] = torch.full_like(broken[4], float('nan'))
    with pytest.raises(FloatingPointError):
        runner.step(broken)
    assert all(torch.equal(a, b) for a, b in zip(before, graph.parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires GPU')
def test_graph_epoch_prefetch_matches_eager_training():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(51)
    iq = tuple(torch.randn((64, 2, 256), generator=generator).cuda() for _ in range(3))
    fields = (*iq, torch.full((64,), .1, device='cuda'), torch.randn((64, 2, 256), generator=generator).cuda())
    eager = train(fields, 'conditional', True, 1)
    graph = graph_train(fields, 1, epoch_times=True)
    assert exact(eager, graph)
