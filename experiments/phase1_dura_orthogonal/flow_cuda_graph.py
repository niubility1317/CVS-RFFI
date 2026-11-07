"""Experimental CUDA-graph step for fixed-shape source IQ flow training."""
import torch

from experiments.phase1_dura_orthogonal.conditional_iq_flow import flow_matching_loss


class FlowGraphStep:
    """Capture forward/backward/clipping; check finite values before optimizer step."""
    def __init__(self, model, optimizer, example):
        if len(example) != 6 or any(not value.is_cuda for value in example):
            raise ValueError('Six fixed-shape CUDA input tensors required')
        self.model = model
        self.optimizer = optimizer
        self.inputs = tuple(value.detach().clone() for value in example)
        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = flow_matching_loss(model, *self.inputs, validate_time=False)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
            del loss
        torch.cuda.current_stream().wait_stream(warm_stream)
        optimizer.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = flow_matching_loss(model, *self.inputs, validate_time=False)
            self.loss.backward()
            self.norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
        if model.out.weight.grad is None:
            raise ValueError('Flow output gradient required')

    def step(self, values):
        if len(values) != len(self.inputs) or any(v.shape != target.shape for v, target in zip(values, self.inputs)):
            raise ValueError('CUDA graph requires the captured input shapes')
        for target, value in zip(self.inputs, values):
            target.copy_(value)
        self.graph.replay()
        if not bool(torch.isfinite(self.loss) & torch.isfinite(self.norm)):
            raise FloatingPointError('Nonfinite source loss or gradient norm')
        self.optimizer.step()
        return self.loss.detach()
