"""CUDA graph replay with explicit input replacement and owned outputs.

Warmup/capture follow PyTorch 2.1 CUDA semantics. No compilation, mixed
precision, approximation, optimizer, model replacement, or fallback is used.
"""
import time
import torch


def validate_iq(x, device=None, shape=None):
    if x.dtype != torch.float32 or x.ndim != 3 or x.shape[1:] != (2,256) or not len(x):
        raise ValueError('Nonempty FP32[B,2,256] IQ required')
    if device is not None and x.device != device:
        raise ValueError('Captured device differs')
    if shape is not None and tuple(x.shape) != shape:
        raise ValueError('Captured static shape differs')


class FrozenPipeline:
    def __init__(self, restorer, reader=None, *, graph_safe=False):
        self.restorer, self.reader = restorer, reader
        self.graph_safe = graph_safe
        for module in (restorer, reader):
            if module is not None and (module.training or any(p.requires_grad for p in module.parameters())):
                raise ValueError('Only eval-mode frozen inference modules may be captured')

    def __call__(self, x):
        if self.graph_safe:
            from .program import outputs
            iq,variance,_,_=outputs(self.restorer,x)
        else:
            iq,variance,_,_=self.restorer.training_outputs(x)
        if self.reader is None:
            return (iq,)
        features = self.reader.features(iq)
        return iq, variance.mean(-1), features, self.reader.classify_features(features)


class CapturedCall:
    @torch.no_grad()
    def __init__(self, function, sample, warmup=3):
        validate_iq(sample)
        if sample.device.type != 'cuda' or type(warmup) is not int or warmup < 3:
            raise ValueError('CUDA sample and at least3 side-stream warmups required')
        self.device, self.shape = sample.device, tuple(sample.shape)
        self.function = function  # Retain parameters, cached Sinc filters and buffers.
        self.input = sample.detach().contiguous().clone()
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        started = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                function(self.input)
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.warmup_seconds = time.perf_counter()-started
        started = time.perf_counter()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = tuple(function(self.input))
        torch.cuda.synchronize(self.device)
        self.capture_seconds = time.perf_counter()-started
        if not self.outputs or any(not isinstance(v,torch.Tensor) for v in self.outputs):
            raise ValueError('Tensor tuple output required')

    @torch.no_grad()
    def __call__(self, x):
        validate_iq(x, self.device, self.shape)
        self.input.copy_(x)
        self.graph.replay()
        # Captured outputs have fixed addresses and are overwritten by replay.
        # Return independent storage to preserve ordinary inference semantics.
        return tuple(v.clone() for v in self.outputs)


class GraphCache:
    def __init__(self, function):
        self.function = function
        self.calls = {}

    @torch.no_grad()
    def __call__(self, x):
        validate_iq(x)
        key = (x.device, tuple(x.shape))
        if key not in self.calls:
            self.calls[key] = CapturedCall(self.function, x)
        return self.calls[key](x)

    def setup(self):
        return [dict(batch=shape[0], warmup_seconds=v.warmup_seconds,
                     capture_seconds=v.capture_seconds) for (_,shape),v in self.calls.items()]
