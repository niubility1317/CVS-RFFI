from types import SimpleNamespace
import torch
from experiments.cvs_t1_e50.channel import Practical
from .boundary import require_training_role, training_tensor
from . import design as d

def load_dura(device):
    from experiments.pamf_stage2_inference import runtime, design as gd
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    if (d.DURA_RELEASE/'release_commit.txt').read_text().strip() != d.DURA_COMMIT:
        raise ValueError('Pinned DURA execution commit differs')
    args = SimpleNamespace(parent_run=d.DURA_PARENT, parent_action=d.PROJECT/'runs'/gd.ACTION_RUN)
    runtime.validate_source(args)
    dev = torch.device(device)
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type=='cuda' else []
    # Constructing a frozen G must not consume F's training RNG sequence.
    with torch.random.fork_rng(devices=devices):
        model, payload = runtime.load_model(args, device)
    if runtime.checkpoint(args) != d.DURA_CHECKPOINT:
        raise ValueError('Frozen DURA v1 checkpoint differs')
    return InferenceEngine(model, reuse=True, graph=True, native=False)

class RestoredPractical:
    def __init__(self, role='L_s'):
        require_training_role(role)
        self.original = Practical(arm='SCRATCH50')
        self.engine = None
        self.calls = self.packets = self.restored_packets = 0
    def __deepcopy__(self, memo):
        # Teacher has no labeled feature-hook calls; do not duplicate G or its graph.
        result = type(self)()
        memo[id(self)] = result
        return result
    def apply(self, x, meta, epoch, step):
        satellite, records = self.original.apply(x, meta, epoch, step)
        if self.engine is None:
            self.engine = load_dura(x.device)
        restored = training_tensor(self.engine, satellite)
        self.calls += 1
        self.packets += len(x)
        self.restored_packets += len(x)
        return restored, [dict(r, DURA_restored=True, DURA_frozen=True) for r in records]
    def audit(self):
        return dict(self.original.audit(), restored_augmentation=True, G_checkpoint=str(d.DURA_CHECKPOINT),
            G_runtime_commit=d.DURA_COMMIT, G_frozen=True, G_parameters=266951,
            role='L_s', U_G_forwards=0, native=False, graph=True, reuse=True, raw_fallback=False)
    def summary(self):
        if self.original.packets != self.restored_packets:
            raise ValueError('Every satellite augmentation must pass DURA exactly once')
        if self.engine is not None and (self.engine.pipeline.model.training or any(p.requires_grad or p.grad is not None for p in self.engine.pipeline.model.parameters())):
            raise ValueError('Frozen DURA gradient/state permission changed')
        return dict(self.original.summary(), DURA_packets=self.restored_packets, DURA_calls=self.calls,
            G_updates=0, G_gradients=False, graph_setup=[] if self.engine is None else self.engine.setup(),
            raw_fallbacks=0, U_G_forwards=0)
