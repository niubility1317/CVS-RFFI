"""Bounded, same-model clean feature reuse; still consume the second loader."""
import torch


class SourceValidationReuse:
    def __init__(self, model, max_bytes=128*1024*1024):
        from model_dual_cvsincnet import DualCVSincNetDisentangle
        # Domain labels only enter MixStyle in this audited class; eval disables it.
        self.allowed = type(model) is DualCVSincNetDisentangle and not any(
            m._forward_hooks or m._forward_pre_hooks for m in model.modules())
        self.model = model
        self.versions = self._versions()
        self.max_bytes, self.used = max_bytes, 0
        self.items = {}
        self.hits = self.misses = 0

    def _versions(self):
        return tuple((id(t), t._version) for t in
                     (*self.model.parameters(), *self.model.buffers()))

    def record(self, index, x, z):
        size=x.numel()*x.element_size()+z.numel()*z.element_size()
        if self.allowed and not self.model.training and self.used+size <= self.max_bytes:
            self.items[index]=(x.detach().cpu().clone(), z.detach().cpu().clone())
            self.used += size

    def take(self, index, x):
        item=self.items.pop(index, None)
        if (item is not None and not self.model.training and self.versions==self._versions()
                and item[0].dtype==x.dtype and item[0].shape==x.shape
                and torch.equal(item[0], x.detach().cpu())):
            self.hits += 1
            return item[1].to(x.device)
        self.misses += 1
        return None
