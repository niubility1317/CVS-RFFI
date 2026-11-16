"""Dominant CE-only AdamW reference; residual includes other L and all U."""
from contextlib import contextmanager
from pathlib import Path
from experiments.pamf_repl_reference_dominance import objective as reference


class CEReferenceGuard(reference.ReferenceDominanceGuard):
    def _reference_step(self, model, named, optimizer, gradients, *, clip_impl):
        if self.pending is None:
            raise ValueError('Actual current full L and CE anchors required')
        ce = self.pending[2][1]
        if not any(g is not None for g in ce):
            raise ValueError('An actual connected labeled CE gradient is required')
        values, meta = super()._reference_step(model, named, optimizer, ce, clip_impl=clip_impl)
        meta['reference_loss'] = 'ordinary_labeled_CE'
        return values, meta

    def state_dict(self):
        state = super().state_dict()
        state['reference_definition'] = 'current same-state ordinary labeled CE-only AdamW actual step; previous joint moments retained'
        state['additional_update_includes'] = 'other supervised objectives plus unlabeled objectives'
        return state


class CEReferenceObjective(reference.ReferenceDominanceObjective):
    def __init__(self,config):
        super().__init__(config)
        cap=float(config['tuning_recipe']['reference_cap'])
        dest=None if config.get('hard_pl_smoke') else Path(config['output_root'])/'step_guard_metrics.jsonl.gz'
        self.parameter_guard=CEReferenceGuard(dest,cap)


@contextmanager
def installed(config,holder):
    previous=reference.ReferenceDominanceObjective
    reference.ReferenceDominanceObjective=CEReferenceObjective
    try:
        with reference.installed(config,holder):
            yield
    finally:
        reference.ReferenceDominanceObjective=previous
