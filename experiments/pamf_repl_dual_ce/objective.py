"""Add the actual ordinary labeled CE as a second AdamW-update anchor."""
from contextlib import contextmanager
import torch
from experiments.pamf_repl_step_guard.objective import StepProtectedREPL
from experiments.pamf_repl_weight.source_ce_anchor import capture_identity_loss


class DualCEObjective(StepProtectedREPL):
    def __init__(self, config):
        super().__init__(config)
        self.ce_derivative_calls = 0

    def guide(self, labeled, student):
        named = [(name, p) for name, p in student.named_parameters() if p.requires_grad]
        ce = getattr(student, '_diagnostic_identity_loss', None)
        if not torch.is_tensor(ce) or ce.ndim or not ce.requires_grad:
            raise ValueError('The original labeled CE tensor is required')
        gradient = tuple(None if g is None else g.detach().clone() for g in
            torch.autograd.grad(ce, [p for _, p in named], retain_graph=True, allow_unused=True))
        if not any(g is not None for g in gradient):
            raise ValueError('Labeled CE must reach trainable parameters')
        guard = self.parameter_guard
        original_capture = guard.capture
        if 'capture' in guard.__dict__:
            raise ValueError('A single native anchor capture is required')
        def capture(model, incoming, full, clean, physical):
            if (model is not student or len(incoming) != len(named)
                    or any(a != b or p is not q for (a, p), (b, q) in zip(named, incoming))
                    or any(g is not None for g in clean+physical)):
                raise ValueError('Native full-L and exact labeled CE anchor required')
            return original_capture(model, incoming, full, gradient, physical)
        guard.capture = capture
        try:
            value = super().guide(labeled, student)
        finally:
            del guard.capture
        self.ce_derivative_calls += 1
        return value

    def state_dict(self):
        result = super().state_dict()
        guard = result['U_parameter_step_guard']
        if guard['calls'] != self.ce_derivative_calls:
            raise ValueError('One actual labeled CE derivative per protected step required')
        guard.update(extra_autograd_calls=self.ce_derivative_calls,
                     second_anchor='ordinary_labeled_CE_recorded_as_L_clean')
        if guard['calls']:
            details = self.parameter_guard.last_details.values()
            if self.c.get('hard_pl_smoke') and all(
                    'L_clean' in value.get('near_zero_component_anchors_omitted', []) for value in details):
                raise ValueError('At least one actual CE anchor must be active')
            for value in details:
                cosine = value['anchor_cosines_after']['L_clean']
                if cosine is not None and cosine < -1.000001e-6:
                    raise ValueError('Actual labeled CE update constraint failed')
        return result


@contextmanager
def installed(config, holder):
    from experiments.cric_dura_comparison import t1_model
    from experiments.pamf_paper_completion.model import installed as paper_scope
    from experiments.pamf_repl_step_guard.plan import validate
    from experiments.pamf_ssl_research.optimizer_step_guard import installed as optimizer_scope
    with paper_scope(config, holder, ssl_enabled=True, objective_factory=DualCEObjective, validate=validate):
        native_install = t1_model.install
        def install(model, cfg):
            return capture_identity_loss(native_install(model, cfg))
        t1_model.install = install
        try:
            with optimizer_scope():
                yield
        finally:
            t1_model.install = native_install
            objective = holder.get('objective')
            if objective is not None:
                observations = objective.close_pseudo_observation()
                if not config.get('hard_pl_smoke') and (observations['calls'] != objective.calls
                        or observations['seen'] != objective.seen or observations['accepted'] != objective.accepted):
                    raise ValueError('Pseudo observation counts differ from actual U execution')
