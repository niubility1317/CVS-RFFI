"""Keep the actual supervised AdamW step as the dominant parameter direction."""
from contextlib import contextmanager
from pathlib import Path
from experiments.pamf_repl_dual_ce import objective as dual
from experiments.pamf_ssl_research import optimizer_step_guard as native
from .projection import cap_extra_to_reference


class ReferenceDominanceGuard(native.OptimizerStepGuard):
    def __init__(self, destination, cap):
        super().__init__(destination)
        self.cap = cap
        self.before_reference = None
        self.reference_zero_fallbacks = 0
        self.reference_rounding_fallbacks = 0
        self.minimum_reference_cosine = 1.
        self.reference_cosines_defined = 0

    def _reference_step(self, model, named, optimizer, gradients, *, clip_impl):
        self.before_reference = tuple(p.detach().clone() for _,p in named)
        return super()._reference_step(model, named, optimizer, gradients, clip_impl=clip_impl)

    def after_step(self, optimizer, args, kwargs):
        if self.before_reference is None:
            raise ValueError('Pre-step source parameters required for reference dominance')
        original = native.project_parameters
        def project(named, baseline, joint, full, clean, physical):
            proposed, details = original(named, baseline, joint, full, clean, physical)
            return cap_extra_to_reference(named, self.before_reference, baseline, proposed,
                                          full, clean, physical, details, self.cap)
        native.project_parameters = project
        try:
            super().after_step(optimizer, args, kwargs)
        finally:
            native.project_parameters = original
            self.before_reference = None
        for details in self.last_details.values():
            self.reference_zero_fallbacks += int(details.get('reference_zero_fallback', False))
            self.reference_rounding_fallbacks += int(details.get('reference_rounding_fallback', False))
            cosine = details.get('reference_dominance_cosine')
            if cosine is not None:
                self.minimum_reference_cosine = min(self.minimum_reference_cosine, cosine)
                self.reference_cosines_defined += 1

    def state_dict(self):
        state = super().state_dict()
        state.update(reference_extra_cap=self.cap,
            reference_zero_fallbacks=self.reference_zero_fallbacks,
            reference_rounding_fallbacks=self.reference_rounding_fallbacks,
            minimum_reference_cosine=self.minimum_reference_cosine if self.reference_cosines_defined else None,
            reference_cosines_defined=self.reference_cosines_defined,
            reference_definition='current same-state full-L-only AdamW actual step; previous joint moments retained')
        return state


class ReferenceDominanceObjective(dual.DualCEObjective):
    def __init__(self, config):
        super().__init__(config)
        cap = float(config['tuning_recipe']['reference_cap'])
        destination = None if config.get('hard_pl_smoke') else Path(config['output_root'])/'step_guard_metrics.jsonl.gz'
        self.parameter_guard = ReferenceDominanceGuard(destination, cap)

    def state_dict(self):
        state = super().state_dict()
        guard = state['U_parameter_step_guard']
        if self.c.get('hard_pl_smoke') and guard['calls'] and not guard['reference_cosines_defined']:
            raise ValueError('Real source smoke must produce an actual labeled reference direction')
        return state


@contextmanager
def installed(config, holder):
    previous = dual.DualCEObjective
    dual.DualCEObjective = ReferenceDominanceObjective
    try:
        with dual.installed(config, holder):
            yield
    finally:
        dual.DualCEObjective = previous
