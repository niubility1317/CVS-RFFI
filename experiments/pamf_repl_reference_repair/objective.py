"""Try progressively inward reference caps before dropping the extra step."""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from experiments.pamf_repl_reference_dominance import objective as reference

INWARD_FACTORS=(1.,.9999,.999,.99,.95,.9,.75,.5)


class RepairedReferenceGuard(reference.ReferenceDominanceGuard):
    def __init__(self,destination,cap):
        super().__init__(destination,cap)
        self.reference_cap_repairs=0;self.reference_cap_nonzero_repairs=0;self.reference_cap_repair_attempts=0

    def after_step(self,optimizer,args,kwargs):
        previous=reference.cap_extra_to_reference
        reference.cap_extra_to_reference=partial(previous,inward_factors=INWARD_FACTORS)
        try:
            super().after_step(optimizer,args,kwargs)
        finally:
            reference.cap_extra_to_reference=previous
        for d in self.last_details.values():
            self.reference_cap_repairs+=int(d.get('reference_cap_repair_used',False))
            self.reference_cap_nonzero_repairs+=int(d.get('reference_cap_repair_nonzero',False))
            self.reference_cap_repair_attempts+=int(d.get('reference_cap_repair_attempts',0))

    def state_dict(self):
        state=super().state_dict()
        state.update(reference_cap_repair_factors=list(INWARD_FACTORS),
                     reference_cap_repairs=self.reference_cap_repairs,
                     reference_cap_nonzero_repairs=self.reference_cap_nonzero_repairs,
                     reference_cap_repair_attempts=self.reference_cap_repair_attempts)
        return state


class RepairedReferenceObjective(reference.ReferenceDominanceObjective):
    def __init__(self,config):
        super().__init__(config)
        dest=None if config.get('hard_pl_smoke') else Path(config['output_root'])/'step_guard_metrics.jsonl.gz'
        self.parameter_guard=RepairedReferenceGuard(dest,float(config['tuning_recipe']['reference_cap']))


@contextmanager
def installed(config,holder):
    previous=reference.ReferenceDominanceObjective;reference.ReferenceDominanceObjective=RepairedReferenceObjective
    try:
        with reference.installed(config,holder):yield
    finally:reference.ReferenceDominanceObjective=previous
