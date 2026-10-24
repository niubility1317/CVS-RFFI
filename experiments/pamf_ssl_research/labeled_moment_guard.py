"""Protect the real joint step relative to independently accumulated L moments."""
from .optimizer_step_guard import OptimizerStepGuard
from .labeled_moment_reference import LabeledMomentReference


class LabeledMomentGuard(OptimizerStepGuard):
    def __init__(self,destination=None):
        super().__init__(destination)
        self.labeled_reference=LabeledMomentReference()

    def _reference_step(self,model,named,optimizer,gradients,*,clip_impl):
        return self.labeled_reference.step(model,named,optimizer,gradients,clip_impl=clip_impl)

    def state_dict(self):
        return dict(super().state_dict(),copied_existing_moments=False,independent_labeled_moments=True,
            reference_moment_steps=self.labeled_reference.calls,reference_moment_origin='fresh_per_native_segment_L_only',
            reference_coordinates='current_live_parameters')

    def close(self):
        super().close();self.labeled_reference.close()
