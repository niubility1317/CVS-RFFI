"""Wider continuous correction around frozen DURA."""
from experiments.phase1_dura_context.model import ContextRecovery

class WideContextRecovery(ContextRecovery):
    def forward(self,received,initial,log_variance):
        ordinary=super().forward(received,initial,log_variance)
        return initial+4.0*(ordinary-initial)
