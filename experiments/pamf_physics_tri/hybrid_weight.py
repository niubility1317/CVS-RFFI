"""Scale only the extra Hybrid objective while retaining all confident U."""
from .hybrid_confidence import HybridConfidence


class HybridWeight(HybridConfidence):
    def additional_fusion_loss(self, views, selected):
        value = super().additional_fusion_loss(views, selected)
        scaled = self.c['hybrid_auxiliary_weight'] * value
        self.metrics['HPL_hybrid_auxiliary_scaled'] = scaled.detach()
        return scaled

    def state_dict(self):
        state = super().state_dict()
        state['U_confidence_only']['hybrid_auxiliary_weight'] = self.c['hybrid_auxiliary_weight']
        return state
