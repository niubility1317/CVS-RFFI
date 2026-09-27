"""Previous accepted final raw field, never a periodic stale-gradient cache."""
import copy
import torch


class RefreshRequired(RuntimeError):
    pass


class BRHistory:
    def __init__(self, layout, refresh_interval=8):
        if type(refresh_interval) is not int or refresh_interval < 1:
            raise ValueError('refresh_interval must be a positive integer')
        self.layout = layout
        self.refresh_interval = refresh_interval
        self.gradients = None
        self.signature = None
        self.accepted_index = None
        self.last_refresh = None

    def refresh_reason(self, signature, accepted_index, activity=None):
        if self.gradients is None: return 'missing_history'
        if self.accepted_index != accepted_index - 1: return 'history_age_not_one'
        if self.signature != signature: return 'stage_boundary'
        if self.last_refresh is None or accepted_index-self.last_refresh >= self.refresh_interval:
            return 'periodic_refresh'
        if activity is None: return 'activity_unavailable'
        if len(activity) != len(self.layout.psi_indices): raise ValueError('activity layout')
        if any(bool(active) != (self.gradients[i] is not None)
               for i, active in zip(self.layout.psi_indices, activity)):
            return 'activity_changed'
        return None

    def predictor(self, nonhead_current_activity, h0, signature, accepted_index):
        reason = self.refresh_reason(signature, accepted_index, nonhead_current_activity)
        if reason: raise RefreshRequired(reason)
        if len(h0) != len(self.layout.phi_indices): raise ValueError('head layout')
        result = [None if g is None else g.clone() for g in self.gradients]
        for index, grad in zip(self.layout.phi_indices, h0):
            result[index] = None if grad is None else grad.detach().clone()
        return tuple(result)

    def commit(self, final_raw_gradients, signature, accepted_index, *, refreshed=False):
        if len(final_raw_gradients) != len(self.layout.names): raise ValueError('gradient layout')
        for grad, shape in zip(final_raw_gradients, self.layout.shapes):
            if grad is not None and (grad.shape != shape or grad.dtype != torch.float32):
                raise ValueError('history requires the exact FP32 layout')
        self.gradients = tuple(None if g is None else g.detach().clone() for g in final_raw_gradients)
        self.signature = copy.deepcopy(signature)
        self.accepted_index = int(accepted_index)
        if refreshed: self.last_refresh = int(accepted_index)

    def state_dict(self):
        return copy.deepcopy(dict(version=1, layout=self.layout.signature,
            refresh_interval=self.refresh_interval, gradients=self.gradients,
            signature=self.signature, accepted_index=self.accepted_index,last_refresh=self.last_refresh))

    def load_state_dict(self, state):
        if state['version'] != 1 or state['layout'] != self.layout.signature:
            raise ValueError('BR history layout/version mismatch')
        if state['refresh_interval'] != self.refresh_interval: raise ValueError('BR refresh mismatch')
        grads=state['gradients']
        if grads is not None:
            self.commit(grads,state['signature'],state['accepted_index'])
        else:
            self.gradients=None;self.signature=None;self.accepted_index=None
        self.last_refresh=state['last_refresh']

    @property
    def bytes(self):
        return sum(g.numel()*g.element_size() for g in (self.gradients or ()) if g is not None)
