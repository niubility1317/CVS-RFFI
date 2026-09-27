"""Record actual divisors per L/U call without altering the origin EMA recursion."""
from copy import deepcopy

class FieldNormalizer:
    def __init__(self, base, replay=None):
        self.base=base
        self.replay=deepcopy(replay)
        self.used=[]
    def normalize(self, components, *, active=None):
        if self.replay is None:
            normalized,reported=self.base.normalize(components,active=active)
            actual={k:max(float(reported[k]),self.base.epsilon) for k in components}
        else:
            if len(self.used)>=len(self.replay): raise ValueError('extra normalization call in corrector')
            actual=self.replay[len(self.used)]
            if set(actual)!=set(components): raise ValueError('normalization component mismatch')
            normalized={k:v/actual[k] for k,v in components.items()}
            reported=actual
        self.used.append(deepcopy(actual))
        return normalized,reported
    def finish(self):
        if self.replay is not None and len(self.used)!=len(self.replay): raise ValueError('missing normalization call in corrector')
    def state_dict(self): return self.base.state_dict()
