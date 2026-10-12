"""Disable explicit CFO identity coordinate without changing historical weights."""
import torch
from experiments.cvs_reference_identity.model import ExcitationResponse

class NoExplicitCFO(ExcitationResponse):
    def forward(self,x):
        v=super().forward(x)
        if v.shape[1]!=29:raise ValueError('Historical reference coordinate layout changed')
        return torch.cat((v[:,:27],torch.zeros_like(v[:,27:28]),v[:,28:]),dim=1)

    def contract(self):
        value=super().contract();value.update(explicit_CFO_identity_input=False,constant_zero_coordinate=27,active_response_features=28,internal_CFO_compensation=True)
        return value

def apply(model):
    modules=[m for m in model.modules() if type(m) is ExcitationResponse]
    if len(modules)!=1:raise ValueError('Expected exactly one unmodified reference response')
    # No construction, RNG draw, parameter or buffer mutation. This policy is
    # part of the evaluated function, not a claim that the old model trained so.
    modules[0].__class__=NoExplicitCFO
    return dict(explicit_CFO_identity_input=False,coordinate=27,shape=29,active_features=28,parameter_mutations=0,internal_CFO_compensation=True,historical_checkpoint_training_policy='explicit_CFO',function_policy='masked_CFO_diagnostic')
