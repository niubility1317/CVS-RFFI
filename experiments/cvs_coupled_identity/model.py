"""Public-input prototype: coupled causal envelope memory, fixed parameter budget."""
import torch
from torch.nn import functional as F
from experiments.cvs_energy_identity.model import EnergyCVS,GlobalEnergyBlock,energy_contract

VARIANTS=('coupled_lag1','coupled_lag4')


def delay(x,n):
    if n==0:return x
    if n>=x.shape[-1]:return torch.zeros_like(x)
    return F.pad(x[...,:-n],(n,0))


def coupled_basis_from_clipped(z,lag):
    """z is the existing radial-clipped IQ, not a recovered TX excitation.

    Twelve complex inputs remain: delays0..3 times orders1/3/5. Envelope
    averaging supplies lagged cross-products before the learned complex FIR.
    No centering, sample fitting, labels, RX metadata or normalization state.
    """
    if z.ndim!=3 or z.shape[1]!=2 or lag not in (1,4):raise ValueError('Expected complex IQ and fixed lag1/4')
    power=z.square().sum(1)
    envelope=.5*(power+delay(power,lag))
    terms=[]
    for m in range(4):
        v=delay(z,m);p=delay(envelope,m)/4
        terms.extend((v,v*p[:,None],v*p.square()[:,None]))
    return torch.stack(terms,2)


class CoupledInputBlock(GlobalEnergyBlock):
    def __init__(self,block,lag):
        super().__init__(block);self.envelope_lag=lag
    def forward(self,basis):
        if basis.ndim!=4 or basis.shape[1:3]!=(2,12):raise ValueError('Expected original twelve-term RF basis')
        # The original delay0/order1 slot is exactly the existing clipped IQ.
        coupled=coupled_basis_from_clipped(basis[:,:,0],self.envelope_lag)
        return super().forward(coupled)


def coupled_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unknown coupled-envelope prototype')
    return dict(energy_contract('energy_equivariant'),mode=variant,prototype_only=True,
                behavior_input='clipped z(n-m) times averaged causal envelope powers0/1/2',
                envelope_lag=1 if variant==VARIANTS[0] else 4,
                envelope_average_weights=[.5,.5],behavior_terms=12,
                original_position_power_readout_retained=True,
                no_frequency_correction=True,new_trainable_parameters=0,
                full_GMP=False,hardware_parameter_recovery=False)


class CoupledCVS(EnergyCVS):
    def __init__(self,variant):
        contract=coupled_contract(variant)
        super().__init__('energy_equivariant');self.coupled_variant=variant
        self.core.behavior[0]=CoupledInputBlock(self.core.behavior[0],contract['envelope_lag'])
    def contract(self):
        d=dict(super().contract(),**coupled_contract(self.coupled_variant))
        d['coupled_lift_active']=isinstance(self.core.behavior[0],CoupledInputBlock)
        d['actual_envelope_lag']=getattr(self.core.behavior[0],'envelope_lag',None)
        return d


def build(variant):return CoupledCVS(variant)
