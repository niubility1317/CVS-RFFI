"""Fixed-budget complex Volterra input with affine-delay-balanced phase memory.

These are received-signal identity features, not fitted TX hardware kernels.
Only the input lift has affine phase covariance; the full model retains CFO.
"""
import torch
from experiments.cvs_energy_identity.model import EnergyCVS,GlobalEnergyBlock
from experiments.cvs_coupled_identity.model import delay,coupled_basis_from_clipped,coupled_contract

VARIANTS=('volterra_lag1','volterra_lag4')


def multiply(a,b):
    return torch.stack((a[:,0]*b[:,0]-a[:,1]*b[:,1],a[:,0]*b[:,1]+a[:,1]*b[:,0]),1)


def volterra_basis_from_clipped(z,lag):
    """Keep 4 delays x 3 orders; mix existing lag4 envelopes with one cubic.

    At t=n-m: C=z[t-l]^2 conjugate(z[t-2l])/4; p=(P[t]+P[t-4])/8.
    Inputs are z, (z*p+C)/2, (z*p*p+C*p)/2. No divisions by IQ, fit,
    batch statistics, metadata, extra parameter or fresh random draw.
    """
    if z.ndim!=3 or z.shape[1]!=2 or lag not in (1,4):raise ValueError('Expected complex IQ and fixed phase lag1/4')
    power=z.square().sum(1);p=(power+delay(power,4))/8
    history=delay(z,lag);conjugate=delay(z,2*lag)*z.new_tensor([1.,-1.])[None,:,None]
    cubic=multiply(multiply(history,history),conjugate)/4
    third=.5*(z*p[:,None]+cubic);fifth=third*p[:,None]
    return torch.stack([delay(term,m) for m in range(4) for term in (z,third,fifth)],2)


class VolterraInputBlock(GlobalEnergyBlock):
    def __init__(self,block,lag):
        super().__init__(block);self.phase_lag=lag;self.envelope_lag=4
    def forward(self,basis):
        if basis.ndim!=4 or basis.shape[1:3]!=(2,12):raise ValueError('Expected original twelve-term RF basis')
        return super().forward(volterra_basis_from_clipped(basis[:,:,0],self.phase_lag))


def volterra_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unknown complex Volterra input candidate')
    lag=1 if variant==VARIANTS[0] else 4
    d=coupled_contract('coupled_lag4')
    d.pop('coupled_lift_active');d.pop('actual_envelope_lag')
    d.update(mode=variant,behavior_input='fixed half envelope memory plus delayed complex cubic; fifth=cubic-mix times envelope',
        phase_lag=lag,actual_phase_lag=lag,envelope_lag=4,actual_envelope_lag=4,
        volterra_lift_active=True,mixture_weights=[.5,.5],cubic_divisor=4,
        cubic_delays=[lag,lag,2*lag],cubic_conjugate_last=True,
        lift_affine_phase_covariant=True,whole_affine_phase_invariant=False,
        complete_volterra=False,hardware_parameter_recovery=False,prototype_only=False,
        identity_core='volterra_identity',new_trainable_parameters=0)
    return d


class VolterraCVS(EnergyCVS):
    def __init__(self,variant):
        contract=volterra_contract(variant)
        super().__init__('energy_equivariant');self.volterra_variant=variant
        self.core.behavior[0]=VolterraInputBlock(self.core.behavior[0],contract['phase_lag'])
    def contract(self):
        d=dict(super().contract(),**volterra_contract(self.volterra_variant))
        d['volterra_lift_active']=isinstance(self.core.behavior[0],VolterraInputBlock)
        d['actual_phase_lag']=getattr(self.core.behavior[0],'phase_lag',None)
        d['actual_envelope_lag']=getattr(self.core.behavior[0],'envelope_lag',None)
        return d

    @torch.no_grad()
    def input_diagnostics(self,x):
        records=[];block=self.core.behavior[0]
        def capture(module,inputs,output):
            actual=inputs[0];z=actual[:,:,0];expected=volterra_basis_from_clipped(z,block.phase_lag)
            control=coupled_basis_from_clipped(z,4);changes={}
            for degree,slots in ((3,[1,4,7,10]),(5,[2,5,8,11])):
                delta=(actual[:,:,slots]-control[:,:,slots]).square().sum((1,2,3)).sqrt()
                base=control[:,:,slots].square().sum((1,2,3)).sqrt().clamp_min(1e-12)
                changes['degree'+str(degree)+'_relative_input_change_mean']=float((delta/base).mean())
            records.append(dict(block='behavior.0.conv',packets=len(actual),complex_terms=actual.shape[2],
                actual_phase_lag=block.phase_lag,actual_envelope_lag=block.envelope_lag,
                input_formula_max_abs_error=float((actual-expected).abs().max()),input_abs_max=float(actual.abs().max()),**changes))
        hook=block.conv.register_forward_hook(capture)
        try:self.features(x)
        finally:hook.remove()
        return dict(scope='Actual Volterra behavior input vs lag4 envelope control; source/public IQ only; no identified hardware',
                    active=isinstance(block,VolterraInputBlock),records=records)

    @torch.no_grad()
    def diagnostics(self,x):return dict(super().diagnostics(x),volterra_input=self.input_diagnostics(x))


def build(variant):return VolterraCVS(variant)
