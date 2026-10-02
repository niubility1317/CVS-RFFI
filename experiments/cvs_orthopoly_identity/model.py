"""Packet-local orthogonal received-envelope coordinates; no fitted state.

This is a closed-form input transform, not target adaptation, a recovered PA
model, or a claim of TX/RX separation. Raw clipped IQ is retained unchanged.
"""
import torch
from experiments.cvs_energy_identity.model import EnergyCVS,GlobalEnergyBlock,energy_contract
from experiments.cvs_coupled_identity.model import delay

VARIANTS=('orthopoly_instant','orthopoly_memory4')
VARIANCE_FLOOR=1e-8
MASS_FLOOR=1e-12
SCALES=(1.,4.,16.)


def packet_basis(z,lag,return_moments=False):
    if z.ndim!=3 or z.shape[1]!=2 or lag not in (0,4):
        raise ValueError('Expected complex IQ and registered envelope lag0/4')
    power=z.square().sum(1)
    envelope=power/4 if lag==0 else (power+delay(power,4))/8
    terms=[];moments=[]
    for m in range(4):
        v=delay(z,m);p=delay(envelope,m);weight=v.square().sum(1)
        mass=weight.sum(-1,keepdim=True);denom=mass.clamp_min(MASS_FLOOR)
        mean=(weight*p).sum(-1,keepdim=True)/denom
        centered=p-mean
        variance=(weight*centered.square()).sum(-1,keepdim=True)/denom
        skew=(weight*centered.pow(3)).sum(-1,keepdim=True)/denom
        projection=skew/variance.clamp_min(VARIANCE_FLOOR)
        second=centered.square()-projection*centered-variance
        terms.extend((v,4*v*centered[:,None],16*v*second[:,None]))
        if return_moments:moments.append(dict(delay=m,mass=mass,mean=mean,variance=variance,projection=projection))
    basis=torch.stack(terms,2)
    return (basis,moments) if return_moments else basis


@torch.no_grad()
def basis_diagnostics(z,lag):
    actual,moments=packet_basis(z,lag,True)
    power=z.square().sum(1);envelope=power/4 if lag==0 else (power+delay(power,4))/8
    records=[]
    for m,stats in enumerate(moments):
        v=delay(z,m);p=delay(envelope,m)
        original=torch.stack((v,v*p[:,None],v*p.square()[:,None]),2)
        new=actual[:,:,3*m:3*m+3]
        mu=stats['mean'];a=stats['projection'];variance=stats['variance']
        reconstructed=torch.stack((new[:,:,0],new[:,:,1]/4+mu[:,None]*v,
            new[:,:,2]/16+(2*mu+a)[:,None]*new[:,:,1]/4+(mu.square()+variance)[:,None]*v),2)
        def gram(x):
            x=x.double();g=torch.einsum('biqt,birt->bqr',x,x)
            norms=g.diagonal(dim1=1,dim2=2).clamp_min(0).sqrt()
            return g/(norms[:,:,None]*norms[:,None,:]).clamp_min(MASS_FLOOR),norms
        before,old_norm=gram(original);after,new_norm=gram(new)
        eligible=(stats['mass'][:,0]>MASS_FLOOR)&(variance[:,0]>=VARIANCE_FLOOR)&(new_norm.min(1).values>1e-8)
        off=torch.tensor([[False,True,True],[True,False,True],[True,True,False]],device=z.device)
        records.append(dict(delay=m,packets=len(z),eligible_packets=int(eligible.sum()),
            raw_order1_max_abs_error=float((new[:,:,0]-v).abs().max()),
            reconstruction_max_abs_error=float((reconstructed-original).abs().max()),
            original_normalized_gram_offdiag_mean=float(before[eligible][:,off].abs().mean()) if eligible.any() else None,
            orthogonal_normalized_gram_offdiag_max=float(after[eligible][:,off].abs().max()) if eligible.any() else None,
            order_energy_mean=new.square().sum((1,3)).mean(0).tolist(),
            mean_envelope=float(mu.mean()),mean_envelope_variance=float(variance.mean()),
            near_degenerate_packets=int((variance[:,0]<VARIANCE_FLOOR).sum())))
    return dict(active=True,envelope_lag=lag,complex_terms=12,per_packet_only=True,
        persistent_moments=False,energy_whitening=False,records=records)


class OrthogonalInputBlock(GlobalEnergyBlock):
    def __init__(self,block,lag):super().__init__(block);self.envelope_lag=lag
    def forward(self,basis):
        if basis.ndim!=4 or basis.shape[1:3]!=(2,12):raise ValueError('Expected twelve complex input terms')
        return super().forward(packet_basis(basis[:,:,0],self.envelope_lag))


def orthopoly_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered packet orthogonal basis')
    lag=0 if variant==VARIANTS[0] else 4
    return dict(energy_contract('energy_equivariant'),mode=variant,identity_core='orthopoly_identity',
        behavior_input='weighted centered envelope and second orthogonal polynomial; raw clipped z retained',
        envelope_lag=lag,actual_envelope_lag=lag,behavior_terms=12,orthopoly_active=True,
        delays=[0,1,2,3],polynomial_scales=list(SCALES),variance_floor=VARIANCE_FLOOR,mass_floor=MASS_FLOOR,
        projection_scope='same packet, same lag, closed-form IQ-only moments; no fitting objective or persistent state',
        projection_formula='weight=abs(z)^2;mu=Ew[p];v=Ew[(p-mu)^2];a=Ew[(p-mu)^3]/max(v,1e-8)',
        polynomial_formula='z;4*z*(p-mu);16*z*((p-mu)^2-a*(p-mu)-v)',
        per_order_energy_whitening=False,new_trainable_parameters=0,original_position_power_readout_retained=True,
        packet_future_scope='full received256packet moments, same scope as RMS; not streaming causal projection',
        entire_raw_clipped_IQ_retained=True,whole_affine_phase_invariant=False,hardware_parameter_recovery=False,
        orthogonality_scope='within each same-packet lag/order input block when nondegenerate; not all12crosslag or classes')


class OrthogonalCVS(EnergyCVS):
    def __init__(self,variant):
        contract=orthopoly_contract(variant);super().__init__('energy_equivariant');self.orthopoly_variant=variant
        self.core.behavior[0]=OrthogonalInputBlock(self.core.behavior[0],contract['envelope_lag'])
    def contract(self):
        d=dict(super().contract(),**orthopoly_contract(self.orthopoly_variant))
        d['orthopoly_active']=isinstance(self.core.behavior[0],OrthogonalInputBlock)
        d['actual_envelope_lag']=getattr(self.core.behavior[0],'envelope_lag',None)
        return d
    @torch.no_grad()
    def input_diagnostics(self,x):
        block=self.core.behavior[0];captured=[]
        def capture(module,inputs,output):
            actual=inputs[0];z=actual[:,:,0];expected=packet_basis(z,block.envelope_lag)
            captured.append(dict(block='behavior.0.conv',packets=len(z),
                actual_envelope_lag=block.envelope_lag,input_formula_max_abs_error=float((actual-expected).abs().max()),
                input_abs_max=float(actual.abs().max()),**basis_diagnostics(z,block.envelope_lag)))
        hook=block.conv.register_forward_hook(capture)
        try:self.features(x)
        finally:hook.remove()
        return dict(active=isinstance(block,OrthogonalInputBlock),records=captured,
            scope='Actual conv input; packet-local IQ-only projection, no mutable state or TX coefficients')
    @torch.no_grad()
    def diagnostics(self,x):return dict(super().diagnostics(x),orthopoly_input=self.input_diagnostics(x))


def build(variant):return OrthogonalCVS(variant)
