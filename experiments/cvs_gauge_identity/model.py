"""Information-preserving packet phase gauge; no identified TX parameters."""
import torch
from torch import nn
from experiments.cvs_residual_identity.model import build as build_residual

VARIANTS=('gauge_peak','gauge_coherent')


def gauge_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered gauge variant')
    return dict(mode=variant,epsilon=1e-6,near_peak_relative_tolerance=1e-5,
                coherent_window=[80,160],period_samples=20,cycles=4,
                degenerate_reference_relative_power=1e-6,
                whole_model_global_phase_invariant_real_arithmetic=True,
                removes_linear_phase=False,original_temporal_samples=256,
                input='received_equalized1_unit_rms_IQ',sample_rate_hz=25000000,
                learned_frontend_parameters=0,original_waveform_preserved_up_to_packet_complex_scalar=True,
                interpretation='Packet carrier-phase gauge;not TX hardware identification or arbitrary channel/RX invariance')


def near_peak_index(power,tolerance):
    maximum=power.max(-1,keepdim=True).values
    # First near-maximum gives deterministic handling of repeated equal peaks.
    return (power>=maximum*(1-tolerance)).to(torch.int64).argmax(-1)


class PacketPhaseGauge(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.epsilon=1e-6;self.tie_tolerance=1e-5;self.degenerate_relative=1e-6
        self.window=(80,160);self.period=20
        gauge_contract(variant)

    def contract(self):
        d=gauge_contract(self.variant)
        d.update(epsilon=self.epsilon,near_peak_relative_tolerance=self.tie_tolerance,
                 coherent_window=list(self.window),period_samples=self.period,
                 degenerate_reference_relative_power=self.degenerate_relative)
        return d

    def reference(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected received IQ[B,2,256]')
        r,i=x[:,0],x[:,1];power=r.square()+i.square()
        index=near_peak_index(power,self.tie_tolerance)
        rr=r.gather(1,index[:,None]).squeeze(1);ri=i.gather(1,index[:,None]).squeeze(1)
        fallback=torch.zeros(len(x),dtype=torch.bool,device=x.device);coherence=None
        if self.variant=='gauge_coherent':
            lo,hi=self.window
            ar=r[:,lo:hi].reshape(-1,4,self.period);ai=i[:,lo:hi].reshape(-1,4,self.period)
            cr=(ar[:,1:]*ar[:,:-1]+ai[:,1:]*ai[:,:-1]).sum((1,2))
            ci=(ai[:,1:]*ar[:,:-1]-ar[:,1:]*ai[:,:-1]).sum((1,2))
            norm=(cr.square()+ci.square()+self.epsilon**2).sqrt()
            ur,ui=cr/norm,-ci/norm  # Conjugate of the cycle-to-cycle phase.
            vr,vi=torch.ones_like(ur),torch.zeros_like(ui)
            aligned_r=[];aligned_i=[]
            for cycle in range(4):
                aligned_r.append(ar[:,cycle]*vr[:,None]-ai[:,cycle]*vi[:,None])
                aligned_i.append(ai[:,cycle]*vr[:,None]+ar[:,cycle]*vi[:,None])
                vr,vi=vr*ur-vi*ui,vi*ur+vr*ui
            mr=torch.stack(aligned_r,1).mean(1);mi=torch.stack(aligned_i,1).mean(1)
            mp=mr.square()+mi.square();pick=near_peak_index(mp,self.tie_tolerance)
            qr=mr.gather(1,pick[:,None]).squeeze(1);qi=mi.gather(1,pick[:,None]).squeeze(1)
            fallback=(qr.square()+qi.square())<=self.degenerate_relative*power.max(-1).values
            rr=torch.where(fallback,rr,qr);ri=torch.where(fallback,ri,qi)
            e1=(ar[:,1:].square()+ai[:,1:].square()).sum((1,2))
            e0=(ar[:,:-1].square()+ai[:,:-1].square()).sum((1,2))
            coherence=(cr.square()+ci.square()).sqrt()/(e1*e0+self.epsilon**2).sqrt()
        return rr,ri,index,fallback,coherence

    def forward(self,x):
        rr,ri,_,_,_=self.reference(x)
        denominator=(rr.square()+ri.square()+self.epsilon).sqrt()
        qr,qi=rr/denominator,ri/denominator
        r,i=x[:,0],x[:,1]
        return torch.stack([r*qr[:,None]+i*qi[:,None],i*qr[:,None]-r*qi[:,None]],1)

    @torch.no_grad()
    def diagnostics(self,x):
        rr,ri,_,fallback,coherence=self.reference(x)
        return dict(reference_power_mean=float((rr.square()+ri.square()).mean()),
                    fallback_fraction=float(fallback.float().mean()),
                    coherent_correlation_mean=float(coherence.mean()) if coherence is not None else None,
                    scope='Last source batch only;measured without model state update')


class GaugeCVS(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;self.gauge=PacketPhaseGauge(variant)
        self.core=build_residual('residual_fusion')
    def contract(self):return self.gauge.contract()
    def forward(self,x):return self.core(self.gauge(x))
    def features(self,x):return self.core.features(self.gauge(x))
    def classify_features(self,features):return self.core.id_backbone.cls_head.classify(features)


def build(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered gauge variant')
    return GaugeCVS(variant)


@torch.no_grad()
def frozen_synthetic_diagnostics(model):
    """Hand-set period20 signal, never a formal training or query sample."""
    import torch.nn.functional as F
    model.eval();parameter=next(model.parameters());n=torch.arange(256,device=parameter.device,dtype=parameter.dtype)
    k=n.remainder(20)
    amp=1+.3*torch.cos(2*torch.pi*k/20)+.07*torch.sin(4*torch.pi*k/20)
    phase=.19*n+.1*torch.sin(2*torch.pi*k/20)
    base=torch.stack([amp*phase.cos(),amp*phase.sin()],0)[None]
    normalize=lambda x:x/(x.square().sum(1).mean(-1,keepdim=True)+1e-12).sqrt().unsqueeze(1)
    base=normalize(base)
    def rotate(x,p):
        r,i=x[:,0],x[:,1]
        return torch.stack([r*p.cos()-i*p.sin(),r*p.sin()+i*p.cos()],1)
    power=base.square().sum(1,keepdim=True)
    interventions=dict(handset_am_am=normalize(base/(1+.12*power)),handset_am_pm=rotate(base,.17*power[:,0]),
        rx_iq_counterexample=normalize(base*base.new_tensor([1.08,.92])[None,:,None]),
        multipath_counterexample=normalize(base+.2*F.pad(base[...,:-2],(2,0))))
    baseline=model(base);embedding=F.normalize(model.features(base),dim=1,eps=1e-4)
    distances={name:float((F.normalize(model.features(x),dim=1,eps=1e-4)-embedding).norm()) for name,x in interventions.items()}
    return dict(phase_logit_max_abs_error=float((model(rotate(base,n.new_tensor(.731)))-baseline).abs().max()),
        affine_logit_max_abs_error=float((model(rotate(base,.4+.031*n))-baseline).abs().max()),
        affine_invariance_claimed=False,unit_embedding_distances=distances,gauge_reference=model.gauge.diagnostics(base),
        synthetic_only=True,training_augmentation=False,target_access=False,
        input_rms_normalized=True,synthetic_signal='Hand-set period20 amplitude/phase signal;not standard L-STF coefficients',
        physical_parameters=dict(positive_amp_compression=.12,amp_phase_beta=.17,rx_iq_scale=[1.08,.92],channel_delay_samples=2,channel_tap=.2,constant_phase_radians=.731,affine_phase_slope_radians_per_sample=.031),
        claim='Frozen transform/sensitivity diagnosis only;not TX parameter identification or synthetic identity accuracy.')
