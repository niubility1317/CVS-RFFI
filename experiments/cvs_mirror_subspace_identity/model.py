"""Qualified mirror-pair mixing invariance alongside retained absolute RFF paths."""
import math

import torch
from torch import nn

from experiments.cvs_equivariant_identity.model import EquivariantCVS, InvariantReadout, behavior_basis
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS, neural_contract
from experiments.cvs_spectral_relation_identity.model import diagnostic_mode, gradient_norm

VARIANTS = ('mirror_energy', 'mirror_subspace')
BASE_VARIANT = 'neural_residual_shallow'
BASE_PARAMETERS, NEW_PARAMETERS = 220987, 26744
FRAME_LENGTH, FRAME_HOP, FREQUENCY_PAIRS = 64, 32, 31
POWER_FLOOR, RELATIVE_POWER_FLOOR, DETERMINANT_FLOOR = 1e-6, 1/64, 1e-3


class MirrorPairRelation(nn.Module):
    def __init__(self, use_subspace):
        super().__init__()
        if type(use_subspace) is not bool:
            raise ValueError('use_subspace must be boolean')
        self.use_subspace = use_subspace
        self.frame_length, self.frame_hop = FRAME_LENGTH, FRAME_HOP
        self.power_floor, self.relative_power_floor = POWER_FLOOR, RELATIVE_POWER_FLOOR
        self.determinant_floor = DETERMINANT_FLOOR
        self.register_buffer('window', torch.hann_window(FRAME_LENGTH, periodic=True))
        self.mix_real = nn.Parameter(torch.randn(4, 7)/math.sqrt(14))
        self.mix_imag = nn.Parameter(torch.randn(4, 7)/math.sqrt(14))
        self.encoder = nn.Sequential(nn.Conv1d(32,32,3,padding=1),nn.GELU(),
                                     nn.Conv1d(32,32,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool1d(4))
        self.project = nn.Linear(128,160,bias=False)
        nn.init.zeros_(self.project.weight)

    def spectral(self, x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):
            raise ValueError('Expected IQ[B,2,256]')
        frames=x.unfold(-1,self.frame_length,self.frame_hop)*self.window
        spectrum=torch.fft.fftshift(torch.fft.fft(torch.complex(frames[:,0],frames[:,1]),dim=-1),dim=-1)
        spectrum=spectrum/self.window.norm()
        return torch.stack((spectrum.real,spectrum.imag),1).transpose(-1,-2)

    def pair_spectra(self, spectra):
        if spectra.ndim!=4 or spectra.shape[1:]!=(2,64,7):
            raise ValueError('Expected paired spectra[B,2,64,7]')
        real,imag=spectra.unbind(1)
        # +1..+31 pair with conjugate(-1..-31); no DC or Nyquist.
        r=torch.stack((real[:,33:64],real[:,1:32].flip(1)),dim=-2)
        i=torch.stack((imag[:,33:64],-imag[:,1:32].flip(1)),dim=-2)
        return torch.stack((r,i),1)

    def mix_spectra(self, spectra):
        real,imag=self.pair_spectra(spectra).unbind(1)
        # Both mirror rows use the same A, not conjugate(A).
        return torch.stack((real@self.mix_real.T-imag@self.mix_imag.T,
                            real@self.mix_imag.T+imag@self.mix_real.T),1)

    def energy(self, v):
        return v.square().sum((1,3,4))

    def energy_floor(self, v):
        return (self.energy(v).mean(-1,keepdim=True)*self.relative_power_floor).clamp_min(self.power_floor)

    def floor_active(self, v):
        return self.energy(v)<self.energy_floor(v)

    def components(self, v):
        if v.ndim!=5 or v.shape[1:]!=(2,31,2,4):
            raise ValueError('Expected mixed mirror rows[B,2,31,2,4]')
        energy=self.energy(v)
        denominator=torch.maximum(energy,self.energy_floor(v))
        normalized=v/denominator.sqrt()[:,None,:,None,None]
        real,imag=normalized.unbind(1)
        a=(real[...,0,:].square()+imag[...,0,:].square()).sum(-1)
        b=(real[...,1,:].square()+imag[...,1,:].square()).sum(-1)
        cr=(real[...,0,:]*real[...,1,:]+imag[...,0,:]*imag[...,1,:]).sum(-1)
        ci=(imag[...,0,:]*real[...,1,:]-real[...,0,:]*imag[...,1,:]).sum(-1)
        raw_det=a*b-cr.square()-ci.square()
        # Cauchy-Binet avoids subtracting nearly equal energies at rank one.
        ar,ai=real[...,0,:],imag[...,0,:]
        br,bi=real[...,1,:],imag[...,1,:]
        wr=ar[..., :,None]*br[...,None,:]-ai[..., :,None]*bi[...,None,:]-br[..., :,None]*ar[...,None,:]+bi[..., :,None]*ai[...,None,:]
        wi=ar[..., :,None]*bi[...,None,:]+ai[..., :,None]*br[...,None,:]-br[..., :,None]*ai[...,None,:]-bi[..., :,None]*ar[...,None,:]
        determinant=(wr.square()+wi.square()).sum((-2,-1))*.5
        return dict(normalized=normalized,a=a,b=b,cr=cr,ci=ci,raw_determinant=raw_det,
                    wedge_real=wr,wedge_imag=wi,
                    determinant=determinant,alpha=energy/denominator,energy=energy,
                    denominator=denominator,energy_floor_active=energy<self.energy_floor(v),
                    determinant_floor_active=determinant<self.determinant_floor)

    def relations(self, v):
        c=self.components(v)
        real,imag=c['normalized'].unbind(1)
        if self.use_subspace:
            # conj(W) W^T == N^H adj(N N^H) N, W_ij=N_0i N_1j-N_1i N_0j.
            # This Gram evaluation preserves nonnegative trace without clipping.
            wr,wi=c['wedge_real'],c['wedge_imag']
            qr=wr@wr.transpose(-2,-1)+wi@wi.transpose(-2,-1)
            qi=wr@wi.transpose(-2,-1)-wi@wr.transpose(-2,-1)
            scale=c['alpha']/(2*c['determinant'].clamp_min(self.determinant_floor))
        else:
            qr=real.transpose(-2,-1)@real+imag.transpose(-2,-1)@imag
            qi=real.transpose(-2,-1)@imag-imag.transpose(-2,-1)@real
            scale=torch.ones_like(c['alpha'])
        qr=(qr+qr.transpose(-2,-1))*.5
        qi=(qi-qi.transpose(-2,-1))*.5
        return torch.stack((qr,qi),1)*scale[:,None,:,None,None]

    def statistics(self, x):
        return self.relations(self.mix_spectra(self.spectral(x)))

    def forward(self, x):
        q=self.statistics(x).permute(0,1,3,4,2).reshape(len(x),32,31)
        return self.project(self.encoder(q).flatten(1))


class MirrorSubspaceCore(EquivariantCVS):
    def __init__(self, original, use_subspace):
        nn.Module.__init__(self)
        if list(original.parameters(recurse=False)) or list(original.buffers(recurse=False)):
            raise ValueError('Unexpected root-level scratch backbone state')
        self.variant=original.variant
        for name,module in original.named_children():self.add_module(name,module)
        self.mirror_relation=MirrorPairRelation(use_subspace)

    def features(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected IQ[B,2,256]')
        b=self.id_backbone
        sinc=b._sinc_on_iq(x).reshape(len(x),2,24,256)
        time=b.t_proj(self.readout(self.time(sinc)))
        physical=b.pa_proj(self.readout(self.behavior(behavior_basis(x))))
        spectral,rho,dac_stats,pa_stats=b._mirror_compressed_features(x,sinc_iq=None)
        f=b.f_pool(b.f3(b.f2(b.f1(b.freq_gate(spectral))))).squeeze(-1)
        frequency=b.f_proj(f)+self.mirror_relation(x)
        if b.use_stats_path and b.use_freq_stats and b.freq_stats_proj is not None:
            frequency=frequency+b.freq_stats_proj(dac_stats)
        if b.use_stats_path and b.pa_stats_proj is not None:physical=physical+.25*b.pa_stats_proj(pa_stats)
        parts=[time,frequency]
        if rho is not None:parts.append(rho if b.use_stats_path else torch.zeros_like(rho))
        return b.cls_head.components(b.fuse(torch.cat(parts,1)),physical)[2]


def relation_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered mirror architecture')
    subspace=variant==VARIANTS[1]
    d=neural_contract(BASE_VARIANT)
    d.update(mode=variant,identity_core='mirror_subspace_identity',base_variant=BASE_VARIANT,
        base_trainable_parameters=BASE_PARAMETERS,new_trainable_parameters=NEW_PARAMETERS,
        total_parameters=BASE_PARAMETERS+NEW_PARAMETERS,total_trainable_parameters=BASE_PARAMETERS+NEW_PARAMETERS,
        mirror_relation_active=True,mirror_relation_variant=variant,
        mirror_relation_input='full received IQ packet [B,2,256]',
        spectral_frame_length=64,spectral_hop=32,spectral_frames=7,spectral_window='periodic Hann',
        spectral_center=False,spectral_fftshift=True,spectral_fft_normalization='divide by Hann window L2 norm',
        spectral_frequency_bins=64,mirror_frequency_pairs=31,
        mirror_positive_indices=list(range(33,64)),mirror_negative_indices=list(range(31,0,-1)),
        mirror_second_row='conjugate negative-frequency signal before applying the same complex A',
        mirror_excluded_bins=['DC','Nyquist'],spectral_mix_shape=[4,7],spectral_mix_bias=False,
        spectral_mix_shared_across_frequency_and_rows=True,
        spectral_mix_initialization='independent real/imag normal divided by sqrt(14)',
        relation_normalization='determinant_damped_subspace' if subspace else 'pair_energy',
        relation_power_floor=POWER_FLOOR,relation_relative_power_floor=RELATIVE_POWER_FLOOR,
        relation_determinant_floor=DETERMINANT_FLOOR,
        relation_formula=('alpha*N^H*adj(N*N^H)*N/(2*max(det(N*N^H),1e-3))' if subspace else 'N^H*N'),
        relation_numerical_evaluation='Cauchy-Binet determinant and exterior-product Gram; no trace clipping',
        relation_normalized_input='N=Z/sqrt(max(pair_energy,mean_pair_energy/64,1e-6));alpha=trace(N*N^H)',
        relation_shape=[2,31,4,4],relation_channels=32,
        relation_channel_order='real row-major 4x4 then imaginary row-major 4x4',
        relation_encoder='Conv1d32to32k3pad1,GELU,Conv1d32to32k3pad1,GELU,AdaptiveAvgPool4',
        relation_exit='biasfree Linear128to160, zero initialization',
        relation_wiring='original b.f_proj(f)+mirror_relation(x), then original frequency statistics and fusion',
        original_time_behavior_frequency_stats_paths_retained=True,
        relation_cross_packet_state=False,relation_inference_fit=False,
        relation_theoretical_hermitian_psd=True,relation_theoretical_rank_upper_bound=2,
        relation_theoretical_trace_upper_bound=1.,relation_theoretical_frobenius_upper_bound=1/math.sqrt(2) if subspace else 1.,
        relation_ideal_mixing_property='invertible complex 2x2 left mixing' if subspace else 'common scaling and unitary 2x2 left mixing',
        relation_ideal_mixing_domain='before/after energy floors inactive; subspace also requires det(N*N^H)>=1e-3; statistic only',
        relation_weak_or_rank_deficient_invariance=False,
        finite_window_fir_exact_invariance=False,whole_model_channel_invariant=False,
        arbitrary_channel_rx_invariant=False,hardware_parameter_recovery=False,
        tx_linear_iq_fingerprint_preservation_by_relation=False,
        initial_function='exact own scratch neural_residual_shallow; isolated new initialization RNG',
        mirror_relation_loss='unchanged single cross_entropy',new_training_strategy=False,
        actual_spectral_mix_shapes=[[4,7],[4,7]],actual_spectral_window_matches=True,
        actual_relation_normalization='determinant_damped_subspace' if subspace else 'pair_energy',
        actual_relation_relative_power_floor=RELATIVE_POWER_FLOOR,actual_relation_determinant_floor=DETERMINANT_FLOOR,
        actual_relation_encoder_matches=True,actual_relation_project_shape=[160,128],actual_relation_wiring_matches=True,
        interpretation='Mirror-pair row-space information alongside retained absolute RFF paths; no channel estimation or TX/RX recovery')
    return d


class MirrorSubspaceCVS(NeuralResidualCVS):
    def __init__(self,variant):
        relation_contract(variant)
        super().__init__(BASE_VARIANT)
        self.mirror_relation_variant=variant
        with torch.random.fork_rng(devices=[]):self.core=MirrorSubspaceCore(self.core,variant==VARIANTS[1])

    def mirror_relation_parameters(self):return list(self.core.mirror_relation.parameters())

    def contract(self):
        d=relation_contract(self.mirror_relation_variant);r=self.core.mirror_relation
        e=getattr(r,'encoder',None)
        encoder_matches=(type(e) is nn.Sequential and len(e)==5 and all(type(e[i]) is nn.Conv1d
            and (e[i].in_channels,e[i].out_channels,e[i].kernel_size,e[i].stride,e[i].padding,e[i].dilation,e[i].groups,e[i].padding_mode)
            ==(32,32,(3,),(1,),(1,),(1,),1,'zeros') and tuple(e[i].weight.shape)==(32,32,3)
            and e[i].bias is not None and tuple(e[i].bias.shape)==(32,) for i in (0,2))
            and all(type(e[i]) is nn.GELU and e[i].approximate=='none' for i in (1,3))
            and type(e[4]) is nn.AdaptiveAvgPool1d and e[4].output_size==4)
        shapes=[list(getattr(r,k).shape) if hasattr(r,k) else None for k in ('mix_real','mix_imag')]
        window=getattr(r,'window',None)
        window_matches=(isinstance(window,torch.Tensor) and window.shape==(64,) and not window.requires_grad
                        and torch.equal(window,torch.hann_window(64,periodic=True).to(window)))
        project=getattr(r,'project',None)
        project_matches=(type(project) is nn.Linear and project.in_features==128 and project.out_features==160
                         and project.bias is None and tuple(project.weight.shape)==(160,128))
        wiring_matches=(type(self.core) is MirrorSubspaceCore
            and getattr(self.core.features,'__func__',None) is MirrorSubspaceCore.features
            and getattr(self.features,'__func__',None) is NeuralResidualCVS.features
            and isinstance(self.core.readout,InvariantReadout) and self.variant=='energy_equivariant')
        normalization=('determinant_damped_subspace' if getattr(r,'use_subspace',None) is True
                       else 'pair_energy' if getattr(r,'use_subspace',None) is False else None)
        methods=all(getattr(getattr(r,k,None),'__func__',None) is getattr(MirrorPairRelation,k)
                    for k in ('spectral','pair_spectra','mix_spectra','energy','energy_floor','floor_active','components','relations','statistics','forward'))
        active=(type(r) is MirrorPairRelation and shapes==[[4,7],[4,7]]
            and (r.frame_length,r.frame_hop,r.power_floor)==(64,32,POWER_FLOOR)
            and r.relative_power_floor==RELATIVE_POWER_FLOOR and r.determinant_floor==DETERMINANT_FLOOR
            and normalization==d['relation_normalization'] and methods and encoder_matches and window_matches
            and project_matches and wiring_matches)
        params=self.mirror_relation_parameters();ids={id(p) for p in params}
        d.update(mirror_relation_active=active,neural_residual_active=super().contract()['neural_residual_active'],
            actual_spectral_mix_shapes=shapes,actual_spectral_window_matches=window_matches,
            actual_relation_normalization=normalization,actual_relation_encoder_matches=encoder_matches,
            actual_relation_relative_power_floor=getattr(r,'relative_power_floor',None),
            actual_relation_determinant_floor=getattr(r,'determinant_floor',None),
            actual_relation_project_shape=list(project.weight.shape) if hasattr(project,'weight') else None,
            actual_relation_wiring_matches=wiring_matches,
            new_trainable_parameters=sum(p.numel() for p in params if p.requires_grad),
            base_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in ids),
            total_parameters=sum(p.numel() for p in self.parameters()),total_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))
        return d

    @torch.no_grad()
    def mirror_relation_diagnostics(self,x):
        records,frequency=[],[];r=self.core.mirror_relation
        def capture(module,inputs,output):
            if len(frequency)!=1:raise ValueError('Original frequency projection must precede mirror injection')
            v=module.mix_spectra(module.spectral(inputs[0]));q=module.relations(v);c=module.components(v)
            norm=q.square().sum((1,3,4)).sqrt();trace=q[:,0].diagonal(dim1=-2,dim2=-1).sum(-1)
            relative=output.norm(dim=1)/frequency[0].norm(dim=1).clamp_min(1e-12)
            floor=c['energy_floor_active'].float();det_floor=c['determinant_floor_active'].float()
            records.append(dict(block='frequency.mirror_relation',packets=len(output),
                relative_output_change_mean=float(relative.mean()),relative_output_change_max=float(relative.max()),
                floor_fraction=float(floor.mean()),floor_fraction_by_frequency=floor.mean(0).tolist(),
                determinant_floor_fraction=float(det_floor.mean()),determinant_floor_fraction_by_frequency=det_floor.mean(0).tolist(),
                determinant_mean=float(c['determinant'].mean()),determinant_minimum=float(c['determinant'].min()),
                alpha_mean=float(c['alpha'].mean()),alpha_minimum=float(c['alpha'].min()),
                relation_norm_mean=float(norm.mean()),relation_norm_max=float(norm.max()),
                relation_trace_mean=float(trace.mean()),relation_trace_max=float(trace.max()),
                relation_norm_by_frequency=norm.mean(0).tolist(),relation_trace_by_frequency=trace.mean(0).tolist(),
                mix_norm=float(torch.stack((r.mix_real.norm(),r.mix_imag.norm())).norm()),projection_norm=float(r.project.weight.norm()),
                mix_gradient_norm=gradient_norm([r.mix_real,r.mix_imag]),projection_gradient_norm=gradient_norm(r.project.parameters()),
                encoder_gradient_norm=gradient_norm(r.encoder.parameters()),use_subspace=module.use_subspace,
                frames=7,frequency_pairs=31,output_dimension=160))
        hooks=[self.core.id_backbone.f_proj.register_forward_hook(lambda module,inputs,output:frequency.append(output)),
               r.register_forward_hook(capture)]
        try:
            with diagnostic_mode(self,x):self.features(x)
        finally:
            for hook in hooks:hook.remove()
        return dict(active=self.contract()['mirror_relation_active'],variant=self.mirror_relation_variant,records=records,
            scope='All supplied packets/7 frames/31 mirror pairs; source coverage only when all V supplied; last measured CE gradients, no updates')

    @torch.no_grad()
    def diagnostics(self,x):
        with diagnostic_mode(self,x):return dict(super().diagnostics(x),mirror_relation=self.mirror_relation_diagnostics(x))


def build(variant):return MirrorSubspaceCVS(variant)
