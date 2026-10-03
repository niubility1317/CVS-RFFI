"""Within-frequency Hermitian relations alongside the complete absolute paths.

Ideal frequency-bin complex-gain properties belong to the relation statistic,
not arbitrary finite-window FIRs, the complete model, or recovered TX hardware.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn

from experiments.cvs_equivariant_identity.model import EquivariantCVS, InvariantReadout, behavior_basis
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS, neural_contract

VARIANTS = ('relation_packet_energy', 'relation_frequency_energy')
BASE_VARIANT = 'neural_residual_shallow'
BASE_PARAMETERS = 220987
NEW_PARAMETERS = 26744
FRAME_LENGTH = 64
FRAME_HOP = 32
POWER_FLOOR = 1e-6


class SpectralTemporalRelation(nn.Module):
    def __init__(self, per_frequency):
        super().__init__()
        if type(per_frequency) is not bool:
            raise ValueError('per_frequency must be a boolean')
        self.per_frequency = per_frequency
        self.frame_length, self.frame_hop, self.power_floor = FRAME_LENGTH, FRAME_HOP, POWER_FLOOR
        self.register_buffer('window', torch.hann_window(FRAME_LENGTH, periodic=True))
        self.mix_real = nn.Parameter(torch.randn(4, 7) / math.sqrt(14))
        self.mix_imag = nn.Parameter(torch.randn(4, 7) / math.sqrt(14))
        self.encoder = nn.Sequential(nn.Conv1d(32, 32, 3, padding=1), nn.GELU(),
                                     nn.Conv1d(32, 32, 3, padding=1), nn.GELU(),
                                     nn.AdaptiveAvgPool1d(4))
        self.project = nn.Linear(128, 160, bias=False)
        nn.init.zeros_(self.project.weight)

    def spectral(self, x):
        """Seven unpadded periodic-Hann frames; signed FFT frequency axis."""
        if x.ndim != 3 or x.shape[1:] != (2, 256):
            raise ValueError('Expected IQ[B,2,256]')
        frames = x.unfold(-1, self.frame_length, self.frame_hop) * self.window
        z = torch.complex(frames[:, 0], frames[:, 1])
        spectrum = torch.fft.fftshift(torch.fft.fft(z, n=self.frame_length, dim=-1), dim=-1)
        spectrum = spectrum / self.window.norm()
        return torch.stack((spectrum.real, spectrum.imag), 1).transpose(-1, -2)

    def mix_spectra(self, spectra):
        if spectra.ndim != 4 or spectra.shape[1:] != (2, 64, 7):
            raise ValueError('Expected paired spectra[B,2,64,7]')
        real, imag = spectra.unbind(1)
        return torch.stack((real @ self.mix_real.T - imag @ self.mix_imag.T,
                            real @ self.mix_imag.T + imag @ self.mix_real.T), 1)

    def energy(self, v):
        power = v.square().sum(1).sum(-1)
        return power if self.per_frequency else power.mean(-1, keepdim=True)

    def relations(self, v):
        if v.ndim != 4 or v.shape[1:] != (2, 64, 4):
            raise ValueError('Expected mixed spectra[B,2,64,4]')
        real, imag = v.unbind(1)
        numerator_real = real[..., :, None]*real[..., None, :] + imag[..., :, None]*imag[..., None, :]
        numerator_imag = imag[..., :, None]*real[..., None, :] - real[..., :, None]*imag[..., None, :]
        denominator = self.energy(v).clamp_min(self.power_floor)
        return torch.stack((numerator_real, numerator_imag), 1) / denominator[:, None, :, None, None]

    def statistics(self, x):
        return self.relations(self.mix_spectra(self.spectral(x)))

    def forward(self, x):
        # Channel order: sixteen row-major real entries, then sixteen imaginary.
        q = self.statistics(x).permute(0, 1, 3, 4, 2).reshape(len(x), 32, 64)
        return self.project(self.encoder(q).flatten(1))


class SpectralRelationCore(EquivariantCVS):
    def __init__(self, original, per_frequency):
        nn.Module.__init__(self)
        if list(original.parameters(recurse=False)) or list(original.buffers(recurse=False)):
            raise ValueError('Unexpected root-level scratch backbone state')
        self.variant = original.variant
        for name, module in original.named_children():
            self.add_module(name, module)
        self.spectral_relation = SpectralTemporalRelation(per_frequency)

    def features(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, 256):
            raise ValueError('Expected IQ[B,2,256]')
        b = self.id_backbone
        sinc = b._sinc_on_iq(x).reshape(len(x), 2, 24, 256)
        time = b.t_proj(self.readout(self.time(sinc)))
        physical = b.pa_proj(self.readout(self.behavior(behavior_basis(x))))
        spectral, rho, dac_stats, pa_stats = b._mirror_compressed_features(x, sinc_iq=None)
        f = b.f_pool(b.f3(b.f2(b.f1(b.freq_gate(spectral))))).squeeze(-1)
        frequency = b.f_proj(f) + self.spectral_relation(x)
        if b.use_stats_path and b.use_freq_stats and b.freq_stats_proj is not None:
            frequency = frequency + b.freq_stats_proj(dac_stats)
        if b.use_stats_path and b.pa_stats_proj is not None:
            physical = physical + .25*b.pa_stats_proj(pa_stats)
        parts = [time, frequency]
        if rho is not None:
            parts.append(rho if b.use_stats_path else torch.zeros_like(rho))
        base = b.fuse(torch.cat(parts, 1))
        return b.cls_head.components(base, physical)[2]


def relation_contract(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered spectral relation architecture')
    per_frequency = variant == VARIANTS[1]
    d = neural_contract(BASE_VARIANT)
    d.update(
        mode=variant, identity_core='spectral_relation_identity', base_variant=BASE_VARIANT,
        base_trainable_parameters=BASE_PARAMETERS, new_trainable_parameters=NEW_PARAMETERS,
        total_parameters=BASE_PARAMETERS+NEW_PARAMETERS,
        total_trainable_parameters=BASE_PARAMETERS+NEW_PARAMETERS,
        spectral_relation_active=True, spectral_relation_variant=variant,
        spectral_relation_input='original full received IQ packet [B,2,256]',
        spectral_frame_length=FRAME_LENGTH, spectral_hop=FRAME_HOP, spectral_frames=7,
        spectral_window='periodic Hann', spectral_center=False, spectral_fftshift=True,
        spectral_fft_normalization='divide by Hann window L2 norm',
        spectral_complex_fft=True, spectral_frequency_bins=64,
        spectral_mix_shape=[4, 7], spectral_mix_bias=False,
        spectral_mix_shared_across_frequency=True,
        spectral_mix_initialization='independent real/imag normal divided by sqrt(2*7)',
        relation_normalization='per_frequency_energy' if per_frequency else 'mean_frequency_energy_per_packet',
        relation_power_floor=POWER_FLOOR, relation_formula='v_i*conj(v_j)/clamp(energy,1e-6)',
        relation_shape=[2, 64, 4, 4], relation_channels=32,
        relation_channel_order='real row-major 4x4 then imaginary row-major 4x4',
        relation_encoder='Conv1d32to32k3pad1,GELU,Conv1d32to32k3pad1,GELU,AdaptiveAvgPool4',
        relation_exit='biasfree Linear128to160, zero initialization',
        relation_wiring='original b.f_proj(f)+relation(x), then original frequency statistics and fusion',
        original_time_behavior_frequency_stats_paths_retained=True,
        relation_cross_packet_state=False, relation_inference_fit=False,
        relation_theoretical_hermitian_psd=True, relation_theoretical_rank_upper_bound=1,
        relation_theoretical_per_frequency_trace_upper_bound=1 if per_frequency else 64,
        relation_theoretical_per_frequency_frobenius_upper_bound=1 if per_frequency else 64,
        relation_ideal_gain_property=('nonzero arbitrary complex gain per frequency' if per_frequency
                                      else 'common packet gain and independent per-frequency phase'),
        relation_ideal_gain_domain='ideal S_f -> h_f*S_f, before/after denominators above floor; statistic only',
        finite_window_fir_exact_invariance=False, whole_model_channel_invariant=False,
        tx_linear_response_preservation_by_relation=False,
        initial_function='exact own scratch neural_residual_shallow; isolated new initialization RNG',
        spectral_relation_loss='unchanged single cross_entropy', new_training_strategy=False,
        arbitrary_channel_rx_invariant=False, hardware_parameter_recovery=False,
        actual_spectral_mix_shapes=[[4, 7], [4, 7]], actual_spectral_window_matches=True,
        actual_relation_normalization='per_frequency_energy' if per_frequency else 'mean_frequency_energy_per_packet',
        actual_relation_encoder_matches=True, actual_relation_project_shape=[160, 128],
        actual_relation_wiring_matches=True,
        interpretation='Packet spectral temporal association alongside retained absolute RFF paths; not channel recovery or TX/RX separation',
    )
    return d


@contextmanager
def diagnostic_mode(model, x):
    flags = [(module, module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=[x.get_device()] if x.is_cuda else []):
            model.eval()
            yield
    finally:
        for module, training in flags:
            module.training = training


def gradient_norm(parameters):
    grads = [p.grad.norm() for p in parameters if p.grad is not None]
    return float(torch.stack(grads).norm()) if grads else None


class SpectralRelationCVS(NeuralResidualCVS):
    def __init__(self, variant):
        relation_contract(variant)
        super().__init__(BASE_VARIANT)
        self.spectral_relation_variant = variant
        with torch.random.fork_rng(devices=[]):
            self.core = SpectralRelationCore(self.core, variant == VARIANTS[1])

    def spectral_relation_parameters(self):
        return list(self.core.spectral_relation.parameters())

    def contract(self):
        d = relation_contract(self.spectral_relation_variant)
        r = self.core.spectral_relation
        encoder = getattr(r, 'encoder', None)
        encoder_matches = (isinstance(encoder, nn.Sequential) and len(encoder) == 5
                           and all(type(encoder[i]) is nn.Conv1d
                                   and (encoder[i].in_channels, encoder[i].out_channels,
                                        encoder[i].kernel_size, encoder[i].stride, encoder[i].padding,
                                        encoder[i].dilation, encoder[i].groups, encoder[i].padding_mode)
                                   == (32, 32, (3,), (1,), (1,), (1,), 1, 'zeros')
                                   and tuple(encoder[i].weight.shape) == (32, 32, 3)
                                   and encoder[i].bias is not None and tuple(encoder[i].bias.shape) == (32,)
                                   for i in (0, 2))
                           and all(type(encoder[i]) is nn.GELU and encoder[i].approximate == 'none' for i in (1, 3))
                           and type(encoder[4]) is nn.AdaptiveAvgPool1d and encoder[4].output_size == 4)
        mix_shapes = [list(getattr(r, name).shape) if hasattr(r, name) else None for name in ('mix_real', 'mix_imag')]
        window = getattr(r, 'window', None)
        expected_window = torch.hann_window(64, periodic=True)
        window_matches = (isinstance(window, torch.Tensor) and window.shape == (64,)
                          and torch.equal(window, expected_window.to(window)) and not window.requires_grad)
        project = getattr(r, 'project', None)
        project_matches = (type(project) is nn.Linear and project.in_features == 128
                           and project.out_features == 160 and project.bias is None
                           and tuple(project.weight.shape) == (160, 128))
        wiring_matches = (type(self.core) is SpectralRelationCore
                          and getattr(self.core.features, '__func__', None) is SpectralRelationCore.features
                          and getattr(self.features, '__func__', None) is NeuralResidualCVS.features
                          and isinstance(self.core.readout, InvariantReadout)
                          and self.variant == 'energy_equivariant')
        normalization = ('per_frequency_energy' if getattr(r, 'per_frequency', None) is True
                         else 'mean_frequency_energy_per_packet' if getattr(r, 'per_frequency', None) is False else None)
        methods_match = all(getattr(getattr(r, name, None), '__func__', None) is getattr(SpectralTemporalRelation, name)
                            for name in ('spectral', 'mix_spectra', 'energy', 'relations', 'statistics', 'forward'))
        actual = (type(r) is SpectralTemporalRelation and mix_shapes == [[4, 7], [4, 7]]
                  and (r.frame_length, r.frame_hop, r.power_floor) == (64, 32, POWER_FLOOR)
                  and normalization == d['relation_normalization'] and methods_match
                  and encoder_matches and window_matches and project_matches and wiring_matches)
        parameters = self.spectral_relation_parameters()
        ids = {id(p) for p in parameters}
        d.update(spectral_relation_active=actual,
                 neural_residual_active=super().contract()['neural_residual_active'],
                 actual_spectral_mix_shapes=mix_shapes, actual_spectral_window_matches=window_matches,
                 actual_relation_normalization=normalization, actual_relation_encoder_matches=encoder_matches,
                 actual_relation_project_shape=list(project.weight.shape) if hasattr(project, 'weight') else None,
                 actual_relation_wiring_matches=wiring_matches,
                 new_trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),
                 base_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in ids),
                 total_parameters=sum(p.numel() for p in self.parameters()),
                 total_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))
        return d

    @torch.no_grad()
    def spectral_relation_diagnostics(self, x):
        records, frequency = [], []
        r = self.core.spectral_relation

        def capture(module, inputs, output):
            if len(frequency) != 1:
                raise ValueError('Expected original frequency projection before relation injection')
            v = module.mix_spectra(module.spectral(inputs[0]))
            q = module.relations(v)
            norm = q.square().sum((1, 3, 4)).sqrt()
            trace = q[:, 0].diagonal(dim1=-2, dim2=-1).sum(-1)
            floor = (module.energy(v) < module.power_floor).expand(-1, 64)
            relative = output.norm(dim=1) / frequency[0].norm(dim=1).clamp_min(1e-12)
            records.append(dict(block='frequency.spectral_relation', packets=len(output),
                relative_output_change_mean=float(relative.mean()), relative_output_change_max=float(relative.max()),
                floor_fraction=float(floor.float().mean()), floor_fraction_by_frequency=floor.float().mean(0).tolist(),
                relation_norm_mean=float(norm.mean()), relation_norm_max=float(norm.max()),
                relation_trace_mean=float(trace.mean()), relation_trace_max=float(trace.max()),
                relation_norm_by_frequency=norm.mean(0).tolist(), relation_trace_by_frequency=trace.mean(0).tolist(),
                mix_norm=float(torch.stack((r.mix_real.norm(), r.mix_imag.norm())).norm()),
                projection_norm=float(r.project.weight.norm()),
                mix_gradient_norm=gradient_norm([r.mix_real, r.mix_imag]),
                projection_gradient_norm=gradient_norm(r.project.parameters()),
                encoder_gradient_norm=gradient_norm(r.encoder.parameters()),
                per_frequency=module.per_frequency, frames=7, frequency_bins=64, output_dimension=160))

        hooks = [self.core.id_backbone.f_proj.register_forward_hook(lambda module, inputs, output: frequency.append(output)),
                 r.register_forward_hook(capture)]
        try:
            with diagnostic_mode(self, x):
                self.features(x)
        finally:
            for hook in hooks:
                hook.remove()
        return dict(active=self.contract()['spectral_relation_active'], variant=self.spectral_relation_variant,
                    records=records, scope='All provided packets, complete 256 samples/7 frames/64 frequencies per packet; '
                    'not whole-source evidence unless all source packets were provided; last CE gradients, no updates or fitting')

    @torch.no_grad()
    def diagnostics(self, x):
        with diagnostic_mode(self, x):
            return dict(super().diagnostics(x), spectral_relation=self.spectral_relation_diagnostics(x))


def build(variant):
    return SpectralRelationCVS(variant)
