"""Matched two- and three-prior W24 recovery on unlabeled source IQ."""
import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
import importlib
import os
import re
from pathlib import Path, PurePosixPath
import subprocess
from time import monotonic, sleep

if __package__ is None:
    from experiments import pamf_lora_main
    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))


RUN = '20261126-phase1-lora-w24-third-prior-s2026112991-r01'
ARMS = ('two', 'three')
STEPS = 1800
MODES = ('direct', 'all_half', 'two_half', 'three_half', 'two_full', 'three_full')
CONTRASTS = (('three_half', 'two_half'), ('three_full', 'two_full'), ('three_half', 'direct'))
ROUTES = ('full_native', 'native_residual_w24_v1')
ISOLATION = ('F_reads', 'TX_reads', 'V_reads', 'target_reads', 'practical_reads', 'trace_reads')
DIRECTION_RUN = '20261126-phase1-lora-w24-first-direction-s2026112991-r01'
DIRECTION_ARMS = ('direction_control', 'direction_first')
READOUT_RUN = '20261126-phase1-lora-w24-trained-readout-s2026112991-r01'
READOUT_ARMS = ('readout_control', 'readout_adaptive')
ROBUST_RUN = '20261126-phase1-lora-w24-robust-map-s2026112991-r01'
ROBUST_ARMS = ('robust_control', 'robust_map')
RADIUS_RUN = '20261126-phase1-lora-w24-final-radius-s2026112991-r01'
RADIUS_ARMS = ('radius_control', 'radius_adaptive')
PEER_RUN = '20261126-phase1-lora-w24-peer-innovation-s2026112991-r01'
PEER_ARMS = ('innovation_local', 'innovation_peer')
BIAS_RUN = '20261126-phase1-lora-w24-coherent-error-s2026112991-r01'
BIAS_ARMS = ('coherent_control', 'coherent_error')
BAND_TUNE_RUN = '20261126-phase1-lora-w24-band-direction-tune-s2026112991-r01'
BAND_TUNE_ARMS = ('band_tune_control', 'band_tune_direction')
BAND_ADAPTER_RUN = '20261126-phase1-lora-w24-band-adapter-s2026112991-r01'
BAND_ADAPTER_ARMS = ('band_adapter_time', 'band_adapter_frequency')
BAND_REFINE_RUN = '20261126-phase1-lora-w24-band-polar-refine-s2026112991-r02'
BAND_REFINE_ARMS = ('band_cartesian', 'band_polar')
WHITE_RUN = '20261126-phase1-lora-w24-white-corruption-s2026112991-r01'
WHITE_ARMS = ('archive_noise', 'stationary_noise')
BAND_PRECISION_RUN = '20261126-phase1-lora-w24-band-precision-s2026112991-r01'
BAND_PRECISION_ARMS = ('precision_shared', 'precision_resolved')
SEVERITY_RUN = '20261126-phase1-lora-w24-severity-weight-s2026112991-r01'
SEVERITY_ARMS = ('severity_uniform', 'severity_weighted')
HALF_PRIMARY_RUN = '20261126-phase1-lora-w24-half-primary-s2026112991-r01'
HALF_PRIMARY_ARMS = ('half_aux_control', 'half_primary')
CARRIER_ANCHOR_RUN = '20261126-phase1-lora-w24-carrier-anchor-s2026112991-r01'
CARRIER_ANCHOR_ARMS = ('carrier_linear', 'carrier_affine')
SIGNED_PROBE_RUN = '20261126-phase1-lora-w24-signed-probe-s2026112991-r01'
SIGNED_PROBE_ARMS = ('probe_positive', 'probe_signed')
PACKET_UNET_RUN = '20261126-phase1-lora-w24-packet-unet-s2026112991-r01'
PACKET_UNET_ARMS = ('unet_independent', 'unet_packet')
UNET_FIDELITY_RUN = '20261126-phase1-lora-w24-packet-unet-fidelity-s2026112991-r01'
UNET_FIDELITY_ARMS = ('unet_standard_long', 'unet_complex_long')
UNET_COMPONENT_RUN = '20261126-phase1-lora-w24-chirp-components-s2026112991-r01'
UNET_COMPONENT_ARMS = ('unet_component_control', 'unet_component_supervised')
UNET_TRANSPORT_RUN = '20261126-phase1-lora-w24-probe-transport-s2026112991-r01'
UNET_TRANSPORT_ARMS = ('probe_after_channel', 'probe_through_channel')
UNET_NOISE_COLOR_RUN = '20261126-phase1-lora-w24-noise-color-s2026112991-r01'
UNET_NOISE_COLOR_ARMS = ('wideband_noise_control', 'bandlimited_noise')
UNET_COMMON_RUN = '20261126-phase1-lora-w24-common-refine-s2026112991-r01'
UNET_COMMON_ARMS = ('common_refine_control', 'common_refine_bands')
POSTERIOR_RUN = '20261126-phase1-lora-w24-spectral-posterior-s2026112991-r01'
POSTERIOR_ARMS = ('posterior_all_bins', 'posterior_exclude_bin')
PROJECTION_RUN = '20261127-phase1-lora-w24-clean-projection-s2026112991-r01'
PROJECTION_ARMS = ('projection_control', 'projection_clean_aux')
SCRATCH_PRIOR_ARMS = (*POSTERIOR_ARMS, *PROJECTION_ARMS)
WEAK_TRANSFER_RUN = '20261127-phase1-lora-w24-weak-transfer-s2026112991-r01'
WEAK_TRANSFER_ARMS = ('weak_gain_control', 'weak_gain_private')
WEAK_TRANSFER_INTERMEDIATE = 24000
POLYPHASE_RUN = '20261127-phase1-lora-w24-polyphase-s2026112991-r01'
POLYPHASE_ARMS = ('aperture_context', 'native_polyphase')
ADDITIVE_REFERENCE_RUN = '20261127-phase1-lora-w24-additive-reference-s2026112991-r01'
ADDITIVE_REFERENCE_ARMS = ('restore_original', 'remove_additive')
NATIVE_PACKET_RUN = '20261127-phase1-lora-w24-native-packet-dura-s2026112991-r01'
NATIVE_PACKET_ARMS = ('native_dura_center', 'native_dura_whole')
NATIVE_ADDITIVE_RUN = '20261127-phase1-lora-w24-native-additive-s2026112991-r01'
NATIVE_ADDITIVE_ARMS = ('native_restore_original', 'native_remove_additive')
NATIVE_SPECTRAL_RUN = '20261127-phase1-lora-w24-native-band-risk-s2026112991-r01'
NATIVE_SPECTRAL_ARMS = ('native_band_control', 'native_band_risk')
NATIVE_CENTER_RUN = '20261127-phase1-lora-w24-native-center-direction-s2026112991-r01'
NATIVE_CENTER_ARMS = ('native_center_control', 'native_center_direction')
PRACTICAL_CACHE_RUN = '20261127-phase1-lora-w24-native-practical-supervision-s2026112991-r01'
PRACTICAL_RUN = '20261127-phase1-lora-w24-native-practical-supervision-s2026112991-r02'
PRACTICAL_KIND = 'native-practical-supervision'
PRACTICAL_ARMS = ('practical_matched_dose', 'practical_exact_additive')
NATIVE_PAIRED_ARMS = (*NATIVE_ADDITIVE_ARMS, *NATIVE_SPECTRAL_ARMS, *NATIVE_CENTER_ARMS)
NATIVE_MODEL_ARMS = (*NATIVE_PACKET_ARMS, *NATIVE_PAIRED_ARMS, *PRACTICAL_ARMS)
NATIVE_ADD_ONLY = (NATIVE_ADDITIVE_ARMS[1], *NATIVE_SPECTRAL_ARMS, *NATIVE_CENTER_ARMS)
LONG_UNET_ARMS = (*WEAK_TRANSFER_ARMS, *POLYPHASE_ARMS, *ADDITIVE_REFERENCE_ARMS)
UNET_WARM_ARMS = (*UNET_TRANSPORT_ARMS, *UNET_NOISE_COLOR_ARMS, *UNET_COMMON_ARMS)
UNET_ARMS = (*PACKET_UNET_ARMS, *UNET_FIDELITY_ARMS, *UNET_COMPONENT_ARMS, *UNET_WARM_ARMS,
             *LONG_UNET_ARMS)
EXPERT_RUN = '20261126-phase1-lora-w24-spectral-experts-s2026112991-r01'
EXPERT_ARMS = ('experts_packet', 'experts_band')
UNET_BASELINE = 'frozen_unet_half'
OUTER_TUNE_ARMS = (*BAND_TUNE_ARMS, *BAND_ADAPTER_ARMS, *BAND_REFINE_ARMS,
                   *WHITE_ARMS, *BAND_PRECISION_ARMS, *SEVERITY_ARMS, *HALF_PRIMARY_ARMS, *CARRIER_ANCHOR_ARMS,
                   *SIGNED_PROBE_ARMS)
OUTER_BASELINE = 'outer_parent_half'
OUTER_PARENT = '20261125-phase1-lora-w24-outer-objective-source-s2026112991-r01'
VARIANTS = {
    'direction': (DIRECTION_RUN, DIRECTION_ARMS),
    'readout': (READOUT_RUN, READOUT_ARMS),
    'robust': (ROBUST_RUN, ROBUST_ARMS),
    'final-radius': (RADIUS_RUN, RADIUS_ARMS),
    'peer': (PEER_RUN, PEER_ARMS),
    'coherent-error': (BIAS_RUN, BIAS_ARMS),
    'band-tune': (BAND_TUNE_RUN, BAND_TUNE_ARMS),
    'band-adapter': (BAND_ADAPTER_RUN, BAND_ADAPTER_ARMS),
    'band-refine': (BAND_REFINE_RUN, BAND_REFINE_ARMS),
    'white-corruption': (WHITE_RUN, WHITE_ARMS),
    'band-precision': (BAND_PRECISION_RUN, BAND_PRECISION_ARMS),
    'half-primary': (HALF_PRIMARY_RUN, HALF_PRIMARY_ARMS),
    'severity-weight': (SEVERITY_RUN, SEVERITY_ARMS),
    'carrier-anchor': (CARRIER_ANCHOR_RUN, CARRIER_ANCHOR_ARMS),
    'signed-probe': (SIGNED_PROBE_RUN, SIGNED_PROBE_ARMS),
    'packet-unet': (PACKET_UNET_RUN, PACKET_UNET_ARMS),
    'packet-unet-fidelity': (UNET_FIDELITY_RUN, UNET_FIDELITY_ARMS),
    'chirp-components': (UNET_COMPONENT_RUN, UNET_COMPONENT_ARMS),
    'probe-transport': (UNET_TRANSPORT_RUN, UNET_TRANSPORT_ARMS),
    'noise-color': (UNET_NOISE_COLOR_RUN, UNET_NOISE_COLOR_ARMS),
    'common-refine': (UNET_COMMON_RUN, UNET_COMMON_ARMS),
    'spectral-posterior': (POSTERIOR_RUN, POSTERIOR_ARMS),
    'clean-projection': (PROJECTION_RUN, PROJECTION_ARMS),
    'weak-transfer': (WEAK_TRANSFER_RUN, WEAK_TRANSFER_ARMS),
    'polyphase': (POLYPHASE_RUN, POLYPHASE_ARMS),
    'additive-reference': (ADDITIVE_REFERENCE_RUN, ADDITIVE_REFERENCE_ARMS),
    'native-packet-dura': (NATIVE_PACKET_RUN, NATIVE_PACKET_ARMS),
    'native-additive': (NATIVE_ADDITIVE_RUN, NATIVE_ADDITIVE_ARMS),
    'native-band-risk': (NATIVE_SPECTRAL_RUN, NATIVE_SPECTRAL_ARMS),
    'native-center-direction': (NATIVE_CENTER_RUN, NATIVE_CENTER_ARMS),
    PRACTICAL_KIND: (PRACTICAL_RUN, PRACTICAL_ARMS),
    'spectral-experts': (EXPERT_RUN, EXPERT_ARMS),
}
ARM_TRIAL = dict.fromkeys(ARMS, True)
ARM_TRIAL.update({arm: kind for kind, (_, arms) in VARIANTS.items() for arm in arms})


@dataclass(frozen=True)
class Trial:
    run: str
    arms: tuple
    steps: int
    source_modes: tuple
    contrasts: tuple
    routes: tuple = ROUTES
    source_intermediate: int | None = None

    @property
    def fit_arms(self):
        return self.arms[1:] if self.arms == NATIVE_PACKET_ARMS else self.arms


def trial(kind):
    if kind is True:
        return Trial(RUN, ARMS, STEPS, MODES, CONTRASTS)
    if kind not in VARIANTS:
        raise ValueError(f'Unknown matched W24 trial: {kind}')
    run, arms = VARIANTS[kind]
    modes = ('direct', 'all_half', *(f'{arm}_{alpha}' for alpha in ('half', 'full') for arm in arms))
    contrasts = ((modes[3], modes[2]), (modes[5], modes[4]), (modes[3], modes[0]))
    if arms == WEAK_TRANSFER_ARMS:
        intermediate = tuple(f'{arm}_s{WEAK_TRANSFER_INTERMEDIATE}_half' for arm in arms)
        comparisons = ((intermediate[1], intermediate[0]), (modes[2], intermediate[0]), (modes[3], intermediate[1]))
        baselines = (UNET_BASELINE, OUTER_BASELINE)
        return Trial(run, arms, 48000, (*modes, *baselines, *intermediate),
                     (*contrasts, *((modes[3], base) for base in baselines), *comparisons),
                     ROUTES[:1], WEAK_TRANSFER_INTERMEDIATE)
    if arms in (POLYPHASE_ARMS, ADDITIVE_REFERENCE_ARMS, NATIVE_PACKET_ARMS, NATIVE_ADDITIVE_ARMS,
                NATIVE_SPECTRAL_ARMS, NATIVE_CENTER_ARMS, PRACTICAL_ARMS):
        baselines = (UNET_BASELINE, OUTER_BASELINE)
        return Trial(run, arms, 24000 if arms[0] in NATIVE_MODEL_ARMS else 48000, (*modes, *baselines),
                     (*contrasts, *((modes[3], base) for base in baselines)), ROUTES[:1])
    if arms[0] in OUTER_TUNE_ARMS:
        steps = 1200 if kind == 'band-refine' else 1800
        return Trial(run, arms, steps, (*modes, OUTER_BASELINE), (*contrasts, (modes[3], OUTER_BASELINE)), ROUTES[:1])
    if arms[0] in (*EXPERT_ARMS, *UNET_COMPONENT_ARMS, *UNET_WARM_ARMS, *SCRATCH_PRIOR_ARMS):
        baselines = (UNET_BASELINE, OUTER_BASELINE)
        steps = {EXPERT_ARMS: 6000, PROJECTION_ARMS: 48000}.get(arms, 12000)
        if arms in (UNET_COMPONENT_ARMS, POSTERIOR_ARMS):
            steps = 24000
        return Trial(run, arms, steps, (*modes, *baselines),
                     (*contrasts, *((modes[3], baseline) for baseline in baselines)), ROUTES[:1])
    if arms[0] in UNET_ARMS:
        steps = 24000 if arms == UNET_FIDELITY_ARMS else 3600
        return Trial(run, arms, steps, (*modes, OUTER_BASELINE), (*contrasts, (modes[3], OUTER_BASELINE)), ROUTES[:1])
    return Trial(run, arms, 3600, modes, contrasts, ROUTES[:1])


def restorer(training, arm):
    if arm not in ARM_TRIAL:
        raise ValueError(f'Unknown prior arm: {arm}')
    if arm == 'readout_adaptive':
        from experiments.pamf_lora_main.w24_readout_strength import adaptive_final

        class AdaptiveReadout(training.DuraW24):
            def propose(self, y, quality, rms, edges):
                (coarse, first, final), action, variance, noise, support = super().propose(y, quality, rms, edges)
                final = adaptive_final(y, final, noise, rms)
                return (coarse, first, final), action, variance, noise, support

        return AdaptiveReadout()
    if arm == 'robust_map':
        class RobustMap(training.DuraW24):
            def stages(self, y, quality, rms, **options):
                return super().stages(y, quality, rms, **dict(options, robust=True))

        return RobustMap()
    if arm == 'radius_adaptive':
        from experiments.pamf_lora_main.w24_final_radius import projected_stages

        archive_package = training.DuraW24.__module__.rpartition('.')[0]
        old = importlib.import_module('.dura_capacity', archive_package)

        class FinalRadius(training.DuraW24):
            def stages(self, y, quality, rms, *, condition=()):
                return projected_stages(self, old, y, quality, rms, condition=condition)

        return FinalRadius()
    if arm in PEER_ARMS:
        from experiments.pamf_lora_main.w24_peer_innovation import restorer as innovation_restorer

        return innovation_restorer(training, use_peers=arm == 'innovation_peer')
    if arm in BAND_ADAPTER_ARMS:
        from experiments.pamf_lora_main.w24_band_adapter import restorer as adapter_restorer

        return adapter_restorer(training, frequency=arm == BAND_ADAPTER_ARMS[1])
    if arm in BAND_REFINE_ARMS:
        from experiments.pamf_lora_main.w24_band_refine import BandRefiner

        return BandRefiner(training, polar=arm == BAND_REFINE_ARMS[1])
    if arm in BAND_PRECISION_ARMS:
        from experiments.pamf_lora_main.w24_band_precision import restorer as precision_restorer

        return precision_restorer(training, resolved=arm == BAND_PRECISION_ARMS[1])
    if arm in CARRIER_ANCHOR_ARMS:
        from experiments.pamf_lora_main.w24_carrier_anchor import restorer as carrier_restorer

        return carrier_restorer(training, anchored=arm == CARRIER_ANCHOR_ARMS[1])
    if arm in UNET_COMMON_ARMS:
        from experiments.pamf_lora_main.w24_common_refiner import CommonRefiner

        return CommonRefiner()
    if arm in POSTERIOR_ARMS:
        from experiments.pamf_lora_main.w24_spectral_posterior import SpectralPosterior

        return SpectralPosterior(exclude_current=arm == POSTERIOR_ARMS[1])
    if arm in PROJECTION_ARMS:
        from experiments.pamf_lora_main.w24_clean_projection import CleanProjection

        return CleanProjection()
    if arm in POLYPHASE_ARMS:
        from experiments.pamf_lora_main.w24_polyphase import NativePolyphaseUNet

        return NativePolyphaseUNet(native_context=arm == POLYPHASE_ARMS[1])
    if arm in NATIVE_MODEL_ARMS:
        from experiments.pamf_lora_main.w24_packet_dura import NativePacketDura

        return NativePacketDura()
    if arm in UNET_ARMS:
        from experiments.pamf_lora_main.w24_packet_unet import PacketUNet

        return PacketUNet(across_chirps=arm != PACKET_UNET_ARMS[0], separate_chirps=arm in UNET_COMPONENT_ARMS)
    if arm in EXPERT_ARMS:
        from experiments.pamf_lora_main.w24_spectral_experts import SpectralExperts

        return SpectralExperts(training.DuraW24(), local=arm == EXPERT_ARMS[1])
    if arm != 'three':
        return training.DuraW24()

    class ThreePrior(training.DuraW24):
        def __init__(self):
            super().__init__()
            # Leave the two matched initial prior states intact. The third has
            # separately trainable parameters, initialized from prior two.
            self.priors.append(deepcopy(self.priors[-1]))

        def propose(self, *args, **kwargs):
            stages, *physical = super().propose(*args, **kwargs)
            return (stages[0], stages[1], stages[-1]), *physical

    return ThreePrior()


def fit_inputs(rt, arm):
    if arm not in ARM_TRIAL:
        raise ValueError(f'Unknown prior arm: {arm}')
    if arm == NATIVE_PACKET_ARMS[0]:
        raise ValueError('The center-only readout reuses the single complete-packet checkpoint')
    inputs = dict(source_root=rt['source_root'], run_root=str(PurePosixPath(rt['run_root']) / arm))
    if arm in OUTER_TUNE_ARMS:
        inputs['source_parent'] = rt['source_parent']
    if arm in EXPERT_ARMS:
        inputs.update(source_parent=rt['source_parent'], unet_parent=rt['unet_parent'])
    if arm in UNET_WARM_ARMS:
        inputs['unet_parent'] = rt['unet_parent']
    if arm in PRACTICAL_ARMS:
        inputs['practical_cache'] = rt['practical_cache']
    if arm == PRACTICAL_ARMS[0]:
        inputs['dose_parent'] = rt['dose_parent']
    return inputs


def restore_center(original, delta, fraction):
    restored = original.clone()
    restored[..., :384] += fraction * delta
    return restored


def native_spectral_weight(arm):
    return .50 if arm == NATIVE_SPECTRAL_ARMS[1] else 0.


def native_center_weight(arm):
    return .20 if arm == NATIVE_CENTER_ARMS[1] else 0.


def promotion(routes, kind=True):
    spec = trial(kind)
    direct, mature, control, candidate = spec.source_modes[:4]
    controls = (control, direct, mature, OUTER_BASELINE) if OUTER_BASELINE in spec.source_modes else (control, direct, mature)
    if UNET_BASELINE in spec.source_modes:
        controls = (*controls, UNET_BASELINE)
    return len(routes) == len(spec.routes) and all(
        row[candidate]['by_scene']['clean'] >= row[direct]['by_scene']['clean']
        and row[candidate]['six_correct'] > max(row[k]['six_correct'] for k in controls)
        and row[candidate]['low_urban_correct'] >= max(row[k]['low_urban_correct'] for k in controls)
        for row in routes)


def frozen_source_results(root, kind):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.w24_native_full import VIEWS

    spec = trial(kind)
    if read(root / 'state.json')['status'] not in ('SOURCE_SCORED', 'FORMAL_PREDICTING'):
        raise ValueError('Expected frozen and scored source rows')
    results, physical_ids = [], None
    for route in spec.routes:
        predictions = read(root / route / 'source-predictions.json')
        scores = read(root / route / 'source-scores.json')
        ids, waveform = predictions['physical_ids'], predictions['waveform']
        if (predictions['status'] != 'PREDICTIONS_FROZEN' or predictions['target_reads'] != 0
                or predictions['source_truth_reads'] != 0 or predictions['modes'] != list(spec.source_modes)
                or predictions['views'] != list(VIEWS) or len(ids) != 300 or len(set(ids)) != 300
                or (physical_ids is not None and ids != physical_ids)
                or waveform['candidate_steps'] != spec.steps or waveform['candidate_run'] != spec.run
                or waveform.get('source_intermediate') != spec.source_intermediate
                or waveform['route'] != route or waveform['ungated'] is not True
                or waveform['amplitudes'] != [.5, 1.]
                or waveform['activation'] != dict.fromkeys(VIEWS, [300]*(len(spec.source_modes)-1))
                or scores['status'] != 'ANALYZED' or scores['source_truth_last'] is not True
                or scores['target_reads'] != 0 or scores['n_packets'] != 300 or scores['n_practical'] != 1800
                or scores['waveform'] != waveform):
            raise ValueError(f'Invalid source evaluation for {route}')
        physical_ids = ids
        results.append(scores['scores'])
    return results


def qualified_source(root, kind=True):
    if not promotion(frozen_source_results(root, kind), kind):
        raise ValueError('Source comparison did not qualify')


def source_recipe(training, arm, *, initialize=None):
    def finalize(model, _tracker, _calibration):
        if arm in EXPERT_ARMS:
            trained, key = (model.router,), 'router_gradient_norms'
            expert_parameters = (*model.dura.parameters(), *model.unet.parameters())
            if any(p.requires_grad or p.grad is not None for p in expert_parameters):
                raise ValueError('Expected frozen source experts')
        elif arm in UNET_COMMON_ARMS:
            trained, key = (model.head,), 'common_head_gradient_norms'
            if any(p.requires_grad or p.grad is not None for p in model.base.parameters()):
                raise ValueError('Expected a frozen source packet U-Net')
        elif arm in POSTERIOR_ARMS:
            trained, key = (model.lower, model.upper, model.prior), 'spectral_prior_gradient_norms'
        elif arm in PROJECTION_ARMS:
            trained, key = (model.encoder, model.context, model.decoder), 'clean_prior_gradient_norms'
        elif arm in NATIVE_MODEL_ARMS:
            trained, key = (model.coarse, *model.priors), 'native_packet_gradient_norms'
        elif arm in UNET_ARMS:
            trained = (model.stem, model.temporal, model.axis, model.time_edit, model.spectral)
            key = 'unet_gradient_norms'
        elif arm in BAND_ADAPTER_ARMS:
            trained, key = (model.band_adapter,), 'adapter_gradient_norms'
        elif arm in BAND_PRECISION_ARMS:
            trained, key = (model.band_precision,), 'precision_gradient_norms'
            if any(p.requires_grad or p.grad is not None for name, p in model.named_parameters()
                   if not name.startswith('band_precision.')):
                raise ValueError('Expected a frozen source backbone for band precision')
        elif arm in BAND_REFINE_ARMS:
            trained, key = (model.head,), 'head_gradient_norms'
            if any(p.requires_grad or p.grad is not None for p in model.backbone.parameters()):
                raise ValueError('Expected a frozen source backbone for band refinement')
        else:
            trained, key = model.priors, 'prior_gradient_norms'
        grads = [sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)**.5
                 for module in trained]
        if not all(0 < value < float('inf') for value in grads):
            raise FloatingPointError(f'Invalid gradient in a trainable W24 prior or adapter: {grads}')
        info = dict(input_gate=False, prior_count=len(getattr(model, 'priors', ())), **{key: grads},
                    parameters=sum(p.numel() for p in model.parameters()))
        if arm in OUTER_TUNE_ARMS:
            info['source_parent'] = OUTER_PARENT
        if arm in BAND_ADAPTER_ARMS:
            info['trainable_parameters'] = sum(p.numel() for p in model.band_adapter.parameters())
        if arm in BAND_REFINE_ARMS:
            info.update(trainable_parameters=sum(p.numel() for p in model.head.parameters()), backbone_frozen=True)
        if arm in BAND_PRECISION_ARMS:
            info.update(trainable_parameters=sum(p.numel() for p in model.band_precision.parameters()),
                        backbone_frozen=True)
        if arm in WHITE_ARMS:
            info['source_perturbation'] = 'equal_center_energy_white' if arm == WHITE_ARMS[1] else 'archived'
        if arm in SEVERITY_ARMS:
            info['severity_weighted'] = arm == SEVERITY_ARMS[1]
        if arm in CARRIER_ANCHOR_ARMS:
            info['final_carrier_anchor'] = arm == CARRIER_ANCHOR_ARMS[1]
        if arm in SIGNED_PROBE_ARMS:
            info['signed_probe_transfer'] = arm == SIGNED_PROBE_ARMS[1]
        if arm in UNET_ARMS:
            info.update(initialization='random', across_chirps=arm != PACKET_UNET_ARMS[0])
        if arm in UNET_FIDELITY_ARMS:
            info['complex_fidelity'] = arm == UNET_FIDELITY_ARMS[1]
        if arm in WEAK_TRANSFER_ARMS:
            info.update(private_weak_gain=arm == WEAK_TRANSFER_ARMS[1],
                        source_intermediate=WEAK_TRANSFER_INTERMEDIATE)
        if arm in POLYPHASE_ARMS:
            info.update(native_polyphase=arm == POLYPHASE_ARMS[1], pre_channel_probe=True)
        if arm in (*ADDITIVE_REFERENCE_ARMS, *NATIVE_PAIRED_ARMS):
            info.update(additive_only=arm in (ADDITIVE_REFERENCE_ARMS[1], *NATIVE_ADD_ONLY),
                        shared_generic_action=True)
        if arm in NATIVE_SPECTRAL_ARMS:
            info['native_spectral_weight'] = native_spectral_weight(arm)
        if arm in NATIVE_CENTER_ARMS:
            info['native_center_weight'] = native_center_weight(arm)
        if arm in PRACTICAL_ARMS:
            info.update(legal_source_practical_traces=True, supervision=arm,
                        dose_parent=NATIVE_CENTER_RUN if arm == PRACTICAL_ARMS[0] else None)
        if arm in NATIVE_MODEL_ARMS:
            info.update(initialization='random', native_capture_samples=8192, observation_steps=2)
        if arm in UNET_COMPONENT_ARMS:
            info.update(separate_chirps=True, separate_loss=arm == UNET_COMPONENT_ARMS[1])
        if arm in UNET_WARM_ARMS:
            info.update(initialization='verified_unet_model_only', unet_parent=UNET_FIDELITY_RUN)
        if arm in UNET_TRANSPORT_ARMS:
            info['probe_through_channel'] = arm == UNET_TRANSPORT_ARMS[1]
        if arm in UNET_NOISE_COLOR_ARMS:
            info['bandlimited_noise'] = arm == UNET_NOISE_COLOR_ARMS[1]
        if arm in UNET_COMMON_ARMS:
            info.update(common_band_loss=arm == UNET_COMMON_ARMS[1], backbone_frozen=True,
                        trainable_parameters=sum(p.numel() for p in model.head.parameters()))
        if arm in POSTERIOR_ARMS:
            info.update(initialization='random', excludes_center_bin=arm == POSTERIOR_ARMS[1],
                        objective='actual_posterior_with_prior_auxiliary')
        if arm in PROJECTION_ARMS:
            info.update(initialization='random', clean_projection_loss=arm == PROJECTION_ARMS[1],
                        consistency_passes=2)
        if arm in EXPERT_ARMS:
            info.update(expert_parents=[OUTER_PARENT, UNET_FIDELITY_RUN], local_router=arm == EXPERT_ARMS[1],
                        trainable_parameters=sum(p.numel() for p in model.router.parameters()))
        return None, info

    spec = trial(ARM_TRIAL[arm])
    name = f'w24_{arm}_prior' if arm in ARMS else f'w24_{arm}'
    milestones = (spec.steps // 3, 2 * spec.steps // 3)
    if arm in (*UNET_FIDELITY_ARMS, *UNET_COMPONENT_ARMS, *SCRATCH_PRIOR_ARMS, *LONG_UNET_ARMS,
               *NATIVE_MODEL_ARMS):
        milestones = (2000, 6000, 12000, 18000)
    if arm in (*PROJECTION_ARMS, *LONG_UNET_ARMS):
        milestones = (*milestones, 24000, 36000, 42000)
    if arm in UNET_WARM_ARMS:
        milestones = (2000, 6000, 9000)
    return replace(training.RECIPE, name=name, steps=spec.steps,
                   validation_steps=milestones,
                   model=partial(restorer, training, arm), finalize_calibration=finalize,
                   initialize=initialize, coarse_first=training.RECIPE.coarse_first
                   and arm not in (*OUTER_TUNE_ARMS, *UNET_ARMS, *EXPERT_ARMS, *SCRATCH_PRIOR_ARMS,
                                   *NATIVE_MODEL_ARMS))


def snapshot_objective(calculate, archived, contract, snapshot, checkpoint_step):
    import torch

    def capture(model, tracker, raw, rng, *, with_identity, recipe, step):
        # Validation at this milestone follows its update. The next fit call starts
        # before another update, with exactly the completed milestone weights.
        if with_identity and step == checkpoint_step + 1:
            proof = archived.proof(contract, replace(recipe, steps=checkpoint_step))
            with snapshot.open('xb') as stream:
                torch.save(dict(proof=proof, weights=model.state_dict(), threshold=None), stream)
        return calculate(model, tracker, raw, rng, with_identity=with_identity, recipe=recipe, step=step)

    return capture


def fit(config, arm, *, smoke):
    import torch
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    from experiments.cric_dura_comparison.io import read, tensor
    from experiments.pamf_lora_main import channel_blind_source, dura_w24_model, dura_w24_train

    rt = read(config)
    if arm == NATIVE_PACKET_ARMS[0]:
        raise ValueError('The center-only readout has no separate source fit')
    paths = {'source_root', 'run_root'}
    if arm in OUTER_TUNE_ARMS:
        paths.add('source_parent')
    if arm in EXPERT_ARMS:
        paths.update(('source_parent', 'unet_parent'))
    if arm in UNET_WARM_ARMS:
        paths.add('unet_parent')
    if arm in PRACTICAL_ARMS:
        paths.add('practical_cache')
    if arm == PRACTICAL_ARMS[0]:
        paths.add('dose_parent')
    if set(rt) != paths:
        raise ValueError('Fit may receive only registered source paths and its own output root')

    def prepare(roles, device, *, smoke, fit_per_rx, bank_per_rx):
        source = Path(roles['source_root'])
        contract = dura_w24_train.source_contract(source)
        iq, rx, _, _ = dura_w24_train.source_role(source, contract, 'U_s')
        if smoke:
            train, _, held = channel_blind_source.partition(rx, fit=5, cal=4, held=3)
        else:
            train, _, held = channel_blind_source.partition(rx, fit=fit_per_rx)
        train, held = (tensor(iq[index].view('float32'), device).view(torch.complex64) for index in (train, held))
        return (train, torch.empty(0, device=device), held), contract

    calculate, initialize = dura_w24_model.objective, None
    if arm in PRACTICAL_ARMS:
        from experiments.pamf_lora_main.w24_packet_dura import NativePacketDura
        from experiments.pamf_lora_main.w24_practical_supervision import PracticalLoss, training_sets

        run = Path(rt['run_root']).parent
        if run.name != PRACTICAL_RUN or Path(rt['practical_cache']) != run.parent / PRACTICAL_CACHE_RUN / 'source_practical':
            raise ValueError('Expected the immutable source-only Practical cache')
        teacher = None
        if arm == PRACTICAL_ARMS[0]:
            parent = Path(rt['dose_parent'])
            if parent != run.parent / NATIVE_CENTER_RUN:
                raise ValueError('The regional norm source must be the registered frozen native direction model')
            contract = dura_w24_train.source_contract(Path(rt['source_root']))
            _, weights, _ = read_candidate_weights(parent, contract, 'cpu', NATIVE_CENTER_ARMS[1],
                                                   dura_w24_train, channel_blind_source)
            teacher = NativePacketDura().to('cuda:0').eval().requires_grad_(False)
            teacher.load_state_dict(weights, strict=True)
        prepare, calculate = training_sets, PracticalLoss('cuda:0', teacher)
    if arm in NATIVE_PACKET_ARMS:
        from experiments.pamf_lora_main.w24_packet_dura import source_objective

        def calculate(model, _tracker, raw, rng, *, with_identity, recipe, step):
            clean = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
            received = dura_w24_model.condition_waveforms(clean.unflatten(-1, (8, 1024)), rng).flatten(-2)
            if with_identity:
                received[-2:] = clean[-2:]
            return source_objective(model, received, clean, with_identity=with_identity)
    if arm in NATIVE_PAIRED_ARMS:
        from experiments.pamf_lora_main.w24_packet_dura import native_additive_pair, source_objective

        spectral_weight = native_spectral_weight(arm) if arm in NATIVE_SPECTRAL_ARMS else None
        center_project = None
        if arm in NATIVE_CENTER_ARMS:
            from experiments.pamf_lora_main.w24_native_center import NativeCenterProjection

            center_project = NativeCenterProjection().to('cuda:0')

        def calculate(model, _tracker, raw, rng, *, with_identity, recipe, step):
            received, references = native_additive_pair(raw, rng, with_identity=with_identity)
            reference = references[arm in NATIVE_ADD_ONLY]
            return source_objective(model, received, reference, with_identity=with_identity,
                                    audit_references=references, spectral_weight=spectral_weight,
                                    center_project=center_project, center_weight=native_center_weight(arm))
    if arm in ('direction_first', 'coherent_error'):
        from experiments.pamf_lora_main.w24_first_direction import objective_for

        calculate = objective_for(dura_w24_model, output_bias=arm == 'coherent_error')
    if arm in UNET_ARMS:
        from experiments.pamf_lora_main.w24_packet_unet import objective_for

        fidelity = (arm == UNET_FIDELITY_ARMS[1]) if arm in UNET_FIDELITY_ARMS else None
        separation = (arm == UNET_COMPONENT_ARMS[1]) if arm in UNET_COMPONENT_ARMS else None
        transport = (arm == UNET_TRANSPORT_ARMS[1]) if arm in UNET_TRANSPORT_ARMS else None
        colored_noise = (arm == UNET_NOISE_COLOR_ARMS[1]) if arm in UNET_NOISE_COLOR_ARMS else None
        common_bands = (arm == UNET_COMMON_ARMS[1]) if arm in UNET_COMMON_ARMS else None
        weak_gain = (arm == WEAK_TRANSFER_ARMS[1]) if arm in WEAK_TRANSFER_ARMS else None
        polyphase = arm in POLYPHASE_ARMS
        additive_reference = (arm == ADDITIVE_REFERENCE_ARMS[1]) if arm in ADDITIVE_REFERENCE_ARMS else None
        calculate = objective_for(dura_w24_model, complex_fidelity=fidelity, separate_loss=separation,
                                  transport_probe=transport, colored_noise=colored_noise, common_bands=common_bands,
                                  posterior_primary=arm in LONG_UNET_ARMS, weak_gain=weak_gain,
                                  raw_polyphase=polyphase, additive_reference=additive_reference)
    if arm in SCRATCH_PRIOR_ARMS:
        from experiments.pamf_lora_main.w24_packet_unet import objective_for

        clean_aux = (arm == PROJECTION_ARMS[1]) if arm in PROJECTION_ARMS else None
        calculate = objective_for(dura_w24_model, posterior_primary=True, clean_projection=clean_aux)
    if arm in UNET_WARM_ARMS:
        parent = Path(rt['unet_parent'])
        if parent != Path(rt['run_root']).parent.parent / UNET_FIDELITY_RUN:
            raise ValueError('Expected the registered source-only packet U-Net initializer')
        contract = dura_w24_train.source_contract(Path(rt['source_root']))
        _, weights, _ = read_candidate_weights(parent, contract, 'cpu', UNET_FIDELITY_ARMS[0],
                                                dura_w24_train, channel_blind_source)

        def initialize(model, _tracker, _training, _calibration):
            backbone = model.base if arm in UNET_COMMON_ARMS else model
            backbone.load_state_dict(weights, strict=True)
    if arm in EXPERT_ARMS:
        from experiments.pamf_lora_main.w24_outer_objective_source import read_outer_checkpoint
        from experiments.pamf_lora_main.w24_spectral_experts import objective_for

        runs = Path(rt['run_root']).parent.parent
        dura_root, unet_root = (Path(rt[k]) for k in ('source_parent', 'unet_parent'))
        if (dura_root, unet_root) != (runs / OUTER_PARENT, runs / UNET_FIDELITY_RUN):
            raise ValueError('Expected the registered source-only expert runs')
        contract = dura_w24_train.source_contract(Path(rt['source_root']))
        _, dwa, _ = read_outer_checkpoint(dura_root, contract, 'cpu', dura_w24_train, channel_blind_source)
        _, uw, _ = read_candidate_weights(unet_root, contract, 'cpu', UNET_FIDELITY_ARMS[0],
                                          dura_w24_train, channel_blind_source)

        def initialize(model, _tracker, _training, _calibration):
            model.dura.load_state_dict(dwa, strict=True)
            model.unet.load_state_dict(uw, strict=True)

        calculate = objective_for(dura_w24_model)
    if arm in OUTER_TUNE_ARMS:
        from experiments.pamf_lora_main.w24_band_tune import WEIGHT
        from experiments.pamf_lora_main.w24_outer_objective_source import objective_for, read_outer_checkpoint

        parent = Path(rt['source_parent'])
        if parent != Path(rt['run_root']).parent.parent / OUTER_PARENT:
            raise ValueError('Expected the registered source-only outer-loss parent')
        contract = dura_w24_train.source_contract(Path(rt['source_root']))
        _, weights, _ = read_outer_checkpoint(parent, contract, 'cpu', dura_w24_train, channel_blind_source)

        def initialize(model, _tracker, _training, _calibration):
            if arm in BAND_ADAPTER_ARMS:
                from experiments.pamf_lora_main.w24_band_adapter import load_parent

                load_parent(model, weights)
            elif arm in BAND_PRECISION_ARMS:
                from experiments.pamf_lora_main.w24_band_adapter import load_parent

                load_parent(model, weights, new_module='band_precision')
            elif arm in BAND_REFINE_ARMS:
                model.backbone.load_state_dict(weights, strict=True)
            else:
                model.load_state_dict(weights, strict=True)

        weight = WEIGHT if arm == BAND_TUNE_ARMS[1] else 0.
        if arm in BAND_REFINE_ARMS:
            calculate = objective_for(dura_w24_model, refine_band=True)
        elif arm in WHITE_ARMS:
            calculate = objective_for(dura_w24_model, white_corruption=arm == WHITE_ARMS[1])
        elif arm in SEVERITY_ARMS:
            calculate = objective_for(dura_w24_model, severity_weighted=arm == SEVERITY_ARMS[1])
        elif arm in HALF_PRIMARY_ARMS:
            calculate = objective_for(dura_w24_model, half_primary=arm == HALF_PRIMARY_ARMS[1])
        elif arm in CARRIER_ANCHOR_ARMS:
            calculate = objective_for(dura_w24_model)
        elif arm in SIGNED_PROBE_ARMS:
            calculate = objective_for(dura_w24_model, signed_probe=arm == SIGNED_PROBE_ARMS[1])
        else:
            calculate = objective_for(dura_w24_model, band_direction_weight=weight)
    numerical_policy()
    recipe = source_recipe(dura_w24_train, arm, initialize=initialize)
    if arm in WEAK_TRANSFER_ARMS and not smoke:
        contract = dura_w24_train.source_contract(Path(rt['source_root']))
        snapshot = Path(rt['run_root']) / f'{recipe.name}-step{WEAK_TRANSFER_INTERMEDIATE}.pt'
        calculate = snapshot_objective(calculate, channel_blind_source, contract, snapshot, WEAK_TRANSFER_INTERMEDIATE)
    channel_blind_source.fit(rt, smoke=smoke, recipe=recipe,
                             calculate=calculate, receiver_batches=True, prepare=prepare,
                             make_tracker=dura_w24_train.WideTracker)


def read_candidate_weights(root, contract, device, arm, training, archived, *, source_step=None):
    import torch
    from experiments.cric_dura_comparison.io import read

    recipe = source_recipe(training, arm)
    row = root / arm
    summary = read(row / f'{recipe.name}-complete.json')
    expected = archived.proof(contract, recipe)
    prior_count = 0 if arm in (*UNET_ARMS, *EXPERT_ARMS, *SCRATCH_PRIOR_ARMS) else (3 if arm == 'three' else 2)
    if (any(summary.get(k) != value for k, value in expected.items() if k != 'contract')
            or any(expected.get(key) != 0 for key in ISOLATION)
            or summary.get('n_calibration') != 0 or summary.get('input_gate') is not False
            or summary.get('threshold') is not None or summary.get('prior_count') != prior_count):
        raise ValueError(f'Unverified source freeze for {arm}')
    if arm in OUTER_TUNE_ARMS and summary.get('source_parent') != OUTER_PARENT:
        raise ValueError(f'Unverified warm start for {arm}')
    if arm in (*BAND_REFINE_ARMS, *BAND_PRECISION_ARMS) and summary.get('backbone_frozen') is not True:
        raise ValueError(f'Unverified frozen source backbone for {arm}')
    if arm in CARRIER_ANCHOR_ARMS and summary.get('final_carrier_anchor') is not (arm == CARRIER_ANCHOR_ARMS[1]):
        raise ValueError(f'Unverified final carrier anchor for {arm}')
    if arm in SIGNED_PROBE_ARMS and summary.get('signed_probe_transfer') is not (arm == SIGNED_PROBE_ARMS[1]):
        raise ValueError(f'Unverified signed probe transfer for {arm}')
    initialization = 'verified_unet_model_only' if arm in UNET_WARM_ARMS else 'random'
    if arm in UNET_ARMS and (summary.get('initialization') != initialization
                            or summary.get('across_chirps') is not (arm != PACKET_UNET_ARMS[0])):
        raise ValueError(f'Unverified packet U-Net architecture for {arm}')
    if arm in UNET_FIDELITY_ARMS and summary.get('complex_fidelity') is not (arm == UNET_FIDELITY_ARMS[1]):
        raise ValueError(f'Unverified packet U-Net complex fidelity objective for {arm}')
    if arm in WEAK_TRANSFER_ARMS and (summary.get('private_weak_gain') is not (arm == WEAK_TRANSFER_ARMS[1])
                                      or summary.get('source_intermediate') != WEAK_TRANSFER_INTERMEDIATE):
        raise ValueError(f'Unverified weak gain training objective for {arm}')
    if arm in POLYPHASE_ARMS and (summary.get('native_polyphase') is not (arm == POLYPHASE_ARMS[1])
                                  or summary.get('pre_channel_probe') is not True):
        raise ValueError(f'Unverified polyphase input path for {arm}')
    additive_arms = (*ADDITIVE_REFERENCE_ARMS, *NATIVE_PAIRED_ARMS)
    additive_only = arm in (ADDITIVE_REFERENCE_ARMS[1], *NATIVE_ADD_ONLY)
    if arm in additive_arms and (summary.get('additive_only') is not additive_only
                                 or summary.get('shared_generic_action') is not True):
        raise ValueError(f'Unverified additive reference for {arm}')
    if arm in NATIVE_SPECTRAL_ARMS and summary.get('native_spectral_weight') != native_spectral_weight(arm):
        raise ValueError(f'Unverified native spectral supervision for {arm}')
    if arm in NATIVE_CENTER_ARMS and summary.get('native_center_weight') != native_center_weight(arm):
        raise ValueError(f'Unverified native center supervision for {arm}')
    if arm in PRACTICAL_ARMS and (summary.get('legal_source_practical_traces') is not True
                                  or summary.get('supervision') != arm
                                  or summary.get('dose_parent') != (NATIVE_CENTER_RUN if arm == PRACTICAL_ARMS[0] else None)):
        raise ValueError(f'Unverified legal source-only Practical supervision for {arm}')
    if arm in NATIVE_MODEL_ARMS and (summary.get('initialization') != 'random'
                                      or summary.get('native_capture_samples') != 8192
                                      or summary.get('observation_steps') != 2):
        raise ValueError('Unverified native full-packet checkpoint')
    if arm in UNET_COMPONENT_ARMS and (summary.get('separate_chirps') is not True
                                       or summary.get('separate_loss') is not (arm == UNET_COMPONENT_ARMS[1])):
        raise ValueError(f'Unverified packet U-Net chirp decomposition for {arm}')
    if arm in UNET_WARM_ARMS and summary.get('unet_parent') != UNET_FIDELITY_RUN:
        raise ValueError(f'Unverified packet U-Net ancestor for {arm}')
    if arm in UNET_TRANSPORT_ARMS and summary.get('probe_through_channel') is not (arm == UNET_TRANSPORT_ARMS[1]):
        raise ValueError(f'Unverified packet U-Net probe transport for {arm}')
    if arm in UNET_NOISE_COLOR_ARMS and summary.get('bandlimited_noise') is not (arm == UNET_NOISE_COLOR_ARMS[1]):
        raise ValueError(f'Unverified packet U-Net source noise distribution for {arm}')
    if arm in UNET_COMMON_ARMS and (summary.get('common_band_loss') is not (arm == UNET_COMMON_ARMS[1])
                                    or summary.get('backbone_frozen') is not True):
        raise ValueError(f'Unverified common refinement recipe for {arm}')
    if arm in POSTERIOR_ARMS and (summary.get('initialization') != 'random'
                                   or summary.get('excludes_center_bin') is not (arm == POSTERIOR_ARMS[1])
                                   or summary.get('objective') != 'actual_posterior_with_prior_auxiliary'):
        raise ValueError(f'Unverified spectral posterior recipe for {arm}')
    if arm in PROJECTION_ARMS and (summary.get('initialization') != 'random'
                                    or summary.get('clean_projection_loss') is not (arm == PROJECTION_ARMS[1])
                                    or summary.get('consistency_passes') != 2):
        raise ValueError(f'Unverified clean projection recipe for {arm}')
    if arm in EXPERT_ARMS and (summary.get('expert_parents') != [OUTER_PARENT, UNET_FIDELITY_RUN]
                               or summary.get('local_router') is not (arm == EXPERT_ARMS[1])):
        raise ValueError(f'Unverified spectral expert provenance for {arm}')
    if source_step is None:
        path = row / f'{recipe.name}-model.pt'
    else:
        if arm not in WEAK_TRANSFER_ARMS or source_step != WEAK_TRANSFER_INTERMEDIATE:
            raise ValueError('The requested source comparison checkpoint was not registered')
        path = row / f'{recipe.name}-step{source_step}.pt'
        expected = archived.proof(contract, replace(recipe, steps=source_step))
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['proof'] != expected or checkpoint['threshold'] is not None:
        raise ValueError(f'Invalid {arm} checkpoint source binding')
    return recipe, checkpoint['weights'], summary


def load_candidate(root, contract, device, arm, *, source_step=None):
    training = importlib.import_module('lora_wide_dura_archive.dura_w24_train')
    archived = importlib.import_module('lora_wide_dura_archive.channel_blind_source')
    fit_arm = NATIVE_PACKET_ARMS[1] if arm in NATIVE_PACKET_ARMS else arm
    recipe, weights, summary = read_candidate_weights(root, contract, device, fit_arm, training, archived,
                                                       source_step=source_step)
    model = recipe.model().to(device).eval().requires_grad_(False)
    model.load_state_dict(weights, strict=True)
    return model, summary


def dispatch(runtime):
    from experiments.cric_dura_comparison.io import read, write
    from experiments.pamf_lora_main.packet_runner import schedule
    from experiments.pamf_lora_main.runner import available
    from experiments.pamf_lora_main.w24_native_full import W24_COMMIT
    from experiments.pamf_lora_main.w24_wide_transfer import training_runtime

    rt = read(runtime)
    kind = rt['third_prior']
    spec = trial(kind)
    archive, _ = training_runtime(rt)
    reference = Path(rt['w24_release_root'])
    root = Path(rt['run_root'])
    root.mkdir()
    state = root / 'state.json'
    own = Path(__file__).resolve()
    evaluator = own.with_name('w24_native_residual_source.py')

    def status(**fields):
        write(state, fields, exclusive=False)

    def start(action, label, gpu, cwd, entry, config, *extra):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   PYTHONPATH=str(cwd), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (root / f'{label}.log').open('x', encoding='utf-8') as log:
            job = subprocess.Popen([rt['python'], '-u', str(entry), action, str(config), *extra],
                                   cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print('W24_PRIOR_STARTED', label, job.pid, gpu, flush=True)
        return job

    def training_job(_plan, _phase, arm, gpu):
        return start('fit-arm', f'{arm}-fit', gpu, archive, own, root / arm / 'fit-inputs.json', arm)

    try:
        if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Expected the registered W24 evaluator archive')
        if kind == PRACTICAL_KIND:
            cache = Path(rt['practical_cache'])
            if read(cache/'complete.json')['status'] != 'SOURCE_TRACE_FROZEN' or not (cache/'pairs.npy').is_file():
                raise RuntimeError('Closed r01 source-only Practical pair bank required')
        for arm in spec.fit_arms:
            (root / arm).mkdir()
            write(root / arm / 'fit-inputs.json', fit_inputs(rt, arm))
        def eligible():
            policy = {'training_cap': True} if rt.get('training_cap') else {}
            return available(rt['allowed_gpus'], rt['max_compute_processes'], **policy)

        cards = eligible()
        if kind != PRACTICAL_KIND and len(cards) < len(spec.fit_arms):
            raise RuntimeError('Insufficient eligible GPUs for the registered training rows')
        status(status='SOURCE_TRAINING' if cards else 'WAIT_GPU_TRAINING', arms=list(spec.fit_arms),
               steps=spec.steps, available_gpus=cards)
        schedule(rt, [('fit', arm) for arm in spec.fit_arms], training_job)
        status(status='SOURCE_FROZEN', arms=list(spec.arms), steps=spec.steps)
        deadline = monotonic() + 600
        while not (cards := eligible()):
            if kind != PRACTICAL_KIND and monotonic() >= deadline:
                raise RuntimeError('No GPU available for frozen source evaluation')
            sleep(15)
        for phase, gpu in (('predict-source', cards[0]), ('score-source', None)):
            proc = start(phase, phase, gpu, reference, evaluator, runtime)
            status(status='RUNNING', phase=phase, pid=proc.pid, gpu=gpu)
            if proc.wait():
                raise RuntimeError(f'{phase} failed')
        results = [read(root / route / 'source-scores.json')['scores'] for route in spec.routes]
        promote = promotion(results, kind)
        fields = dict(formal_requested=True) if kind == PRACTICAL_KIND else {}
        status(status='SOURCE_SCORED', promote=promote, arms=list(spec.arms), steps=spec.steps, **fields)
        if promote or kind == PRACTICAL_KIND:
            if kind == PRACTICAL_KIND:
                frozen_source_results(root, kind)
            else:
                qualified_source(root, kind)
            config = root / 'formal-runtime.json'
            write(config, dict(rt, candidate_root=str(root), run_root=str(root/'formal'), ungated=True))
            dispatcher = own.parents[2] if kind == PRACTICAL_KIND else reference
            job = start('dispatch', 'formal-dispatch', None, dispatcher, own.with_name('w24_formal.py'), config)
            status(status='FORMAL_PREDICTING', pid=job.pid)
            if job.wait() or read(root/'formal'/'state.json')['status'] != 'PREDICTIONS_FROZEN':
                raise RuntimeError('Formal predictions did not freeze')
            status(status='PREDICTIONS_FROZEN', target_truth_read=False)
    except Exception as error:
        status(status='FAILED', error=str(error))
        raise


def publish(args):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.publish import release, release_commit
    from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN
    from experiments.pamf_lora_main.w24_formal import RUN_UNGATED
    from experiments.pamf_lora_main.w24_native_full import DATA_RUN, W24_RUN
    from experiments.pamf_lora_main.w24_native_residual_source import SWEEP_RUN
    from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN

    ctx = read(args.context)
    kind = args.kind
    run = trial(kind).run
    rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    releases, runs = project / 'releases', project / 'runs'
    pools = {'readout': [0, 3, 7], 'robust': [5, 6, 1, 7], 'final-radius': [0, 5, 6, 7],
             'peer': [3, 7, 6, 5], 'coherent-error': [0, 3, 5, 7], 'band-tune': [7, 6, 3, 0, 2],
             'band-adapter': [0, 3, 5, 7], 'band-refine': [6, 7, 2, 0, 3],
             'white-corruption': [0, 3, 6, 7, 2], 'band-precision': [0, 3, 6, 7],
             'half-primary': [0, 3, 5, 7],
             'severity-weight': [6, 5, 3, 0, 7, 2], 'carrier-anchor': [0, 3, 6, 7, 5, 2],
             'signed-probe': [0, 3, 7, 6, 2, 5], 'packet-unet': [0, 3, 7, 6, 2, 5],
             'packet-unet-fidelity': [2, 7, 6, 5], 'spectral-experts': [0, 3, 5, 7, 2],
             'chirp-components': [3, 2], 'probe-transport': [0, 3, 7], 'noise-color': [2, 3, 0],
             'common-refine': [0, 3, 2, 7], 'spectral-posterior': [0, 3, 2, 7],
             'clean-projection': [0, 2, 3, 5], 'weak-transfer': [0, 3, 2, 5],
             'polyphase': [0, 2, 3, 5], 'additive-reference': [0, 7, 2, 3],
             'native-packet-dura': [3, 5, 6], 'native-additive': [0, 3, 5, 6],
             'native-band-risk': [0, 3, 5, 6], 'native-center-direction': [0, 3, 5, 6],
             PRACTICAL_KIND: [0, 7, 2, 3, 6, 5, 1, 4]}
    rt.update(run_id=run, runner=__spec__.name, code_commit=release_commit(args.repo),
              release_root=str(releases / f'{run}-release'), run_root=str(runs / run),
              source_root=str(releases / f'{DATA_RUN}-release' / 'data/source'),
              native_target_root=str(releases / f'{DATA_RUN}-release' / 'data/target'),
              classifier_root=str(runs / AUG_RUN), w24_release_root=str(releases / f'{W24_RUN}-release'),
              training_release_root=str(releases / f'{TRAIN_RUN}-release'), third_prior=kind,
              cpu_workers=8, allowed_gpus=pools.get(kind, [5, 6, 1, 0]),
              max_workers=2, max_compute_processes=8)
    rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root',
                                               'training_release_root', 'native_target_root')] + [str(runs / name) for name in (SWEEP_RUN, RUN_UNGATED)]
    if OUTER_BASELINE in trial(kind).source_modes:
        rt['source_parent'] = str(runs / OUTER_PARENT)
        rt['external_releases'].append(rt['source_parent'])
    if UNET_BASELINE in trial(kind).source_modes:
        rt['unet_parent'] = str(runs / UNET_FIDELITY_RUN)
        rt['external_releases'].append(rt['unet_parent'])
    if kind == PRACTICAL_KIND:
        rt['dose_parent'] = str(runs / NATIVE_CENTER_RUN)
        rt['practical_cache'] = str(runs / PRACTICAL_CACHE_RUN / 'source_practical')
        rt.update(max_compute_processes=2, training_cap=True)
        rt['external_releases'].extend((rt['dose_parent'], rt['practical_cache']))
    release(args.repo, args.output, rt, args.ssh_config)


def collect_score(args):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.w24_formal import RUN_PRACTICAL_CPU
    from experiments.pamf_lora_main.w24_formal_score import score
    import json

    root = PurePosixPath(read(args.submit)['run_root'])
    kind = args.kind
    cpu = args.cpu_formal
    expected = RUN_PRACTICAL_CPU if cpu else trial(kind).run
    if (not re.fullmatch(r'/[A-Za-z0-9_./-]+', str(root)) or root.name != expected
            or (cpu and kind != PRACTICAL_KIND)):
        raise ValueError('Unexpected immutable N607 run root')
    connection = ['-F', str(args.ssh_config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    wait_hours = 48 if cpu else 12
    deadline = monotonic() + wait_hours*60*60
    last = None
    while True:
        state = None
        try:
            remote = subprocess.run(['ssh', *connection, '-T', 'N607', 'cat ' + str(root/'state.json')],
                                    timeout=25, capture_output=True)
            if remote.returncode == 0:
                state = json.loads(remote.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            # A short SSH failure or concurrent state write can interrupt a read.
            # Reobserve the same run without resubmitting it.
            pass
        if state is not None:
            if last != state['status']:
                print(state, flush=True)
                last = state['status']
            if state['status'] == 'FAILED':
                raise RuntimeError(state['error'])
            if state['status'] == 'SOURCE_SCORED' and not state['promote'] and kind != PRACTICAL_KIND:
                return
            if state['status'] == 'PREDICTIONS_FROZEN':
                break
        if monotonic() >= deadline:
            raise TimeoutError('No terminal N607 state in scoring window')
        sleep(90 if cpu else 40)
    args.output.mkdir(parents=True)
    predictions = (root if cpu else root/'formal')/'predictions'
    subprocess.run(['scp', *connection, '-r', 'N607:' + str(predictions), str(args.output)],
                   check=True, timeout=180)
    score(args.output/'predictions', args.dataset, args.output/'scores', third_prior=kind)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('dispatch').add_argument('runtime', type=Path)
    actions.add_parser('cache-practical').add_argument('runtime', type=Path)
    for phase in ('fit-smoke', 'fit', 'fit-arm'):
        cmd = actions.add_parser(phase)
        cmd.add_argument('runtime', type=Path)
        cmd.add_argument('arm', choices=ARM_TRIAL)
    pub = actions.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + key, type=Path, required=True)
    collector = actions.add_parser('collect-score')
    collector.add_argument('--cpu-formal', action='store_true')
    for command in (pub, collector):
        variants = command.add_mutually_exclusive_group()
        variants.set_defaults(kind=True)
        for kind in VARIANTS:
            variants.add_argument('--' + kind, dest='kind', action='store_const', const=kind)
    for key in ('submit', 'dataset', 'output', 'ssh-config'):
        collector.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        publish(args)
    elif args.action == 'cache-practical':
        from experiments.cric_dura_comparison.io import read
        from experiments.pamf_lora_main.w24_practical_supervision import cache_source

        cache_source(read(args.runtime))
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'collect-score':
        collect_score(args)
    elif args.action == 'fit-arm':
        import sys

        # Run the startup check in an independent process with a fresh optimizer.
        for action in ('fit-smoke', 'fit'):
            subprocess.run([sys.executable, '-u', __file__, action, str(args.runtime), args.arm], check=True)
    else:
        fit(args.runtime, args.arm, smoke=args.action == 'fit-smoke')
