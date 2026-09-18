"""Direct post-compensation residual from the SAME practical latent channel.

No full-channel waveform is generated. Frozen-within-record approximation:
construct H, estimate Hhat, compute G, then apply the equivalent residual GH
once. Noise is colored by G; IQ image and untracked carrier terms survive.
This is not an independently weakened channel, and G=1 when EQ is off.
"""
from dataclasses import replace, asdict
import math
import numpy as np

from .channel import (ChannelStream, VERSION, STATES, C, slant_range,
    receiver_quality, iq_imbalance, compensate_iq, waveform_metrics)
from .equalization import propagation_kernel, design_equalizer, linear_filter


class ResidualChannel:
    """One independent short snapshot; not a continuous-orbit stream."""
    def __init__(self, cfg, seed, receiver, geometry=None, receiver_processor=None):
        if cfg.mode != "post_sync":
            raise ValueError("residual route requires post_sync")
        if receiver_processor is not None:
            raise ValueError("arbitrary receiver transforms are not supported")
        self.cfg, self.seed, self.receiver = cfg, seed, receiver
        # Only initialize shared physical parameters. NEVER call its process().
        self.latent = ChannelStream(replace(cfg, processing_route="full"), seed, receiver, geometry)
        self.used = False

    def process(self, x, *, preceding_context=None, return_traces=False, quality_snr_db=None):
        if self.used:
            raise ValueError("ResidualChannel is snapshot-only; create a new instance per record")
        cfg, latent = self.cfg, self.latent
        x = np.asarray(x, complex)
        if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
            raise ValueError("expected finite 1D raw IQ")
        n = len(x)
        if n/cfg.fs_hz > min(cfg.local_geometry_max_duration_s, .001):
            raise ValueError("residual frozen-channel approximation limited to 1ms snapshots")
        s = latent.state
        coefficients = latent.ray_coef.sum(axis=1)*np.sqrt(latent.diffuse[s]*latent.pdp)
        los = 0. if s == 2 else 10**((cfg.los_mean_db[s]+cfg.los_std_db[s]*latent.z)/20)
        coefficients[0] += los
        coefficients /= math.sqrt(latent.reference_power)
        ref = slant_range(cfg.reference_altitude_m, cfg.reference_elevation_deg)
        gain = ref/latent.geometry.distance_m * 10**(-cfg.atmosphere_loss_db/20)
        h = propagation_kernel(latent, s, coefficients, gain)
        input_power = float(np.mean(abs(x)**2))
        noise_variance = 10**(-cfg.reference_snr_db/10)
        # PSD-based expected power: no synthesis of H*x just to get quality.
        # White-input approximation is explicitly logged; external estimate wins.
        proxy_quality = float(10*np.log10(max(input_power*np.sum(abs(h)**2),1e-300)/noise_variance))
        quality = proxy_quality if quality_snr_db is None else float(quality_snr_db)
        status, scale = receiver_quality(cfg, quality)
        fhat = (latent.f_doppler+latent.coarse_doppler_error_hz if status == "unlocked"
                else latent.f_total-latent.residual_hz*scale)
        residual_frequency = latent.f_total-fhat
        phase_active = cfg.phase_tracking_enabled and status != "unlocked"
        eq_active = cfg.equalization_enabled and status != "unlocked"
        if eq_active:
            g, eq_meta = design_equalizer(h,cfg,latent.rngs["equalizer"],noise_variance,input_power,scale)
        else:
            g = np.ones(1,complex)
            eq_meta = {"reason":"disabled_by_default" if not cfg.equalization_enabled else "receiver_unlocked"}
        # Widely-linear residual of only the injected RX + its calibration.
        probe = iq_imbalance(np.array([1.,1j]), self.receiver)
        if cfg.iq_compensation_enabled:
            probe = compensate_iq(probe,self.receiver.estimated_amplitude_imbalance_db,
                                  self.receiver.estimated_quadrature_error_deg)
        alpha, beta = (probe[0]+probe[1]/1j)/2, (probe[0]-probe[1]/1j)/2
        t = np.arange(n)/cfg.fs_hz
        phase_inc = latent.rngs["phase"].normal(size=n)*math.sqrt(2*np.pi*cfg.phase_noise_linewidth_hz/cfg.fs_hz)
        phase = latent.phase+np.concatenate([[0.],np.cumsum(phase_inc[:-1])])
        rho = math.exp(-1/(cfg.fs_hz*cfg.phase_tracking_correlation_time_s))
        innovation = math.sqrt(-math.expm1(-2/(cfg.fs_hz*cfg.phase_tracking_correlation_time_s)))
        tracking = np.empty(n)
        z = latent.tracking_z
        for i,e in enumerate(latent.rngs["tracking"].normal(size=n)):
            tracking[i] = scale*math.radians(cfg.phase_tracking_residual_std_deg)*z
            z = rho*z+innovation*e
        main_phase = tracking if phase_active else phase
        image_phase = -2*phase+tracking if phase_active else -phase
        image_frequency = -(latent.f_total+fhat)
        main_rotation = np.exp(1j*(2*np.pi*residual_frequency*t+main_phase))
        image_rotation = np.exp(1j*(2*np.pi*image_frequency*t+image_phase))
        # Collapse H then G into one residual FIR per widely-linear branch.
        # Carrier modulation of G is exact for constant frequency. Fast phase
        # changes over the equalizer span are approximated as locally constant.
        offsets = np.arange(len(g))/cfg.fs_hz
        main_kernel = np.convolve(h,g*np.exp(-2j*np.pi*residual_frequency*offsets))
        image_kernel = np.convolve(h.conj(),g*np.exp(-2j*np.pi*image_frequency*offsets))
        main, _ = linear_filter(x,main_kernel,cfg.boundary,preceding_context)
        image_context = None if preceding_context is None else np.conj(preceding_context)
        mirror, _ = linear_filter(x.conj(),image_kernel,cfg.boundary,image_context)
        signal = alpha*main*main_rotation+beta*mirror*image_rotation
        draws = latent.rngs["noise"].normal(size=(n,2))
        w = math.sqrt(noise_variance/2)*(draws[:,0]+1j*draws[:,1])
        noise_input = alpha*w*main_rotation+beta*w.conj()*image_rotation
        # No propagation pass on noise, only post-receiver residual coloration.
        noise_boundary = cfg.boundary if cfg.boundary != "require_context" else "edge"
        noise, _ = linear_filter(noise_input,g,noise_boundary)
        y = signal+noise
        rms = float(np.sqrt(np.mean(abs(y)**2)))
        requested = -20*np.log10(max(rms,1e-300))
        agc_db = float(np.clip(requested,-cfg.agc_limit_db,cfg.agc_limit_db)) if cfg.agc_enabled else 0.
        y *= 10**(agc_db/20)
        if not np.isfinite(y).all():
            raise FloatingPointError("nonfinite residual output")
        self.used = True
        ps,pn = float(np.mean(abs(signal)**2)),float(np.mean(abs(noise)**2))
        meta = dict(version=VERSION,config_hash=cfg.config_hash,seed=int(self.seed),
            scenario=cfg.scenario,processing_route="residual",full_channel_waveform_synthesized=False,
            residual_model="frozen_practical_GH_widely_linear_proxy",geometry=asdict(latent.geometry),
            receiver=asdict(self.receiver),receiver_location=cfg.receiver_location,
            link_direction="ground_to_satellite" if cfg.receiver_location=="satellite" else "satellite_to_ground",
            fs_hz=cfg.fs_hz,fc_hz=cfg.fc_hz,mode=cfg.mode,
            state_start=STATES[s],state_end=STATES[s],state_counts={name:n if i==s else 0 for i,name in enumerate(STATES)},
            state_probabilities=latent.probs.tolist(),receiver_lock_status=status,quality_snr_db=quality,
            quality_source="white_input_expected_power_proxy" if quality_snr_db is None else "caller_supplied_receiver_quality",
            residual_scale=scale,orbital_doppler_hz=latent.f_doppler,estimated_frequency_hz=fhat,
            output_frequency_hz=residual_frequency,frequency_compensation_applied=True,
            iq_compensation_applied=cfg.iq_compensation_enabled,phase_tracking_applied=phase_active,
            fine_frequency_tracking_applied=status!="unlocked",coarse_doppler_compensation_applied=True,
            channel_equalization_applied=eq_active,equalizer=eq_meta,tx_impairment_correction_applied=False,
            propagation_kernel_length=len(h),residual_kernel_length=len(main_kernel),
            residual_kernel_energy=float(np.sum(abs(main_kernel)**2)),
            numerical_filter_latency_samples=cfg.fractional_delay_half_length,
            equalizer_target_total_delay_samples=cfg.equalizer_delay_samples if eq_active else None,
            boundary="provided_context" if preceding_context is not None else "synthetic_"+cfg.boundary,
            noise_boundary="synthetic_"+noise_boundary,
            added_noise_variance_reference=noise_variance,measured_added_noise_power_pre_agc=pn,
            signal_to_added_noise_db=float(10*np.log10(ps/pn)) if ps>0 and pn>0 else None,
            true_total_snr_known=False,agc_gain_db=agc_db,
            agc_limited=bool(cfg.agc_enabled and abs(requested)>cfg.agc_limit_db),
            input_metrics=waveform_metrics(x),output_metrics=waveform_metrics(y),
            assumptions=["frozen_state_shadow_scatter_geometry_within_record",
                         "phase_locally_constant_over_EQ_memory",
                         "quality_proxy_may_differ_from_full_route_on_colored_input"],
            parameter_provenance="same_practical_engineering_parameters_not_independent_weak_channel")
        if return_traces:
            return y.astype(np.complex64),meta,dict(pre_agc_signal=signal,pre_agc_noise=noise,
                propagation_kernel=h,residual_kernel=main_kernel,equalizer_kernel=g)
        return y.astype(np.complex64),meta
