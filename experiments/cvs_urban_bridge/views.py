"""Change added noise inside a frozen practical realization, never test IQ."""
import math
import numpy as np
from experiments.cvs_urban_sigtime.channel import imports

FROZEN_FIELDS=('geometry','receiver','state_start','state_end','state_probabilities',
    'receiver_lock_status','quality_snr_db','quality_source','residual_scale',
    'estimated_frequency_hz','output_frequency_hz','phase_tracking_applied',
    'channel_equalization_applied','iq_compensation_applied','residual_kernel_length',
    'residual_kernel_energy','numerical_filter_latency_samples','boundary','noise_boundary')


def coupled(x,cfg,seed,receiver,*,noise_improvement_db=6.,independent_seed=None,
            quality_snr_db=None,preceding_context=None):
    """Return full, shared/independent HQ, and fixed-state no-added-noise controls.

    Positive real scaling commutes with the original widely-linear RX/rotation/G
    noise operator. Every output uses the *full* realization's AGC gain. No
    original-recording noise is removed. There is no output RMS renormalization.
    """
    if (cfg.processing_route!='residual' or cfg.mode!='post_sync'
            or not math.isfinite(noise_improvement_db) or not 0<=noise_improvement_db<=30):
        raise ValueError('Bounded source residual/post_sync noise intervention required')
    ch,residual=imports()
    if independent_seed is None:independent_seed=ch.stable_seed(seed,'bridge_independent_added_noise')
    full,meta,traces=residual.ResidualChannel(cfg,seed,receiver).process(x,
        quality_snr_db=quality_snr_db,preceding_context=preceding_context,return_traces=True)
    second=residual.ResidualChannel(cfg,seed,receiver)
    second.latent.rngs['noise']=np.random.default_rng(independent_seed)
    _,other,independent=second.process(x,quality_snr_db=quality_snr_db,
        preceding_context=preceding_context,return_traces=True)
    for key in FROZEN_FIELDS:
        if meta[key]!=other[key]:raise ValueError('Independent noise changed receiver/propagation field: '+key)
    for key in ('pre_agc_signal','propagation_kernel','residual_kernel','equalizer_kernel'):
        if not np.array_equal(traces[key],independent[key]):
            raise ValueError('Independent noise changed frozen signal/operator: '+key)
    signal=traces['pre_agc_signal'];noise=traces['pre_agc_noise']
    gain=10**(meta['agc_gain_db']/20);amplitude=10**(-noise_improvement_db/20)
    reconstructed=((signal+noise)*gain).astype(np.complex64)
    if not np.array_equal(full,reconstructed):raise ValueError('Full canonical trace reconstruction differs')
    waves=dict(full=full,bridge_shared=((signal+amplitude*noise)*gain).astype(np.complex64),
        bridge_independent=((signal+amplitude*independent['pre_agc_noise'])*gain).astype(np.complex64),
        no_added_noise=(signal*gain).astype(np.complex64))
    powers={key:float(np.mean(abs(v)**2)) for key,v in dict(signal=signal,
        full_noise=noise,independent_noise=independent['pre_agc_noise']).items()}
    def snr(power):return 10*math.log10(powers['signal']/power) if powers['signal']>0 and power>0 else None
    details=dict(noise_improvement_db=float(noise_improvement_db),noise_amplitude=amplitude,
        full_signal_to_added_noise_db=snr(powers['full_noise']),
        bridge_shared_signal_to_added_noise_db=snr(amplitude**2*powers['full_noise']),
        bridge_independent_signal_to_added_noise_db=snr(amplitude**2*powers['independent_noise']),
        independent_noise_seed=int(independent_seed),frozen_fields=list(FROZEN_FIELDS),
        frozen_fields_identical=True,frozen_kernels_identical=True,
        full_AGC_gain_db=meta['agc_gain_db'],full_AGC_limited=meta['agc_limited'],
        independent_native_AGC_gain_ignored_db=other['agc_gain_db'],
        AGC_policy='original full-view gain and gain-limit decision; not recomputed for HQ',
        sample_hard_clipping_present=False,output_RMS_renormalized=False,
        receiver_state_quality_is_full_view=True,reference_snr_db_unchanged=cfg.reference_snr_db,
        original_recording_noise_removed=False,true_total_SNR_known=False,
        EQ_noise_coloration_preserved=True,full_waveform_exact=True)
    if any(not np.isfinite(v).all() for v in waves.values()):raise FloatingPointError('Nonfinite coupled view')
    return waves,meta,details,traces
