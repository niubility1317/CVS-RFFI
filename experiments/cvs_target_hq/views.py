"""Replay the canonical target realization; intervene only on added noise."""
import numpy as np
from experiments.cvs_urban_bridge.views import coupled
from experiments.cvs_urban_sigtime.channel import imports
from . import design as d

def generate(clean,canonical,pid,manifest):
    ch,_=imports()
    item=next(v for v in manifest['evidence'] if v['scene']==d.SCENE)
    cfg=ch.Config(**item['config'])
    if (cfg.processing_route!='residual' or cfg.mode!='post_sync' or cfg.equalization_enabled
            or cfg.reference_snr_db!=30 or cfg.fs_hz!=25e6):
        raise ValueError('Original practical low urban configuration differs')
    raw=np.asarray(clean,dtype=np.float64)
    if raw.shape!=(2,256) or not np.isfinite(raw).all():raise ValueError('Finite original target IQ required')
    x,_=ch.normalize_input(raw[0]+1j*raw[1])
    seed=ch.stable_seed(manifest['seed'],manifest['namespace'],d.SCENE,str(pid))
    receiver=ch.receiver_for_session(cfg,manifest['receiver_seed'],manifest['session'])
    waves,meta,proof,_=coupled(x,cfg,seed,receiver,noise_improvement_db=6.)
    control,meta0,proof0,_=coupled(x,cfg,seed,receiver,noise_improvement_db=0.)
    if not np.array_equal(waves['full'],control['full']) or meta!=meta0:
        raise ValueError('Noise-dose controls changed the original full realization')
    as_iq=lambda value:np.stack((value.real,value.imag)).astype(np.float32)
    replay=as_iq(waves['full']);canonical=np.asarray(canonical,dtype=np.float32)
    if canonical.shape!=(2,256) or not np.isfinite(canonical).all():raise ValueError('Fixed full target IQ differs')
    error=float(np.max(abs(replay-canonical)))
    if not np.allclose(replay,canonical,rtol=2e-6,atol=2e-6):
        raise ValueError('Canonical target replay mismatch; preserve run and do not weaken controls')
    outputs=dict(zip(d.VIEWS,(canonical.copy(),as_iq(control['bridge_independent']),
        as_iq(waves['bridge_independent']),as_iq(waves['bridge_shared']))))
    metadata=dict(id=str(pid),state=meta['state_end'],lock=meta['receiver_lock_status'],
        added_noise_snr_db=meta['signal_to_added_noise_db'],receiver_quality_proxy_db=meta['quality_snr_db'],
        residual_CFO_hz=meta['output_frequency_hz'],residual_kernel_length=meta['residual_kernel_length'],
        AGC_gain_db=meta['agc_gain_db'],AGC_limited=meta['agc_limited'],full_replay_max_abs_error=error,
        independent_noise_seed=proof['independent_noise_seed'],
        independent_HQ_added_noise_snr_db=proof['bridge_independent_signal_to_added_noise_db'],
        independent_samelevel_added_noise_snr_db=proof0['bridge_independent_signal_to_added_noise_db'],
        receiver_actions_and_kernels_frozen=True,output_RMS_renormalized=False,
        original_recording_noise_removed=False,total_SNR_known=False)
    return outputs,metadata
