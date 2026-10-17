"""Source-only interventions on a shared latent draw. Never a test generator."""
from dataclasses import replace
import math
import numpy as np
from .kernel import flat_kernel
def imports():
 import leo_practical.channel as ch
 import leo_practical.residual as r
 return ch,r

def render(x,cfg,seed,receiver,*,frozen,delay_samples,zero_phase):
    control='full';boundary=None
    if cfg.processing_route!='residual' or cfg.equalization_enabled or delay_samples not in (0,24) or not isinstance(zero_phase,bool) or frozen is None:raise ValueError('Fixed G1 flat-source controls only')
    ch,r=imports()
    from leo_practical.equalization import propagation_kernel,design_equalizer,linear_filter
    latent=ch.ChannelStream(replace(cfg,processing_route='full'),seed,receiver)
    state=latent.state;n=len(x);x=np.asarray(x,complex)
    coefficients=latent.ray_coef.sum(axis=1)*np.sqrt(latent.diffuse[state]*latent.pdp)
    los=0. if state==2 else 10**((cfg.los_mean_db[state]+cfg.los_std_db[state]*latent.z)/20)
    coefficients[0]+=los;coefficients/=math.sqrt(latent.reference_power)
    gain=ch.slant_range(cfg.reference_altitude_m,cfg.reference_elevation_deg)/latent.geometry.distance_m*10**(-cfg.atmosphere_loss_db/20)
    h=propagation_kernel(latent,state,coefficients,gain)
    h=flat_kernel(h,delay_samples,zero_phase)
    power=float(np.mean(abs(x)**2));noise_variance=10**(-cfg.reference_snr_db/10)
    white_power=power*float(np.sum(abs(h)**2));b=boundary or cfg.boundary
    propagated,_=linear_filter(x,h,b);actual_power=float(np.mean(abs(propagated)**2))
    proxy=10*np.log10(max(white_power,1e-300)/noise_variance)
    actual_quality=10*np.log10(max(actual_power,1e-300)/noise_variance)
    quality=actual_quality if control=='actual_power_quality' else float(proxy)
    if control=='no_noise':quality=200. # Counterfactual added-noise quality; original recording noise remains unknown.
    status,scale=ch.receiver_quality(cfg,quality)
    if frozen is not None:status,scale=frozen['receiver_lock_status'],frozen['residual_scale']
    fhat=(latent.f_doppler+latent.coarse_doppler_error_hz if status=='unlocked' else latent.f_total-latent.residual_hz*scale)
    frequency=latent.f_total-fhat;phase_active=cfg.phase_tracking_enabled and status!='unlocked';eq_active=cfg.equalization_enabled and status!='unlocked'
    if frozen is not None:
        fhat=frozen['estimated_frequency_hz'];frequency=frozen['output_frequency_hz'];phase_active=frozen['phase_tracking_applied'];eq_active=frozen['channel_equalization_applied']
    if eq_active:g,eq_meta=design_equalizer(h,cfg,latent.rngs['equalizer'],noise_variance,power,scale)
    else:g=np.ones(1,complex);eq_meta={}
    if frozen is not None and '_equalizer' in frozen:g=frozen['_equalizer']
    probe=ch.iq_imbalance(np.array([1.,1j]),receiver)
    if cfg.iq_compensation_enabled:probe=ch.compensate_iq(probe,receiver.estimated_amplitude_imbalance_db,receiver.estimated_quadrature_error_deg)
    alpha,beta=(probe[0]+probe[1]/1j)/2,(probe[0]-probe[1]/1j)/2
    if control=='no_RX_IQ':alpha,beta=1.,0.
    t=np.arange(n)/cfg.fs_hz
    increments=latent.rngs['phase'].normal(size=n)*math.sqrt(2*np.pi*cfg.phase_noise_linewidth_hz/cfg.fs_hz)
    phase=latent.phase+np.concatenate([[0.],np.cumsum(increments[:-1])])
    rho=math.exp(-1/(cfg.fs_hz*cfg.phase_tracking_correlation_time_s));innovation=math.sqrt(-math.expm1(-2/(cfg.fs_hz*cfg.phase_tracking_correlation_time_s)))
    tracking=np.empty(n);z=latent.tracking_z
    for i,e in enumerate(latent.rngs['tracking'].normal(size=n)):
        tracking[i]=scale*math.radians(cfg.phase_tracking_residual_std_deg)*z;z=rho*z+innovation*e
    main_phase=tracking if phase_active else phase;image_phase=-2*phase+tracking if phase_active else -phase
    image_frequency=-(latent.f_total+fhat)
    if control=='no_carrier':frequency=0.;image_frequency=0.;main_phase=np.zeros(n);image_phase=np.zeros(n)
    main_rotation=np.exp(1j*(2*np.pi*frequency*t+main_phase));image_rotation=np.exp(1j*(2*np.pi*image_frequency*t+image_phase))
    offsets=np.arange(len(g))/cfg.fs_hz
    kernel=np.convolve(h,g*np.exp(-2j*np.pi*frequency*offsets));image_kernel=np.convolve(h.conj(),g*np.exp(-2j*np.pi*image_frequency*offsets))
    main,_=linear_filter(x,kernel,b);mirror,_=linear_filter(x.conj(),image_kernel,b)
    signal=alpha*main*main_rotation+beta*mirror*image_rotation
    draws=latent.rngs['noise'].normal(size=(n,2));w=math.sqrt(noise_variance/2)*(draws[:,0]+1j*draws[:,1])
    noise_input=alpha*w*main_rotation+beta*w.conj()*image_rotation
    noise,_=linear_filter(noise_input,g,b)
    if control=='no_noise':noise=np.zeros_like(noise)
    y=signal+noise;rms=float(np.sqrt(np.mean(abs(y)**2)));requested=-20*np.log10(max(rms,1e-300))
    agc=float(np.clip(requested,-cfg.agc_limit_db,cfg.agc_limit_db)) if cfg.agc_enabled else 0.
    if frozen is not None:agc=frozen['agc_gain_db']
    y=y*10**(agc/20)
    ps,pn=float(np.mean(abs(signal)**2)),float(np.mean(abs(noise)**2))
    fftg=abs(np.fft.fft(g,1024))**2;positive=fftg[fftg>1e-20]
    return y.astype(np.complex64),dict(control=control,effect='fixed_receiver_state_and_compensation' if frozen is not None else 'recomputed_receiver_state',
        state=ch.STATES[state],receiver_lock_status=status,residual_scale=scale,quality_snr_db=float(quality),
        white_proxy_quality_db=float(proxy),actual_waveform_quality_db=float(actual_quality),
        waveform_minus_white_power_db=float(actual_quality-proxy),output_frequency_hz=float(frequency),estimated_frequency_hz=float(fhat),phase_tracking_applied=bool(phase_active),channel_equalization_applied=bool(eq_active),
        signal_to_added_noise_db=float(10*np.log10(ps/pn)) if ps>0 and pn>0 else None,true_total_SNR_known=False,
        noise_coloration_dynamic_range_db=float(10*np.log10(positive.max()/positive.min())) if len(positive) else None,
        agc_gain_db=float(agc),agc_limited=bool(frozen['agc_limited']),counterfactual_requested_agc_limited=bool(abs(requested)>cfg.agc_limit_db),
        residual_kernel_energy=float(np.sum(abs(kernel)**2)),kernel_length=len(kernel),
        numerical_delay=delay_samples,synthetic_prefix_affected_outputs=delay_samples,
        boundary=b,interior_begin=delay_samples,interior_count=n-delay_samples,
        reference_received_window=[80,160],reference_input_window_due_to_numerical_delay=[80-delay_samples,160-delay_samples],
        _equalizer=g,_signal=signal,_noise=noise)
