"""Condition existing residual processing on the same raw finite-window H*x power."""
import math
import numpy as np

def latent_kernel(channel):
 from leo_practical.channel import slant_range
 from leo_practical.equalization import propagation_kernel
 latent=channel.latent;cfg=channel.cfg;s=latent.state
 coefficients=latent.ray_coef.sum(axis=1)*np.sqrt(latent.diffuse[s]*latent.pdp)
 los=0. if s==2 else 10**((cfg.los_mean_db[s]+cfg.los_std_db[s]*latent.z)/20)
 coefficients[0]+=los;coefficients/=math.sqrt(latent.reference_power)
 ref=slant_range(cfg.reference_altitude_m,cfg.reference_elevation_deg)
 gain=ref/latent.geometry.distance_m*10**(-cfg.atmosphere_loss_db/20)
 return propagation_kernel(latent,s,coefficients,gain)
def process(channel,x,return_traces=False):
 from leo_practical.equalization import linear_filter
 if channel.used or channel.cfg.equalization_enabled:raise ValueError('Single unused residual/noEQ realization required')
 x=np.asarray(x,dtype=np.complex128)
 if x.shape!=(256,) or not np.isfinite(x).all():raise ValueError('Original finite IQ256 required')
 h=latent_kernel(channel);filtered,_=linear_filter(x,h,channel.cfg.boundary)
 noise_variance=10**(-channel.cfg.reference_snr_db/10)
 power=float(np.mean(abs(filtered)**2));quality=float(10*np.log10(max(power,1e-300)/noise_variance))
 value=channel.process(x,quality_snr_db=quality,return_traces=return_traces)
 y,meta=value[:2]
 meta.update(quality_input='finite_window_Hx_to_same_added_noise_floor',quality_prefilter_calls=1,quality_prefilter_power=power,quality_white_expected_db=float(10*np.log10(max(float(np.mean(abs(x)**2)*np.sum(abs(h)**2)),1e-300)/noise_variance)),no_complete_channel_prepass=True)
 if return_traces and not np.array_equal(h,value[2]['propagation_kernel']):raise ValueError('Power prefilter differs from exact current propagation')
 return value
