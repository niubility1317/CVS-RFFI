"""Native isolated proper-complex AWGN on raw;no labels,clipping or RMS restoration."""
import torch
from . import design as d

def generate(x,native):
 if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or x.dtype!=torch.float32 or x.device.type!='cpu' or x.requires_grad or not bool(torch.isfinite(x).all()):raise ValueError('Finite CPU sourcefloat32 IQ[B,2,256] withoutautograd required')
 px=x.double().square().sum((1,2))
 if not bool((px>0).all()):raise ValueError('Nonzero rawpacket power required,no filtering')
 snr=native._rand_uniform((len(x),1),x.device,low=d.SNR[0],high=d.SNR[1],dtype=x.dtype);z=native._to_complex(x);out=native._from_complex(native._apply_awgn(z,snr)).to(x.dtype)
 if out.shape!=x.shape or not bool(torch.isfinite(out).all()):raise ValueError('Finite matching nativeAWGN output required')
 n=out.double()-x.double();pn=n.square().sum((1,2));py=out.double().square().sum((1,2))
 if not bool((pn>0).all()):raise ValueError('Nonzero actual noise,no redraw')
 info=dict(nominal_SNR_dB=snr.view(-1).tolist(),raw_RMS=torch.sqrt(px/256).tolist(),noise_RMS=torch.sqrt(pn/256).tolist(),output_RMS=torch.sqrt(py/256).tolist(),relative_IQ_RMS_change=torch.sqrt(pn/px).tolist(),actual_added_noise_SNR_dB=(10*torch.log10(px/pn)).tolist(),relative_power_change=(py/px-1).tolist(),normalized_signal_noise_cross_term=(2*(x.double()*n).sum((1,2))/px).tolist())
 return out,info
