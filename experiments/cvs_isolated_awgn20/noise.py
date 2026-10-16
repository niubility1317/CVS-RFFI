"""Isolated native AWGN on currentdevice;one stateless view,all energy tails retained."""
import inspect,sys
import torch
from . import design as d

def native_module():
 n=sys.modules['SSDG.train_ssdg'];a=d.make_args(d.config(d.rows()[0]),'cpu');return inspect.getmodule(type(n.make_augmentor(n.build_aug_base_cfg(a))))

def apply(x,step):
 if not isinstance(step,int) or step not in range(1,4441) or not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or x.dtype!=torch.float32 or x.requires_grad or x.device.type not in ('cpu','cuda') or not bool(torch.isfinite(x).all()):raise ValueError('Finite rawfloat32 IQ[B,2,256] and fixed optimizerstep required')
 px=x.double().square().sum((1,2))
 if not bool((px>0).all()):raise ValueError('Nonzero rawpacket power required,no filtering')
 module=native_module();devices=[x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
 with torch.random.fork_rng(devices=devices):
  torch.random.default_generator.manual_seed(d.NOISE_SEED+step)
  if devices:torch.cuda.default_generators[devices[0]].manual_seed(d.NOISE_SEED+step)
  snr=module._rand_uniform((len(x),1),x.device,low=d.SNR[0],high=d.SNR[1],dtype=x.dtype);value=module._from_complex(module._apply_awgn(module._to_complex(x),snr)).to(x.dtype)
 if value.shape!=x.shape or not bool(torch.isfinite(value).all()):raise ValueError('Finite fullnative noiseview required,no dropping')
 n=value.double()-x.double();pn=n.square().sum((1,2));py=value.double().square().sum((1,2))
 if not bool((pn>0).all()):raise ValueError('Actual nonzero noise,no redraw')
 return value,dict(seed=d.NOISE_SEED+step,nominal_SNR_dB=snr.view(-1).tolist(),raw_RMS=torch.sqrt(px/256).tolist(),noise_RMS=torch.sqrt(pn/256).tolist(),output_RMS=torch.sqrt(py/256).tolist(),relative_IQ_RMS_change=torch.sqrt(pn/px).tolist(),actual_added_noise_SNR_dB=(10*torch.log10(px/pn)).tolist(),relative_power_change=(py/px-1).tolist(),normalized_signal_noise_cross_term=(2*(x.double()*n).sum((1,2))/px).tolist())
