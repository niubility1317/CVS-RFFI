"""IQ only additive observation noise;private RNG,no TX input or hardware inverse."""
import hashlib,math
import torch
NOISE=dict(SNR_added_db=30.,impulse_probability=.02,impulse_excess_variance=49.,per_packet_noise_RMS_exact=True,post_input_normalization=False,seed=202610185301,parameters_TX_independent=True,no_waveform_clipping=True)
MODES=('NONE','WHITE','BG')
def apply(x,step,mode):
 if mode not in MODES or type(step)is not int or step<1 or x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not len(x) or not bool(torch.isfinite(x).all()):raise ValueError('Finite float32 Bx2x256 trueL IQ and positive cumulative step only')
 if mode=='NONE':return x.clone(),dict(mode=mode,packets=len(x),applied=0,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None,new_boundary_samples=0)
 seed=int.from_bytes(hashlib.sha256(f"{NOISE['seed']}/noise43/{step}".encode()).digest()[:8],'little')%(2**63-1);g=torch.Generator(device='cpu').manual_seed(seed)
 eps=torch.randn(x.shape,generator=g,dtype=torch.float32)
 mask=torch.rand((len(x),1,256),generator=g)<NOISE['impulse_probability']
 if mode=='BG':eps=eps*torch.sqrt(1+NOISE['impulse_excess_variance']*mask)
 eps=eps.to(x.device);energy=eps.square().sum((1,2),keepdim=True);power=x.square().sum(1).mean(1).reshape(-1,1,1);delta=eps*torch.sqrt((256*power*10**(-NOISE['SNR_added_db']/10))/energy.clamp_min(torch.finfo(torch.float32).tiny))
 out=x+delta;realized=(out-x).square().sum(1).mean(1);expected=power.flatten()*10**(-NOISE['SNR_added_db']/10)
 if not bool(torch.isfinite(out).all()) or not torch.allclose(realized,expected,rtol=2e-5,atol=1e-10):raise FloatingPointError('Added-noise energy differs;no input RMS adjustment')
 peaks=delta.square().sum(1);relative=torch.sqrt(delta.square().sum((1,2))/x.square().sum((1,2)).clamp_min(1e-30))
 return out,dict(mode=mode,packets=len(x),applied=len(x),relative_noise_RMS=float(relative.mean()),noise_peak_fraction_mean=float((peaks.max(1).values/peaks.sum(1).clamp_min(1e-30)).mean()),SNR_added_db=NOISE['SNR_added_db'],total_SNR_unknown=True,impulse_count=int(mask.sum()) if mode=='BG' else 0,new_boundary_samples=0)
