import math
import torch
from experiments.cvs_rx_spectral_audit.spectral import observations,grid,fit_contrast

def basis(n,device,dtype):
 pos=(torch.arange(n,device=device,dtype=dtype)+.5)/n;k=torch.arange(1,9,device=device,dtype=dtype)
 return torch.cos(math.pi*pos[:,None]*k[None,:])*math.sqrt(2/32)

def fit(L):
 values=observations(L['x'],'rectangular')['DCT1to8'];means,_,labels=grid(values,L['y'],L['rx'],L['day'])
 effects=torch.stack([torch.stack([fit_contrast(means,t,d) for d in range(3)]) for t in range(6)])
 return dict(effects=effects.float(),TX=torch.tensor(labels['TX']),day=torch.tensor(labels['day']),RX=torch.tensor(labels['RX']),labels=labels)

def coefficients(state,y,rx,day,shift,kind):
 if shift not in (1,2,3,4):raise ValueError('Must target another source RX')
 t=torch.searchsorted(state['TX'],y);r=torch.searchsorted(state['RX'],rx);d=torch.searchsorted(state['day'],day)
 if not torch.equal(state['TX'][t],y) or not torch.equal(state['RX'][r],rx) or not torch.equal(state['day'][d],day):raise ValueError('Unknown source metadata')
 delta=state['effects'][t,d,(r+shift)%5]-state['effects'][t,d,r]
 if kind=='rotated':delta=delta.roll(4,-1)
 elif kind!='physical':raise ValueError('Unknown action kind')
 return delta

def fir(delta,scale,taps,max_amplitude_dB=3.):
 if delta.ndim!=2 or delta.shape[1]!=8 or taps not in (17,33) or scale not in (0.,.5,1.):raise ValueError('Unregistered finite IQ action')
 if not torch.isfinite(delta).all():raise ValueError('Nonfinite response')
 # delta is a log POWER change; amplitude exponent therefore has a factor1/2.
 desired_log_amplitude=.5*scale*(delta@basis(256,delta.device,delta.dtype).T)
 limit=max_amplitude_dB*math.log(10)/20
 clipped=desired_log_amplitude.clamp(-limit,limit)
 response=torch.fft.ifftshift(clipped.exp(),dim=-1)
 impulse=torch.fft.ifft(response);half=taps//2;offset=torch.arange(-half,half+1,device=delta.device)
 h=impulse[:,offset.remainder(256)]*torch.hann_window(taps,periodic=False,device=delta.device,dtype=delta.dtype)
 # Response of the centered, truncated filter on the original256 FFT grid.
 sequence=torch.zeros(len(h),256,dtype=h.dtype,device=h.device);sequence[:,offset.remainder(256)]=h
 effective=torch.fft.fftshift(torch.fft.fft(sequence),dim=-1)
 target=(2*clipped);realized=2*effective.abs().clamp_min(1e-8).log()
 target-=target.mean(-1,keepdim=True);realized-=realized.mean(-1,keepdim=True)
 return h,dict(target_log_power=target,effective_log_power=realized,clipped_fraction=(desired_log_amplitude.abs()>limit).float().mean(1),effective_unshifted=torch.fft.ifftshift(effective,dim=-1))

def apply(x,delta,scale,taps,max_amplitude_dB=3.):
 if x.ndim!=3 or tuple(x.shape[1:])!=(2,256) or len(x)!=len(delta):raise ValueError('Wrong IQ/action shape')
 h,info=fir(delta,scale,taps,max_amplitude_dB);z=torch.complex(x[:,0],x[:,1]);half=taps//2
 # 512 >=256+taps-1: no circular wraparound contaminates this convolution.
 full=torch.fft.ifft(torch.fft.fft(z,n=512)*torch.fft.fft(h,n=512))
 out=full[:,half:half+256]
 old_power=z.abs().square().mean(1);new_power=out.abs().square().mean(1)
 if (old_power<=0).any() or (new_power<=0).any():raise ValueError('Zero-energy IQ/filter')
 gain=(old_power/new_power).sqrt();circular=torch.fft.ifft(torch.fft.fft(z)*info.pop('effective_unshifted'))
 residual=(out-circular).abs().square();norm=old_power.clamp_min(1e-12)
 info.update(boundary_residual_fraction=torch.cat([residual[:,:half],residual[:,-half:]],1).mean(1)/norm,interior_residual_fraction=residual[:,half:-half].mean(1)/norm,normalization_gain=gain)
 out*=gain[:,None]
 result=torch.stack([out.real,out.imag],1).to(x.dtype)
 if not torch.isfinite(result).all():raise ValueError('Nonfinite transformed IQ')
 return result,info
