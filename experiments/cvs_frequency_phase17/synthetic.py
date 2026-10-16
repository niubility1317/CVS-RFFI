"""Analytic multitone observations;truth never passed to the IQ estimator."""
import numpy as np
from . import design as d
Q=np.array(d.HARMONICS,dtype=float);OM=2*np.pi*Q/20
def response(a,w):
 e=np.exp(-1j*w);return (np.asarray(a)[...,None]+e)/(1+np.asarray(a)[...,None]*e)
def closure(h):
 j=np.array([d.HARMONICS.index(k) for k in d.CENTERS]);v=h[...,j-1]*h[...,j+1]*h[...,j].conj()**2;return v/np.abs(v)
def centered_phase(z):
 a=np.angle(z*z.mean(-2,keepdims=True).conj());return a-a.mean(-2,keepdims=True)
def generate(template,snr,case,seed,*,packets=70):
 if template not in d.TEMPLATES or case not in d.CASES or snr not in (*d.SNRS,None) or not isinstance(seed,int) or not 1<=packets<=70:raise ValueError('Fixed synthetic family and valid seed/packets required')
 rng=np.random.default_rng(seed);shape=(6,3,4,packets);n=np.prod(shape);t=np.repeat(np.arange(6),3*4*packets);day=np.tile(np.repeat(np.arange(3),4*packets),6);rx=np.tile(np.repeat(np.arange(4),packets),18)
 base={'flat12':np.zeros(12),'qpsk12':np.pi/2*np.array([0,1,3,2,1,0,3,2,0,1,2,3]),'chirp12':np.pi*Q**2/7}[template]
 amp=.95+.15*np.cos((t[:,None]+1)*Q[None,:]/4);phase=base[None,:]+.15*np.cos(np.pi*(t[:,None]+1)*Q[None,:]/7)+.25*(t[:,None]-2.5)/2.5*(Q[None,:]/6)**2
 coef=amp*np.exp(1j*phase)/np.sqrt(12)
 if case=='TX_CURVATURE':coef*=response(np.linspace(-.45,.45,6)[t],OM[None,:])
 if case=='DEGENERATE':coef[:,d.HARMONICS.index(3)]=0
 true=response(np.array(d.ALLPASS),OM[None,:]);expected=centered_phase(closure(true)) if case=='RX_LTI' else np.zeros((4,8))
 cfo=(rx-1.5)*60000+(t-2.5)*2000+rng.uniform(-5000,5000,n);tau=np.array([-5.25,-1.75,2.25,6.5])[rx]+np.array([-.5,0,.5])[day]+rng.uniform(-.5,.5,n);clock=1+np.array([-80,40,-40,80])[rx]*np.array([1,-.5,.75])[day]*1e-6
 gain=np.exp(rng.uniform(-.2,.2,n)+.1*(rx-1.5));rotation=rng.uniform(-np.pi,np.pi,n)
 if case in ('RX_LTI','TX_RX_INTERACTION'):
  h=response(np.array(d.ALLPASS)[rx],OM[None,:]+2*np.pi*(.5*cfo[:,None])/d.FS)
  if case=='TX_RX_INTERACTION':h=np.where((t%2==0)[:,None],h,h.conj())
  coef*=h
 elif case=='RX_GAIN_ONLY':coef*=np.exp(.115*((rx-1.5)/1.5)[:,None]*(Q[None,:]/6))
 coef*=np.exp(-1j*clock[:,None]*tau[:,None]*OM[None,:]);z=np.empty((n,256),dtype=np.complex128)
 for r in range(4):
  for dd in range(3):
   mask=(rx==r)&(day==dd);basis=np.exp(1j*clock[mask][0]*np.arange(256)[:,None]*OM[None,:]);z[mask]=coef[mask]@basis.T
 z*=gain[:,None]*np.exp(1j*(rotation[:,None]+2*np.pi*cfo[:,None]*np.arange(256)[None,:]/d.FS))
 power=np.mean(np.abs(z[:,80:160])**2,axis=1);white=(rng.standard_normal(z.shape)+1j*rng.standard_normal(z.shape))/np.sqrt(2);edges=(rng.standard_normal(z.shape)+1j*rng.standard_normal(z.shape))/np.sqrt(2)
 noise=white*np.sqrt(power[:,None]/(10**(snr/10))) if snr is not None else np.zeros_like(z)
 observed=z+noise
 if case=='EDGE_ONLY':
  mask=(np.arange(256)<80)|(np.arange(256)>=160);observed[:,mask]=edges[:,mask]*np.sqrt(power[:,None])*(2+rx[:,None])
 scale=np.sqrt(np.mean(np.abs(observed)**2,axis=1));observed=(observed/scale[:,None]).astype(np.complex64)
 measured=10*np.log10(power/np.mean(np.abs(noise[:,80:160])**2,axis=1)) if snr is not None else None
 truth=dict(expected_RX_phase=expected.tolist(),CFO_Hz=cfo,synthetic_packets=int(n),synthetic_complex_points=int(n*256),snr_measured_summary=None if measured is None else dict(min=float(measured.min()),mean=float(measured.mean()),max=float(measured.max())),family='Steady-state analytic LTI on fixed12tones;outside-reference locality stress is not a proof about realfilter transients',CFO_before_RX_fraction=.5)
 return observed,truth
def score(est,truth,case):
 if not est['identifiable']:
  rejection=1-est['valid_packets']/truth['synthetic_packets'];passed=case=='DEGENERATE' and rejection>=.99
  return dict(passed=bool(passed),expected='reject' if case=='DEGENERATE' else 'identify',rejected_fraction=rejection,profile_relative_RMSE=None,null_profile_RMS_rad=None)
 p=np.array(est['profile']);expected=np.array(truth['expected_RX_phase']);den=float(np.mean(expected**2));rel=float(np.sqrt(np.mean((p-expected)**2)/den)) if den>1e-18 else None
 if case=='RX_LTI':passed=est['detected_RX'] and rel<=.25
 elif case=='TX_RX_INTERACTION':passed=(not est['detected_RX']) and est['crossfit_risk_improvement'] is not None and est['crossfit_risk_improvement']<=0 and est['profile_RMS_rad']<=.01
 elif case=='DEGENERATE':passed=False
 else:passed=(not est['detected_RX']) and est['profile_RMS_rad']<=.01
 return dict(passed=bool(passed),expected='positive' if case=='RX_LTI' else 'interaction_reject' if case=='TX_RX_INTERACTION' else 'null',rejected_fraction=0.,profile_relative_RMSE=rel,null_profile_RMS_rad=est['profile_RMS_rad'] if case!='RX_LTI' else None)
