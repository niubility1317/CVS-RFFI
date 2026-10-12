import torch
from torch.nn import functional as F

DIMS=dict(SUBBAND_MODULATION=64,POLYNOMIAL=128,AFFINE_IQ=48,LOG_SPECTRUM=64,SCALAR_CLOSURE=6,MATRIX_CLOSURE=12)

def complex_iq(x):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not torch.isfinite(x).all():raise ValueError('Finite receivedIQ Bx2x256 required')
 return torch.complex(x[:,0].float(),x[:,1].float())

def polynomial(x):
 z=complex_iq(x);m2=z.abs().square().mean(1);center=z[:,8:248];values=[]
 for lag in (1,2,4,8):
  q=z[:,8+lag:248+lag]*z[:,8-lag:248-lag]*center.conj().square()/m2.square().clamp_min(1e-20)[:,None]
  q=torch.where((m2>1e-10)[:,None],q,torch.zeros_like(q));values.extend((q.real,q.imag))
 return F.adaptive_avg_pool1d(torch.stack(values,1),16).flatten(1)

def affine(x):
 complex_iq(x);v=x.double()-x.double().mean(2,keepdim=True);g=v@v.transpose(1,2)/256;e=torch.linalg.eigvalsh(g);good=(e[:,0]>1e-8*e[:,1])&(e[:,1]>1e-12)
 safe=torch.where(good[:,None,None],g,torch.eye(2,device=x.device,dtype=g.dtype));iv=torch.linalg.solve(safe,v)/256;vals=[(v*iv).sum(1)]
 for lag in (1,2,4,8,16):vals.append((v[:,:,:-lag]*iv[:,:,lag:]).sum(1))
 out=torch.cat([F.adaptive_avg_pool1d(a[:,None],8).squeeze(1) for a in vals],1);return torch.where(good[:,None],out,torch.zeros_like(out)).float()

def haar(a,scale):return (a[...,scale:]-a[...,:-scale])/2

def subband(x):
 z=complex_iq(x);spec=torch.fft.fft(z,dim=1);band=torch.arange(256,device=x.device)//32;mask=F.one_hot(band,8).T.to(spec.real.dtype);env=torch.fft.ifft(spec[:,None,:]*mask[None],dim=2).abs();energy=env.mean(2);good=energy>1e-5*energy.mean(1,keepdim=True).clamp_min(1e-10);vals=[]
 for scale in (1,2,4,8):
  a=haar(env,scale).abs();s1=a.mean(2);s2=haar(a,scale*2).abs().mean(2);floor=energy*1e-4+1e-10
  vals.extend((torch.where(good,s1/(energy+floor),torch.zeros_like(s1)),torch.where(good&(s1>floor),s2/(s1+floor),torch.zeros_like(s1))))
 return torch.cat(vals,1)

def log_basis(device,dtype):
 f=torch.arange(256,device=device,dtype=dtype)/256
 return torch.stack((torch.ones_like(f),torch.sin(2*torch.pi*f),torch.cos(2*torch.pi*f),torch.sin(4*torch.pi*f),torch.cos(4*torch.pi*f)),1)

def remove_log_basis(s):
 b=log_basis(s.device,s.dtype);q=torch.linalg.qr(b,mode='reduced').Q;return s-(s@q)@q.T

def log_spectrum(x):
 z=complex_iq(x);amp=torch.fft.fft(z,dim=1).abs();floor=amp.mean(1,keepdim=True)*1e-5+1e-10;s=remove_log_basis((amp+floor).log());return F.adaptive_avg_pool1d(s[:,None],64).squeeze(1)

TRIPLES=((3,6,9),(6,9,12),(9,12,15))
def fragments(x):return torch.fft.fft(complex_iq(x).reshape(-1,8,32),dim=2)

def scalar_loops(a):
 vals=[]
 for i,j,k in TRIPLES:
  def coherence(p,q):
   u,v=a[:,:,p],a[:,:,q];den=(u.abs().square().mean(1)*v.abs().square().mean(1)).sqrt();return (u*v.conj()).mean(1)/den.clamp_min(1e-12)
  c=coherence(i,j)*coherence(j,k)*coherence(k,i);vals.extend((c.real,c.imag))
 return torch.stack(vals,1)

def scalar_closure(x):return scalar_loops(fragments(x))

def matrix_loops(v):
 # v[B,node,fragment,2] transforms as v'=A_node v (column convention).
 cov=lambda i,j:torch.einsum('bna,bnc->bac',v[:,i],v[:,j].conj())/v.shape[2]
 vals=[];valid=[]
 for i,j,k in TRIPLES:
  nodes=(i,j,k);cs={n:cov(n,n) for n in nodes};ee={n:torch.linalg.eigvalsh(cs[n]) for n in nodes};good=torch.ones(len(v),device=v.device,dtype=torch.bool)
  for n in nodes:good=good&(ee[n][:,0]>1e-6*ee[n][:,1])&(ee[n][:,1]>1e-10)
  inv={n:torch.linalg.inv(torch.where(good[:,None,None],cs[n],torch.eye(2,device=v.device,dtype=v.dtype))) for n in nodes}
  loop=cov(i,j)@inv[j]@cov(j,k)@inv[k]@cov(k,i)@inv[i];tr=loop.diagonal(dim1=1,dim2=2).sum(1);det=torch.linalg.det(loop)
  vals.extend(torch.where(good,a,torch.zeros_like(a)) for a in (tr.real,tr.imag,det.real,det.imag));valid.append(good)
 return torch.stack(vals,1),torch.stack(valid,1)

def matrix_closure(x):
 a=fragments(x);mirror=(-torch.arange(32,device=x.device))%32;v=torch.stack((a,a[:,:,mirror].conj()),3).permute(0,2,1,3);return matrix_loops(v)[0]

def phase_rotate(x,k):
 if k not in range(4):raise ValueError('C4element')
 return (x,torch.stack((-x[:,1],x[:,0]),1),-x,torch.stack((x[:,1],-x[:,0]),1))[k]

def digital_action(x,kind,eta):
 # Dimensionless bounded proxy: smooth spectral log-gain or circular fractional delay.
 z=complex_iq(x);f=torch.fft.fftfreq(256,device=x.device);s=torch.fft.fft(z,dim=1)
 if kind=='smooth_gain':s=s*torch.exp(eta*.1*torch.log(torch.tensor(10.,device=x.device))/20*torch.cos(2*torch.pi*f))
 elif kind=='delay':s=s*torch.exp(-2j*torch.pi*f*(eta*.02))
 else:raise ValueError('Unregistered tangent action')
 y=torch.fft.ifft(s,dim=1);return torch.stack((y.real,y.imag),1)

def features(arm,x):
 fn=dict(SUBBAND_MODULATION=subband,POLYNOMIAL=polynomial,AFFINE_IQ=affine,LOG_SPECTRUM=log_spectrum,SCALAR_CLOSURE=scalar_closure,MATRIX_CLOSURE=matrix_closure)
 if arm not in fn:raise ValueError('Unregistered report frontend')
 z=fn[arm](x)
 if z.shape!=(len(x),DIMS[arm]) or not torch.isfinite(z).all():raise FloatingPointError('Report feature shape/value')
 return z

def bounded_tangent_distance(delta,jacobian):
 # Exact two-coordinate box LS: interior, four constrained edges and zero.
 if delta.ndim!=1 or jacobian.shape!=(len(delta),2):raise ValueError('Two registered tangent coordinates required')
 zero=torch.zeros(2,device=delta.device,dtype=delta.dtype);raw=torch.linalg.pinv(jacobian)@delta;candidates=[zero]
 if (raw.abs()<=1).all():candidates.append(raw)
 for fixed in (0,1):
  free=1-fixed
  for bound in (-1.,1.):
   a=zero.clone();a[fixed]=bound;r=delta-jacobian[:,fixed]*bound;a[free]=((r*jacobian[:,free]).sum()/jacobian[:,free].square().sum().clamp_min(1e-20)).clamp(-1,1);candidates.append(a)
 coefficients=torch.stack(candidates);residual=(delta[None]-coefficients@jacobian.T).square().sum(1);i=residual.argmin();return residual[i],coefficients[i]
