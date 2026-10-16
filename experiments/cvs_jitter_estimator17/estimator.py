"""Four variances from observations only;all samples retained,zero classifiers."""
from itertools import combinations
import numpy as np
from . import design as d
PARTITIONS=(((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2)))

def symcross(a,b):
 # sqrt(2) weights make the3-vector inner product equal 2x2 Frobenius product.
 return np.stack((a.real*b.real,(a.real*b.imag+a.imag*b.real)/np.sqrt(2.),a.imag*b.imag),axis=-1)
def outer(a):return symcross(a,a)
def derivative(y):
 return np.fft.ifft(np.fft.fft(y,axis=-1)*(2j*np.pi*np.fft.fftfreq(d.PERIOD)),axis=-1)
def pair_basis(y,dy,pair):
 a,b=pair;z=symcross(y[:,a],y[:,b]);phase=z[...,::-1].copy();phase[...,1]*=-1
 jitter=symcross(dy[:,a],dy[:,b]);white=np.broadcast_to([.5,0.,.5],z.shape)
 return np.stack((white,z,phase,jitter),axis=-1).reshape(d.PACKETS,-1,4)
def nnls_moment(G,h):
 best=np.zeros(4);best_value=0.
 for size in range(1,5):
  for indices in combinations(range(4),size):
   sub=np.asarray(indices);v=np.linalg.solve(G[np.ix_(sub,sub)],h[sub])
   if np.any(v<0):continue
   w=np.zeros(4);w[sub]=v;value=float(w@G@w-2*h@w)
   if value<best_value:best,best_value=w,value
 return best

def solve_moments(G,G_second,h):
 diag=np.diag(G)
 if np.any(diag<=0):return dict(status='UNIDENTIFIABLE_NONPOSITIVE_DIAGONAL',variances=None,min_eigenvalue=None,standard_error=None,threshold=None)
 scale=1/np.sqrt(diag);scaled=G*scale[:,None]*scale[None,:]
 variance=np.maximum(0.,G_second-G*G)/(d.PACKETS-1)
 standard_error=float(np.sqrt(np.sum(variance*scale[:,None]**2*scale[None,:]**2)))
 eigenvalues=np.linalg.eigvalsh(scaled);minimum=float(eigenvalues[0]);threshold=5.*standard_error
 if minimum<=threshold:return dict(status='UNIDENTIFIABLE_UNCERTAINTY',variances=None,min_eigenvalue=minimum,standard_error=standard_error,threshold=threshold)
 coeff=scale*nnls_moment(scaled,scale*h)
 if not np.isfinite(coeff).all() or np.any(coeff<0):raise ValueError('Invalid physical variance solve')
 return dict(status='IDENTIFIABLE',variances=coeff.tolist(),min_eigenvalue=minimum,standard_error=standard_error,threshold=threshold)

def estimate(observations):
 y=np.asarray(observations)
 if y.shape!=(d.PACKETS,d.CYCLES,d.PERIOD) or not np.iscomplexobj(y) or not np.isfinite(y).all():raise ValueError('Exactly70x4x20 finite complex observations required')
 y=y.astype(np.complex128,copy=False);dy=derivative(y);Gs=[];hs=[]
 for a,b in PARTITIONS:
  Ma,Mb=pair_basis(y,dy,a),pair_basis(y,dy,b)
  Ga=np.einsum('pni,pnj->pij',Ma,Mb);Gs.append((Ga+Ga.transpose(0,2,1))/2)
  ca=outer(y[:,a[0]]-y[:,a[1]]).reshape(d.PACKETS,-1)/2
  cb=outer(y[:,b[0]]-y[:,b[1]]).reshape(d.PACKETS,-1)/2
  hs.append((np.einsum('pni,pn->pi',Ma,cb)+np.einsum('pni,pn->pi',Mb,ca))/2)
 Gp=np.mean(Gs,axis=0);hp=np.mean(hs,axis=0);G=np.mean(Gp,axis=0);G_second=np.mean(Gp*Gp,axis=0);h=np.mean(hp,axis=0)
 result=solve_moments(G,G_second,h);result.update(Gram=G.tolist(),Gram_second_moment=G_second.tolist(),rhs=h.tolist(),packets=d.PACKETS,all_observations_retained=True)
 return result
