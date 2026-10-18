"""Bounded positive shared-I/Q local field with matched within-packet energy."""
import numpy as np
N=256;BLOCK=16;COUNT=16;GAIN=.1;STEP=.01

def windows():
 t=np.sin(np.pi*np.arange(BLOCK,dtype=np.float64)/(BLOCK-1))**2;t[0]=t[-1]=0.;w=np.zeros((COUNT,N))
 for b in range(COUNT):w[b,b*BLOCK:(b+1)*BLOCK]=t
 return w

def prepare(x):
 if not isinstance(x,np.ndarray) or x.ndim!=3 or x.shape[1:]!=(2,N) or x.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(x).all():raise ValueError('Finite existingIQ N,2,256 float only')
 a=x.astype(np.float64);power=np.square(a).sum((1,2));norm=np.sqrt(power);energy=np.sqrt(np.einsum('nct,bt->nb',np.square(a),windows()**2));delta=np.minimum(STEP,GAIN*energy.min(1)/np.maximum(norm,1e-24));amplitude=delta[:,None]*norm[:,None]/np.maximum(energy,1e-24)
 if np.any(amplitude>GAIN+1e-12):raise ValueError('Bounded perpacket energycontrol failed')
 return delta,amplitude

def apply(x,block,sign):
 delta,amplitude=prepare(x)
 if type(block) is not int or not 0<=block<COUNT or type(sign) is not int or sign not in (-1,1):raise ValueError('Fixed block andsign only')
 field=sign*amplitude[:,block,None]*windows()[block];z=(x.astype(np.float64)*(1+field[:,None,:])).astype(x.dtype);den=np.maximum(np.square(x.astype(float)).sum((1,2)),1e-24);actual=np.sqrt(np.square(z.astype(float)-x).sum((1,2))/den);outside=windows()[block]==0
 stats=dict(block=block,sign=sign,max_gain=float(np.abs(field).max()),step_error=float(np.abs(actual-delta).max()),max_relative_change=float(actual.max()),outside_unchanged=bool(np.array_equal(z[:,:,outside],x[:,:,outside])))
 return z,stats

def rank(derivatives,labels,rx,day):
 a=np.asarray(derivatives,dtype=np.float64);y=np.asarray(labels);r=np.asarray(rx);d=np.asarray(day)
 if a.shape!=(180,16) or not np.isfinite(a).all() or len(y)!=180 or len(r)!=180 or len(d)!=180:raise ValueError('All180 finite L16 source sensitivities required')
 denom=np.abs(a).sum(1);relative=np.divide(np.abs(a),denom[:,None],out=np.full_like(a,1/16),where=denom[:,None]>0);cells=[]
 for t in range(6):
  for rr in (1,3,4,6,8):
   for dd in (1,2,3):
    mask=(y==t)&(r==rr)&(d==dd)
    if mask.sum()!=2:raise ValueError('All90 sourcecells requiretwofitL')
    cells.append(relative[mask].mean(0))
 scores=np.asarray(cells).max(0);order=np.argsort(scores,kind='stable');return dict(scores=scores.tolist(),low_blocks=order[:4].tolist(),high_blocks=order[-4:].tolist(),source_zero_gradient_packets=int((denom==0).sum()),cell_scores=np.asarray(cells).tolist())
