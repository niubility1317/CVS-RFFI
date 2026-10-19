import torch
from experiments.cvs_rdrop_sinc_units.descriptors import path_signature
MODES=('NATIVE','E','I','EI')
def coarsen(u,mode):
 if mode not in MODES or u.ndim!=3 or u.shape[1:]!=(8,6) or not torch.isfinite(u).all():raise ValueError('Fixed8x6 finite window and registered boundary mode required')
 if mode=='NATIVE':return u
 v=u.reshape(-1,4,2,6).clone();indices=([0,3] if 'E' in mode else [])+([1,2] if 'I' in mode else []);v[:,indices]=v[:,indices].mean(2,keepdim=True)
 return v.reshape_as(u)
def descriptor(u,mode):return path_signature(coarsen(u,mode),True)
def identities(u):
 v=path_signature(u,True);ij=torch.triu_indices(7,7,1,device=u.device);mixed=7+torch.where(ij[1]==6)[0];trap=(.5*(u[:,:-1]+u[:,1:])).mean(1);area=trap-.5*(u[:,0]+u[:,-1]);return dict(displacement_max_error=float((v[:,:6]-(u[:,-1]-u[:,0])).abs().max()),time_area_max_error=float((v[:,mixed]-area).abs().max()))
def from_coordinates(pool,coordinates,mode):
 if len(coordinates)!=3 or len(pool.views)!=3 or any(v.mode!='SIG_TIME' for v in pool.views):raise ValueError('Original three SIG_TIME views required')
 all_desc=[]
 for view,u in zip(pool.views,coordinates):
  if u.ndim!=3 or u.shape[1:]!=(32,6):raise ValueError('Actual t2 local32x6 path required')
  n=len(u);v=descriptor(u.reshape(n*4,8,6),mode)@view.descriptor_lift;all_desc.append(v.reshape(n,112))
 return .2*pool.gain_raw.tanh()*pool.output_norm(pool.output_projection(torch.cat(all_desc,1)))
