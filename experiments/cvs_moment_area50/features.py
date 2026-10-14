"""Keep local level/variance and replace only 15 off-diagonal relations."""
import torch
from experiments.cvs_signature_stats50.features import DescriptorView,install as install_base
from experiments.cvs_signature_stats50.descriptors import path_signature
MODES={'SIG_TIME':'SIG_TIME','COV_SHAPE':'COV_SHAPE','AREA_SHAPE':'AREA_SHAPE'}
def shape_descriptor(u,mode):
 if u.ndim!=3 or u.shape[-1]!=6 or u.shape[1]<2 or mode not in ('COV_SHAPE','AREA_SHAPE'):raise ValueError('Fixed6D statistical/order descriptor required')
 mean=u.mean(1);centered=u-mean[:,None];cov=centered.transpose(1,2)@centered/u.shape[1]
 variance=cov.diagonal(dim1=-2,dim2=-1);ij=torch.triu_indices(6,6,offset=1,device=u.device)
 relation=cov[:,ij[0],ij[1]] if mode=='COV_SHAPE' else path_signature(u,False)[:,6:]
 return torch.cat((mean,variance,relation),-1)
class ShapeView(DescriptorView):
 def forward(self,h):
  u=torch.tanh(self.local_projection(h)).transpose(1,2);b,t,_=u.shape
  windows=u.reshape(b,4,t//4,6).reshape(b*4,t//4,6)
  return (shape_descriptor(windows,self.mode)@self.descriptor_lift).reshape(b,112)
def install(model,seed,arm):
 if arm not in MODES:raise ValueError('Unregistered moment-area arm')
 model=install_base(model,seed,'SIG_TIME');pool=model.id_backbone.encoder.sequence_pool
 if arm!='SIG_TIME':
  view=pool.views[0];fixed=DescriptorView(view.local_projection.in_channels,'COV',seed,0)
  view.descriptor_lift=fixed.descriptor_lift.clone();view.mode=arm;view.__class__=ShapeView
 pool.arm=arm;pool.modes=(MODES[arm],)
 model._feature_learning_report.update(arm=arm,modes=[MODES[arm]],shared_single_local_projection=True,descriptor_order='increment7,area21' if arm=='SIG_TIME' else 'mean6,variance6,offdiag_cov15_or_skew_area15; fixed27to28lift',parameter_matched=True,intrinsic_descriptor_rank_matched=False)
 return model
