"""One shared path, global-time local control, and matched-capacity coarse contexts."""
import torch
from torch import nn
from experiments.cvs_signature_stats50.features import DescriptorView,install as base_install
from experiments.cvs_source_hierarchy.algebra import signature,fold
from experiments.cvs_moment_area50.features import shape_descriptor

ARMS=('SIG_TIME','LOCAL_GTIME','HIER_STAT','HIER_ORDER')
def continuous(u):
 if u.ndim!=3 or u.shape[1:]!=(32,6):raise ValueError('Native32x6 path required')
 clock=torch.linspace(0,1,32,dtype=u.dtype,device=u.device)[None,:,None].expand(len(u),-1,-1);p=torch.cat((u,clock),-1)
 q=[signature(p[:,j*8:(j+1)*8]) for j in range(4)]
 bridges=[signature(p[:,j*8+7:j*8+9]) for j in range(3)]
 h=[fold([q[0],bridges[0],q[1]]),fold([q[2],bridges[2],q[3]])]
 return q,h,fold([h[0],bridges[1],h[1]])
class HierarchyView(DescriptorView):
 def forward(self,h):
  u=torch.tanh(self.local_projection(h)).transpose(1,2)
  if u.shape[1:]!=(32,6):raise ValueError('Fixed native32position path required')
  if self.mode=='LOCAL_GTIME':
   clock=torch.linspace(0,1,32,dtype=u.dtype,device=u.device)[None,:,None].expand(len(u),-1,-1);p=torch.cat((u,clock),-1)
   return torch.cat([signature(p[:,j*8:(j+1)*8]) for j in range(4)],-1)
  if self.mode=='HIER_ORDER':
   q,halves,full=continuous(u);return torch.cat([*q,*halves,full],-1)
  if self.mode=='HIER_STAT':
   clock=torch.linspace(0,1,32,dtype=u.dtype,device=u.device)[None,:,None].expand(len(u),-1,-1);p=torch.cat((u,clock),-1)
   q=[signature(p[:,j*8:(j+1)*8]) for j in range(4)]
   coarse=[shape_descriptor(segment,'COV_SHAPE')@self.coarse_stat_lift for segment in (u[:,:16],u[:,16:],u)]
   return torch.cat([*q,*coarse],-1)
  raise ValueError('Unregistered hierarchical view')
def install(model,seed,arm):
 if arm not in ARMS:raise ValueError('Unregistered hierarchy arm')
 model=base_install(model,seed,'SIG_TIME');pool=model.id_backbone.encoder.sequence_pool
 if arm!='SIG_TIME':
  view=pool.views[0];view.__class__=HierarchyView;view.mode=arm
  if arm in ('HIER_STAT','HIER_ORDER'):
   fixed=DescriptorView(view.local_projection.in_channels,'COV',seed,0);view.register_buffer('coarse_stat_lift',fixed.descriptor_lift.clone())
   with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(int(seed)+80411);nn.Conv1d(pool.channels,6,1,bias=False);pool.output_projection=nn.Linear(196,160,bias=False)
 pool.arm=arm;pool.modes=(arm,)
 model._feature_learning_report.update(arm=arm,modes=[arm],shared_single_local_projection=True,independent_local_projections=False,hierarchy=[4,2,1] if arm.startswith('HIER_') else [4],descriptor_total_width=196 if arm.startswith('HIER_') else 112,global_clock=arm!='SIG_TIME',three_boundary_increments=arm=='HIER_ORDER',coarse_statistics_control=arm=='HIER_STAT',extra_parameters=sum(p.numel() for p in pool.parameters()),parameter_matched_control='HIER_STAT' if arm=='HIER_ORDER' else None,descriptor_order='quarter4 then adjacenthalf2 then full1' if arm.startswith('HIER_') else 'quarter4',layer='t2',main_body_dropout_unchanged=True)
 return model
