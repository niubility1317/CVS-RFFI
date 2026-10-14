import json,torch
from . import design as d
from .weights import losses
from .probe import gradient_stats,statistics,smoke
from experiments.cvs_moment_area50 import model as api

def checks():
 torch.set_num_threads(2);torch.manual_seed(219);z=torch.randn(8,6,requires_grad=True);p=torch.tensor([0,1,2,3,4,5,0,1]);a=torch.tensor([True,True,False,False,True,False,True,False]);saved=a.clone();v=losses(z,p,a)
 assert torch.equal(a,saved) and torch.equal(v['accept'],a) and not v['raw_weight'].requires_grad and not v['hardness'].requires_grad and v['raw_weight'].min()>=.75 and v['raw_weight'].max()<=1.25
 ce=torch.nn.functional.cross_entropy(z,p,reduction='none');manual=(v['raw_weight'][a]*ce[a]).sum()/v['raw_weight'][a].sum();g1=torch.autograd.grad(v['weighted'],z,retain_graph=True)[0];g2=torch.autograd.grad(manual,z)[0];assert torch.equal(g1,g2) and torch.allclose(v['effective_weight'][a].sum(),a.sum().float())
 empty=losses(z.detach().requires_grad_(),p,torch.zeros_like(a));assert empty['uniform']==empty['weighted']==0 and torch.count_nonzero(empty['effective_weight'])==0
 negative=0
 for zz,pp,aa in [(z[:,:5],p,a),(z*float('nan'),p,a),(z,p,torch.ones(8)),(z,torch.full_like(p,6),a),(z[:0],p[:0],a[:0])]:
  try:losses(zz,pp,aa)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid weight input accepted')
 for key,value in [('U_IQ_read',True),('U_truth_read',True),('head_fit',True),('query_read',True),('weight_bounds',[.5,2.]),('hardness_scale',.2)]:
  try:d.validate(dict(d.config(d.rows()[0]),**{key:value}))
  except ValueError:negative+=1
  else:raise AssertionError('Foreign protocol/config accepted')
 model=api.build(d.source_config(),'cpu').eval();real=smoke(model)
 gs=[torch.randn(n) for n in (3456,6912,9216)];r=gradient_stats(gs,gs,gs);assert all(v['weighted_uniform_gradient_difference_norm']==0 and abs(v['weighted_uniform_cosine']-1)<1e-12 for v in r)
 ix=torch.arange(6300);raw=dict(y=ix//1050,rx=torch.tensor(d.RX)[ix//210%5],day=ix//70%3+1,accept=ix%4==0,teacher_correct=ix%7!=0,effective_weight=(ix%4==0).float(),hardness=torch.ones(6300)*.05,raw_weight=torch.ones(6300));records=statistics(raw);assert len(records)==15 and records[0]['uniform_precision']==records[0]['weighted_precision']
 return dict(status='PASS',synthetic_only=True,source_or_target_IQ_read=False,stopgrad_exact=True,acceptmask_unchanged=True,normalized_weighted_CE_gradient_exact=True,empty_accept_loss_zero=True,bounds_fixed=True,protocol_negatives=negative,native_model_no_data_smoke=real,gradient_recount_math=True,source_precision_records=15)
if __name__=='__main__':print(json.dumps(checks()))
