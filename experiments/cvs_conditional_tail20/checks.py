import copy,json
import torch
from . import design as d
from .loss import objective

def run():
 g=torch.Generator().manual_seed(2063);a=torch.randn(128,6,generator=g,requires_grad=True);b=torch.randn(128,6,generator=g,requires_grad=True);y=torch.arange(128)%6;rng=torch.get_rng_state().clone();q=torch.nn.functional.one_hot(y,6)*.99+.01/6
 for mode in ('CE2','TAIL2'):
  loss,info=objective(a,b,y,mode);w=torch.tensor(info['draw_weights']);ce=torch.stack([torch.nn.functional.cross_entropy(z,y,reduction='none',label_smoothing=.01) for z in (a,b)])
  expected=ce.mean() if mode=='CE2' else .5*ce.mean()+.5*ce.max(0).values.mean()
  torch.testing.assert_close(loss,expected)
  ga,gb=torch.autograd.grad(loss,(a,b))
  for z,grad,weights in ((a,ga,w[0]),(b,gb,w[1])):torch.testing.assert_close(grad,weights[:,None]*(z.detach().softmax(1)-q)/128,atol=1e-7,rtol=1e-5)
  assert bool((w>0).all()) and torch.equal(w.sum(0),torch.ones(128))
  same,sw=objective(a,a,y,mode);assert torch.equal(torch.tensor(sw['draw_weights']),torch.full((2,128),.5));torch.testing.assert_close(same,torch.nn.functional.cross_entropy(a,y,label_smoothing=.01))
  swapped,details=objective(b,a,y,mode);torch.testing.assert_close(loss,swapped);torch.testing.assert_close(torch.tensor(details['draw_weights']),w.flip(0))
 assert torch.equal(rng,torch.get_rng_state())
 negatives=0
 for args in ((a,b,y,'NEW'),(a[:,:5],b[:,:5],y,'TAIL2'),(a,b,y.float(),'TAIL2'),(a,b,torch.full_like(y,6),'TAIL2'),(a*float('nan'),b,y,'TAIL2')):
  try:objective(*args)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid true-label risk input accepted')
 for change in ({'epochs':50},{'checkpoint_sources':['old.pth']},{'model_seed':2},{'training_mechanism':{}},{'data_contract':{}},{'test_views':['clean']}):
  c=copy.deepcopy(d.config(d.rows()[0]));c.update(change)
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Changed frozen research recipe accepted')
 # Draw-0 and draw-1 namespaces cannot coincide within any native epoch.
 for step in range(1,4441):
  ep=(step-1)//222;other=4441-step
  assert (other-1)//222!=ep
 return dict(status='PASS',exact_risk_formula_and_gradients=True,all_observations_positive_CE=True,per_packet_weights_sum_one=True,ties_and_draw_swap=True,global_RNG_unchanged=True,disjoint_all_4440_namespace_keys=True,protocol_negative_tests=negatives,NN=0,target_access=False)

def flow():
 from . import model as api
 from experiments.cvs_rdrop_sinc_units import optimizer as units
 torch.set_num_threads(2);before_states=[];packets=0
 class Fake:
  def __init__(self,draw):self.draw=draw
  def apply(self,x,meta,epoch,step):
   records=[dict(sample_id=[i],session_id='rx1:day1:eq1',namespace='synthetic-draw'+str(self.draw)) for i in range(len(x))]
   return x*(.8 if self.draw==0 else .6)+torch.roll(x,1,-1)*.1,records
 for c in map(d.config,d.rows()):
  torch.manual_seed(d.SEED);m=api.build(c,'cpu').train();assert sum(p.numel() for p in m.parameters())==900722
  before={k:v.detach().clone() for k,v in m.state_dict().items()};before_states.append(before)
  m._nontrue_channel=Fake(0);m._second_channel=Fake(1);m._feature_l_steps=1;calls=[];original=m.forward_identity_only
  def tracked(*args,**kw):
   z=original(*args,**kw);calls.append(torch.get_rng_state().clone());return z
  m.forward_identity_only=tracked
  x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(2064));labels=torch.arange(128)%6;domains=torch.arange(128)%15
  z=m.forward_identity_only(x,domain_labels=domains)['tx_logits'];loss=torch.nn.functional.cross_entropy(z,labels,label_smoothing=.01)
  correction,_=api.hook(m,model=m,labeled_output=dict(tx_logits=z),labels=labels,extra=(None,{}),domains=domains,optimizer_step=1,identity_loss=loss,labeled_count=128,labeled_inputs=x)
  assert len(calls)==3 and torch.equal(torch.get_rng_state(),calls[1]) and m._nontrue_L_packets==384 and len(m._nontrue_records)==1
  opt=units.build(m,m.parameters(),'CENTER_UNITS',lr=.0002,weight_decay=.0001);(loss+correction).backward()
  named=dict(m.named_parameters());head=next(k for k in named if k.endswith('cls_head.weight'))
  assert named[head].grad is not None and float(named[head].grad.norm())>0
  assert any('.sinc.' in k and p.grad is not None and float(p.grad.norm())>0 for k,p in named.items())
  assert any('.core.id_backbone.' in k and '.cls_head.' not in k and '.sinc.' not in k and p.grad is not None and float(p.grad.norm())>0 for k,p in named.items())
  units.clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before[head],m.state_dict()[head]);packets+=384
 assert all(torch.equal(before_states[0][k],before_states[1][k]) for k in before_states[0])
 return dict(status='PASS',same_initial_F_both_modes=True,actual_three_forward_hook=True,private_second_dropout_restores_native_RNG=True,real_Sinc_body_defaulthead_joint_gradients=True,synthetic_NN=packets,synthetic_updates=2,checkpoint_loads=0,real_IQ=0,target_access=False)

if __name__=='__main__':
 import sys
 print(json.dumps(flow() if '--flow' in sys.argv else run()))
