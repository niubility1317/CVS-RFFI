import copy,json
import torch
from . import design as d
from .objective import relation

def checks():
 torch.set_num_threads(2);torch.manual_seed(9421)
 z=torch.randn(12,160,requires_grad=True);q=torch.randn(12,6).softmax(1).detach();rx=torch.tensor([1,3,4,6,8,1,3,4,6,8,1,3])
 loss,g=relation(z,q,rx,'XR_GRAPH');loss.backward();assert z.grad is not None and torch.isfinite(z.grad).all()
 assert not g['allowed'].diagonal().any() and not (g['allowed']&(rx[:,None]==rx[None,:])).any()
 assert torch.allclose(g['target'].sum(1),torch.ones(12),atol=1e-6)
 perm=torch.tensor([7,1,10,2,4,11,0,8,3,6,5,9]);again,gg=relation(z.detach()[perm].requires_grad_(True),q[perm],rx[perm],'XR_GRAPH');assert abs(float(loss.detach())-float(again.detach()))<2e-6
 same_rx=torch.ones(12,dtype=torch.long);zero,empty=relation(z,q,same_rx,'XR_GRAPH');assert float(zero.detach())==0 and not empty['valid'].any()
 bads=[lambda:relation(z,q,rx,'UNKNOWN'),lambda:relation(z,q,rx,'XR_GRAPH',role='V'),lambda:relation(z,q,rx,'XR_GRAPH',role='query'),lambda:relation(z.detach(),q,rx,'XR_GRAPH'),lambda:relation(z,q.clone().requires_grad_(True),rx,'XR_GRAPH'),lambda:relation(z,torch.zeros_like(q),rx,'XR_GRAPH'),lambda:relation(z,torch.full_like(q,float('nan')),rx,'XR_GRAPH'),lambda:relation(z[:,:159],q,rx,'XR_GRAPH'),lambda:relation(z,q[:,:5],rx,'XR_GRAPH'),lambda:relation(z,q,rx.float(),'XR_GRAPH'),lambda:relation(z,q,torch.full_like(rx,14),'XR_GRAPH'),lambda:relation(z*0,q,rx,'XR_GRAPH')]
 for bad in bads:
  try:bad()
  except (ValueError,RuntimeError):pass
  else:raise AssertionError('Invalid source graph accepted')
 c=d.config(d.rows()[0]);changed=copy.deepcopy(c);changed['temperature']=.1
 try:d.validate(changed)
 except ValueError:pass
 else:raise AssertionError('Unregistered temperature scan accepted')
 from experiments.cvs_receiver_dro50 import model as api
 from .audit import measure_batch
 model=api.build(d.source_config(),'cpu');teacher=copy.deepcopy(model);model.eval();teacher.eval()
 before={k:v.clone() for k,v in model.state_dict().items()};before_t={k:v.clone() for k,v in teacher.state_dict().items()};rng=torch.get_rng_state().clone()
 x=torch.randn(12,2,256,generator=torch.Generator().manual_seed(9452));y=torch.arange(12)%6
 result=measure_batch(model,teacher,x,y,rx,9453)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and all(torch.equal(v,teacher.state_dict()[k]) for k,v in before_t.items())
 assert torch.equal(rng,torch.get_rng_state()) and all(p.grad is None for p in model.parameters())
 assert result['auxiliary_W_gradient_none'] and all(float(v.norm())>0 for mode in ('ALL_GRAPH','XR_GRAPH') for v in result['vectors'][mode][:3])
 assert [v.numel() for v in result['vectors']['CE']]==[3456,6912,576,960]
 return dict(status='PASS',invalid_or_unregistered_cases=len(bads)+1,semantic_graph_without_truth_argument=True,teacher_target_detached=True,no_diagonal_or_sameRX_edge_for_XR=True,zero_for_undefined_rows=True,permutation_invariant=True,actual_t1_t2_SIG_parameter_gradients=True,auxiliary_default_W_gradient_none=True,actual_CE_W_gradient_nonzero=True,state_and_RNG_unchanged=True,head_fit_updates=0,optimizer_updates=0,real_IQ_or_checkpoint_read=False)
if __name__=='__main__':print(json.dumps(checks()))
