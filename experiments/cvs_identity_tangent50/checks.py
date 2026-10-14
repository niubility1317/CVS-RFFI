import torch
from .tangent import direction,perturb,flatness

def run():
 gen=torch.Generator().manual_seed(62813)
 y=torch.arange(6).repeat_interleave(10);rx=torch.tensor([1,3,4,6,8]).repeat_interleave(2).repeat(6);day=torch.ones(60,dtype=torch.long)
 h=torch.randn(60,16,24,generator=gen,dtype=torch.double);h+=y[:,None,None]*torch.randn(1,16,1,generator=gen,dtype=torch.double)*.2
 safe,scale,ss=direction(h,y,rx,day,'IDSAFE');raw,_,rr=direction(h,y,rx,day,'RAW_MATCHED');zero,_,_=direction(h,y,rx,day,'ZERO')
 assert ss['identity_rank']<=5 and ss['identity_overlap_IDSAFE']<1e-18 and torch.allclose(raw.norm(dim=1),safe.norm(dim=1),atol=1e-12,rtol=1e-12)
 assert not safe.requires_grad and not scale.requires_grad and ss['maximum_coordinate']<=1+1e-12
 assert torch.equal(perturb(h,zero,scale,1),h)
 # Additive TX means are protected; label permutation cannot change their span.
 perm=torch.tensor([4,1,5,0,3,2]);other,_,_=direction(h,perm[y],rx,day,'IDSAFE');assert torch.allclose(other,safe,atol=1e-12,rtol=1e-10)
 # Finite-difference regularizer must update a feature transform, with no head.
 projection=torch.nn.Linear(16,160,bias=False,dtype=torch.double);optimizer=torch.optim.SGD(projection.parameters(),lr=.01);initial=projection.weight.detach().clone()
 plus=projection(perturb(h,safe,scale,1).mean(-1));minus=projection(perturb(h,safe,scale,-1).mean(-1));loss=flatness(plus,minus);loss.backward()
 assert torch.isfinite(projection.weight.grad).all() and float(projection.weight.grad.norm())>0
 optimizer.step();assert not torch.equal(initial,projection.weight)
 z=torch.randn(60,160,generator=gen,dtype=torch.double);assert flatness(z,z).item()==0
 rejected=0
 for role in ('U_s','V','target'):
  try:direction(h,y,rx,day,'IDSAFE',role)
  except ValueError:rejected+=1
  else:raise AssertionError('Forbidden fitting role accepted')
 try:direction(h,y,rx,day+torch.arange(60)%2,'IDSAFE')
 except ValueError:rejected+=1
 else:raise AssertionError('Mixed days accepted')
 return dict(status='PASS',identity_projection=True,equal_perturbation_energy=True,detached_direction=True,class_permutation=True,zero_control_exact=True,feature_update=True,protocol_negatives=rejected,query_read=False)
if __name__=='__main__':print(run())

def wiring(device='cpu'):
 from copy import deepcopy
 from . import design as d,model as api
 from .tangent import pair,cap,frozen_observation_mode
 torch.set_num_threads(2);models=[api.build(d.config(r),device) for r in d.rows()[1:]]
 states=[m.state_dict() for m in models]
 assert all(all(torch.equal(v,states[0][k]) for k,v in state.items()) for state in states[1:])
 m=models[2];teacher=deepcopy(m);rng=torch.get_rng_state().clone()
 gen=torch.Generator().manual_seed(8912);x=torch.randn(60,2,256,generator=gen).to(device);y=torch.arange(6,device=device).repeat_interleave(10);rx=torch.tensor([1,3,4,6,8],device=device).repeat_interleave(2).repeat(6);day=torch.ones(60,dtype=torch.long,device=device)
 m.train();identity=torch.nn.functional.cross_entropy(m(x),y,label_smoothing=.01);old=api.classifier(m).weight.detach().clone();initial={n:p.detach().clone() for n,p in m.named_parameters()}
 with frozen_observation_mode(m):
  raw,info=pair(m,x,y,rx,day,'t1','IDSAFE');weighted,budget=cap(m,identity,raw,1.)
  wgrad=torch.autograd.grad(raw,api.classifier(m).weight,retain_graph=True,allow_unused=True)[0];assert wgrad is None
 assert m.training and budget['actual_norm_ratio']<=.200001 and raw.item()>0
 optimizer=torch.optim.AdamW(m.parameters(),lr=.0002);(identity+weighted).backward();optimizer.step()
 changed=sum(int((p!=initial[n]).sum()) for n,p in m.named_parameters() if not n.endswith('cls_head.weight'));assert changed>0 and not torch.equal(old,api.classifier(m).weight)
 assert not hasattr(teacher,'_source_feature_auxiliary_hook')
 m.eval();saved={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():
  z=m.forward_identity_only(x[:2])['id_feat_joint'];h=api.classifier(m);w=h.weight
  try:
   h.weight=torch.nn.Parameter(torch.zeros_like(w),requires_grad=False);assert torch.equal(z,m.id_backbone.encoder.features(x[:2]));del h.weight;assert torch.equal(z,m.id_backbone.encoder.features(x[:2]))
  finally:h.weight=w
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in saved.items())
 return dict(status='PASS',device=device,all_three_initial_states_equal=True,actual_joint160=True,regularizer_classW_gradient=None,feature_changed_values=changed,defaultW_updated_by_CE=True,budget=budget,feature_export_W_independent=True,teacher_callback_absent=True,head_fit_updates=0,synthetic_evidence=True)
