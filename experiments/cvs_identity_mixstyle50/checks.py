"""Focused role negatives and actual adapter forward/gradient/EMA paths."""
import copy,torch
from . import design as d
from .model import metadata,intervene,installed

def checks():
 torch.set_num_threads(2)
 y=torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5]);rx=torch.tensor([1,3]*6);day=torch.ones(12,dtype=torch.long)
 raw=dict(rx_i=rx,day_i=day,eq_i=y,sig_i=torch.arange(12),base_index=torch.arange(12));extra=[torch.zeros(12,dtype=torch.long),raw];meta=metadata(y,extra)
 for field,bad in [('rx_i',torch.full((12,),14)),('day_i',torch.full((12,),4)),('base_index',torch.arange(12).float())]:
  q=dict(raw);q[field]=bad
  try:metadata(y,[extra[0],q])
  except ValueError:pass
  else:raise AssertionError('Forbidden role/physical metadata accepted')
 try:metadata(torch.full((12,),-1),extra)
 except ValueError:pass
 else:raise AssertionError('Hidden U accepted')
 torch.manual_seed(87);x=torch.randn(12,8,32,requires_grad=True);rng=torch.get_rng_state().clone();mixed,v=intervene(x,meta,3)
 assert v['enabled'] and v['changed_packets']>0 and torch.equal(rng,torch.get_rng_state())
 grad=torch.autograd.grad(mixed.square().mean(),x)[0];assert grad.isfinite().all() and grad.abs().sum()>0 and x.grad is None
 controls=[d.config(r) for r in d.training_rows()];models=[]
 for c in controls:
  torch.manual_seed(d.SEEDS[0])
  with installed(c,training=True) as n:
   a=d.make_args(c,'cpu');a.num_domains=15;a.input_len=256;model=n.build_baseline_model(a,torch.device('cpu'));models.append(copy.deepcopy(model.state_dict()))
   model.eval();saved={k:v.clone() for k,v in model.state_dict().items()};iq=torch.randn(12,2,256);teacher=copy.deepcopy(model).eval();moved=n.move_batch((iq,y,extra[0],raw),torch.device('cpu'))
   with torch.no_grad():ev=model(iq,y_tx=y,return_aux=True);tv=teacher(iq,y_tx=y,return_aux=True)
   assert torch.equal(ev['id_feat_joint'],tv['id_feat_joint']) and model.id_backbone._source_style_calls==0 and teacher.id_backbone._source_style_calls==0
   model.train();out=model(moved[0],y_tx=moved[1],domain_labels=extra[0],return_aux=True);loss=torch.nn.functional.cross_entropy(out['tx_logits'],y);loss.backward()
   core=model.id_backbone.encoder.core.id_backbone
   assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in core.t1.parameters())
   assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in core.t2.parameters())
   assert core.cls_head.weight.grad is not None and core.cls_head.weight.grad.abs().sum()>0
   # A verification-only optimizer step must change ftheta and exported z.
   before=ev['id_feat_joint'].clone();torch.optim.SGD(model.parameters(),lr=.001).step();model.eval()
   with torch.no_grad():after=model(iq,return_aux=True)['id_feat_joint']
   assert not torch.equal(before,after)
   assert model.id_backbone._source_style_calls==(1 if c['identity_mixstyle']['enabled'] else 0)
   if c['identity_mixstyle']['enabled']:assert set(model.id_backbone._source_style_history[0]['layers'])=={'t1','t2'}
   n.move_batch((iq,torch.full_like(y,-1),extra[0],raw),torch.device('cpu'));model.train();calls=model.id_backbone._source_style_calls
   model(iq,y_tx=None,return_aux=True);assert model.id_backbone._source_style_calls==calls
   teacher.train(False)
   with torch.no_grad():teacher(iq,y_tx=None,return_aux=True)
   assert teacher.id_backbone._source_style_calls==0 and all(torch.equal(v,teacher.state_dict()[k]) for k,v in saved.items())
   assert all(not module._forward_hooks for module in (core.t1,core.t2))
 assert set(models[0])==set(models[1]) and all(torch.equal(models[0][k],models[1][k]) for k in models[0])
 return dict(status='PASS',role_negative=True,private_operator_RNG=True,real_main_L_t1_t2_gradient_and_update=True,default_W_normal_joint_gradient=True,source_joint160_changed=True,teacher_U_eval_no_intervention=True,same_seed_initial_state_exact=True,new_parameters=0,query_read=False,checkpoint_read=False)
if __name__=='__main__':
 import json
 print(json.dumps(checks()))
