"""Focused real architecture synthetic checks; no checkpoint/IQ/query reads."""
import json,math
import torch
from torch.nn import functional as F
from . import design as d,model as api,optimizer as optim
from .loss import objective
from experiments.cvs_F_nontrue20 import model as base
from experiments.cvs_rdrop_sinc_units import optimizer as units

def run():
 torch.set_num_threads(2);torch.manual_seed(20261019642)
 parent=base.build(d.parent_config(),'cpu').eval();state={k:v.detach().clone() for k,v in parent.state_dict().items()}
 x=torch.randn(2,2,256);second=x+.2*torch.randn_like(x);y=torch.tensor([0,4]);rng=torch.get_rng_state().clone();results={};NN=0
 with torch.no_grad():teacher=parent.forward_identity_only(x)['tx_logits'];NN+=2
 for style in ('FULL','LORA','TEMP'):
  c=d.config(next(r for r in d.rows() if r['style']==style and r['As']));m=api.from_state(c,state);assert torch.equal(torch.get_rng_state(),rng)
  info=api.counts(m);captured=[];core=m.id_backbone.encoder.core.id_backbone
  def forward(value):
   result=m.forward_identity_only(value);captured.append(result['id_feat_joint'].detach().clone());return result['tx_logits']
  with torch.no_grad():zero=forward(x);NN+=2
  assert torch.equal(zero,teacher) and captured[-1].shape==(2,160)
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
  if style!='FULL':assert info['trainable_parameters']==(2304 if style=='LORA' else 2384) and info['total_parameters']==900722+info['trainable_parameters']
  opt=optim.build(m,c);frozen={k:v.detach().clone() for k,v in m.state_dict().items() if k not in info['trainable_names']};steps=1 if style=='FULL' else 2
  for step in range(steps):
   opt.zero_grad(set_to_none=True);z=forward(x);a=forward(second);NN+=4
   loss,metrics=objective(z,a,teacher,y);assert torch.isfinite(loss);loss.backward()
   if style!='FULL':
    grads={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if p.requires_grad};up=[v for n,v in grads.items() if '.lora_B.' in n or '.up.' in n];lower=[v for n,v in grads.items() if '.lora_A.' in n or '.down.' in n or '.temporal.' in n]
    assert min(up)>0 and (max(lower)==0 if step==0 else min(lower)>0)
   optim.clip(m,c,opt);opt.step()
  with torch.no_grad():after=forward(x);NN+=2
  assert not torch.equal(after,zero) and not torch.equal(captured[-1],captured[0])
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in frozen.items()) and torch.equal(m.state_dict()[d.HEAD],state[d.HEAD])
  if style=='LORA':
   # Standard linear BA is exactly a rank-r increment to each original pw.
   for name in ('t1','t2'):
    p=getattr(core,name).pw;h=torch.randn(2,p.in_channels,7,generator=torch.Generator().manual_seed(17));delta=(p.lora_B.weight[:,:,0]@p.lora_A.weight[:,:,0]).unsqueeze(-1)*p._lora_scale
    torch.testing.assert_close(p(h),F.conv1d(h,p.weight+delta,p.bias),atol=1e-6,rtol=1e-5)
  if style=='TEMP':
   # Original t2 hooks must observe the adapted result (SIG_TIME consumer).
   seen=[];h=core.t2.register_forward_hook(lambda mod,args,out:seen.append(out.detach().clone()))
   with torch.no_grad():m.forward_identity_only(x);NN+=2
   assert len(seen)==1;h.remove()
  results[style]=dict(**info,zero_exact=True,fixed_state_unchanged=True,learned_default_logits_changed=True,optimizer_steps=steps)
 # Fixed-head FULL physical AdamW algebra, including origin decay and clip.
 c=d.config(d.rows()[0]);m=api.from_state(c,state);opt=optim.build(m,c);named=dict(m.named_parameters());grads={k:torch.full_like(p,.02 if i%2 else -.03) for i,(k,p) in enumerate(named.items()) if p.requires_grad}
 for k,g in grads.items():named[k].grad=g.clone()
 norm=math.sqrt(sum(float((g.double()/(25000. if k in units.QNAMES else 1.)).square().sum()) for k,g in grads.items()));coef=min(1.,5./(norm+1e-6));sinc=m.get_submodule(units.PREFIX[:-1]);origins=dict(zip(units.QNAMES,(sinc.low_origin_hz,sinc.band_origin_hz)))
 optim.clip(m,c,opt);opt.step()
 for k,g in grads.items():
  lr=.0002*(1000. if k in units.QNAMES else 1.);eps=1e-8/(1000. if k in units.QNAMES else 1.);g=g*coef;expected=state[k]*(1-.0002*.0001)
  if k in origins:expected=expected-.0002*.0001*origins[k]/25000.
  expected=expected-lr*g/(g.abs()+eps);torch.testing.assert_close(named[k],expected,atol=2e-6,rtol=1e-6)
 assert torch.equal(named[d.HEAD],state[d.HEAD])
 # Independent loss oracle, including all-packet denominator and no parent error veto.
 z=torch.tensor([[2.,0.,0.,0.,0.,0.],[0.,2.,0.,0.,0.,0.]],requires_grad=True);a=z.detach().clone().requires_grad_();t=torch.tensor([[3.,0.,0.,0.,0.,0.],[3.,0.,0.,0.,0.,0.]],requires_grad=True);labels=torch.tensor([0,1]);loss,info=objective(z,a,t,labels)
 expected=F.cross_entropy(z,labels)+F.cross_entropy(a,labels)+F.kl_div(z.log_softmax(1),t.detach().softmax(1),reduction='none').sum(1)[0]/2
 torch.testing.assert_close(loss,expected);loss.backward();assert t.grad is None and a.grad[1,1]<0 and info['parent_correct']==1
 negatives=0
 bad=[(z[:,:5],a[:,:5],t[:,:5],labels),(z,a,t,torch.tensor([0,6])),(z,a,t,labels.float()),(z*float('nan'),a,t,labels)]
 for args in bad:
  try:objective(*args)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid L objective accepted')
 for mutate in ('epochs','default_head_frozen','U_used'):
  from copy import deepcopy
  q=deepcopy(c)
  if mutate=='epochs':q['epochs']=11
  else:q['training'][mutate]=not q['training'][mutate]
  try:d.validate(q)
  except ValueError:negatives+=1
  else:raise AssertionError('Changed fixed matrix accepted')
 try:api.from_state(c,None)
 except ValueError:negatives+=1
 else:raise AssertionError('Unverified parent fallback accepted')
 assert all(torch.equal(v,parent.state_dict()[k]) for k,v in state.items()) and torch.equal(torch.get_rng_state(),rng)
 return dict(status='PASS',rows=results,physical_FULL_oracle=True,parent_unchanged=True,global_RNG_preserved=True,protocol_negatives=negatives,synthetic_NN=NN,checkpoint_loads=0,real_IQ=0,query_read=False)

if __name__=='__main__':print(json.dumps(run()))
