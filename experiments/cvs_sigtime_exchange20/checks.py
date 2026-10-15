import copy,json,sys,torch
from torch.nn import functional as F
from . import design as d,model as api,mechanism as f
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,TypeError,KeyError):return
 raise AssertionError('Illegal action accepted')
def metadata(device):
 i=torch.arange(128,device=device);y=(i//8)%6
 return y,dict(eq_i=y.clone(),rx_i=torch.tensor([d.FIT[(j//2)%4] for j in range(128)],device=device),day_i=1+i//48,base_index=i.clone())
def main():
 torch.set_num_threads(2);torch.manual_seed(17);y,meta=metadata('cpu');maps,n=f.donors(y,meta);assert n==128
 assert torch.equal(maps['SELF'],torch.arange(128)) and all(torch.equal(y[p],y) for p in maps.values())
 assert torch.equal(meta['rx_i'][maps['SAME_RX']],meta['rx_i']) and bool((meta['rx_i'][maps['CROSS_RX']]!=meta['rx_i']).all())
 for p in maps.values():
  assert torch.equal(meta['eq_i'][p],meta['eq_i']) and torch.equal(meta['day_i'][p],meta['day_i'])
 a=torch.randn(128,160,dtype=torch.float64,requires_grad=True);b=torch.randn_like(a,requires_grad=True);W=torch.randn(6,160,dtype=torch.float64,requires_grad=True)
 native=.5*(F.cross_entropy(f.logits(a,W),y,label_smoothing=.01)+F.cross_entropy(f.logits(b,W),y,label_smoothing=.01));self_ce,audit=f.loss(a,b,W,y,maps['SELF'])
 assert torch.allclose(native,self_ce,atol=1e-12,rtol=0)
 g1=torch.autograd.grad(native,(a,b,W),retain_graph=True);g2=torch.autograd.grad(self_ce,(a,b,W),retain_graph=True)
 assert all(torch.allclose(x,z,atol=1e-12,rtol=1e-10) for x,z in zip(g1,g2))
 for arm,p in maps.items():
  plus,minus,center,delta=f.virtual_features(a,b,p);ce,stats=f.loss(a,b,W,y,p)
  assert torch.allclose((plus+minus)*.5,center,atol=1e-14,rtol=0) and stats['energy_relative_error']<1e-12
  gradients=torch.autograd.grad(ce,(a,b,W),retain_graph=True);assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients)
 zeros=torch.zeros_like(a);assert f.loss(zeros,zeros,W,y,maps['CROSS_RX'])[1]['energy_relative_error']==0
 sparse={k:v[:1] for k,v in meta.items()};m,count=f.donors(y[:1],sparse);assert count==0 and all(p.tolist()==[0] for p in m.values())
 for key,bad in [('rx_i',torch.ones(128,dtype=torch.long)),('day_i',torch.zeros(128,dtype=torch.long)),('base_index',torch.zeros(128,dtype=torch.long))]:
  altered=dict(meta);altered[key]=bad;reject(lambda:f.donors(y,altered))
 reject(lambda:f.donors(torch.full_like(y,-1),meta));reject(lambda:f.donors(y,{}));reject(lambda:f.virtual_features(a,b,torch.zeros(128,dtype=torch.long)))
 reject(lambda:f.virtual_features(a[:,:159],b[:,:159],maps['SELF']));reject(lambda:f.virtual_features(a*float('nan'),b,maps['SELF']))
 negative=8
 for row in d.rows():
  c=d.config(row);d.validate(c);d.make_args(c,'cpu')
  for key,value in [('model_seed',2026092701),('head_fit_updates',1),('test_views',['clean']),('fold',{}),('epochs',50)]:
   bad=dict(c);bad[key]=value;reject(lambda:d.validate(bad));negative+=1
 device='cuda:0' if '--cuda' in sys.argv else 'cpu'
 if device!='cpu':
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 models=[api.build(d.config(row),device) for row in d.rows()];states=[m.state_dict() for m in models]
 assert all(set(s)==set(states[0]) and all(torch.equal(v,s[k]) for k,v in states[0].items()) for s in states[1:])
 torch.manual_seed(2026101550);x=torch.randn(128,2,256,device=device);aug=x*.7+.13;y,meta=metadata(device);end_rng=[];self_difference=None;gradnorms={};self_gradient_difference=None
 get_rng=torch.get_rng_state if device=='cpu' else torch.cuda.get_rng_state
 set_rng=torch.set_rng_state if device=='cpu' else torch.cuda.set_rng_state
 for row,m in zip(d.rows(),models):
  m.train();torch.manual_seed(2026101551)
  with m._phase1_labeled_feature_context():out=m(aug,y_tx=y,return_aux=True)
  assert torch.equal(m._legacy_actual_L_input,aug) and not torch.equal(m._legacy_actual_L_input,x)
  ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);before=get_rng().clone()
  correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(None,meta),domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128)
  assert m._legacy_actual_L_input is None and not torch.equal(before,get_rng()) and m._exchange_calls==m._feature_l_steps==1 and m._exchange_extra_packets==128 and metrics['exchange/cohort_packets']==128
  assert metrics['exchange/native_head_max_abs_error']<2e-5 and metrics['exchange/swapped_packets']==(0 if row['arm']=='SELF' else 128)
  if row['arm']=='SELF':
   with torch.random.fork_rng(devices=[] if device=='cpu' else [0]):
    set_rng(before);second=m.forward_identity_only(aug)['tx_logits']
   original=api.terms(out['tx_logits'],second,y,1.)[0];self_difference=float(((ce+correction)-original).detach().abs());assert self_difference<2e-6
  params=dict(m.named_parameters());names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight')
  selected=[params[k] for k in names]+[api.classifier(m).weight]
  gradients=torch.autograd.grad(ce+correction,selected,retain_graph=row['arm']=='SELF');assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients)
  if row['arm']=='SELF':
   reference=torch.autograd.grad(original,selected)
   self_gradient_difference=max(float((g-r).abs().max()) for g,r in zip(gradients,reference))
   assert all(torch.allclose(g,r,atol=2e-6,rtol=2e-4) for g,r in zip(gradients,reference))
  gradnorms[row['arm']]=[float(g.norm()) for g in gradients];end_rng.append(get_rng().clone())
  teacher=copy.deepcopy(m).eval();reject(lambda:teacher._source_labeled_feature_hook(model=teacher,labeled_output=out,labels=y,extra=(None,meta),domains=None,optimizer_step=2,identity_loss=ce,labeled_count=128));negative+=1
 assert all(torch.equal(r,end_rng[0]) for r in end_rng[1:])
 fixtures={arm:dict(source_health=True,mean_swapped_fraction=.5,exports={'FINAL':dict(count=1260,errors=100,mean_CE=1.)}) for arm in d.ARMS}
 assert not d.qualify(fixtures)['qualified'];fixtures['CROSS_RX']['exports']['FINAL']['errors']=99;assert d.qualify(fixtures)['qualified']
 for change in ('CE','signal','health','second_control'):
  bad=copy.deepcopy(fixtures)
  if change=='CE':bad['CROSS_RX']['exports']['FINAL']['mean_CE']=1.1
  elif change=='signal':bad['CROSS_RX']['mean_swapped_fraction']=.049
  elif change=='health':bad['SELF']['source_health']=False
  else:bad['SAME_RX']['exports']['FINAL']['errors']=98
  assert not d.qualify(bad)['qualified']
 print(json.dumps(dict(status='PASS',device=device,negative_cases=negative,qualification_boolean_failures=5,matched_initial_tensors=len(states[0]),FP64_self_objective_and_gradient_equivalent=True,common_cohort_and_donor_energy_preserved=True,native_HEAD_reused=True,post_augmentation_capture_verified=True,label_smoothing=.01,sinc_coordinates='RAW_HZ',same_native_RNG=True,actual_model_synthetic_L_packets=896,actual_body_SIG_defaultW_gradient_norms=gradnorms,FP32_SELF_RDrop_objective_abs_difference=self_difference,FP32_SELF_RDrop_selected_gradient_max_abs_difference=self_gradient_difference,head_fit_updates=0,optimizer_steps=0,real_IQ_reads=0,checkpoint_loads=0,target_access=False)))
if __name__=='__main__':main()
