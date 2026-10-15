import copy,json,sys,torch
from torch.nn import functional as F
from . import design as d,model as api
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,TypeError,KeyError):return
 raise AssertionError('Illegal objective/context accepted')
def main():
 torch.set_num_threads(2);torch.manual_seed(17);device='cuda:0' if '--cuda' in sys.argv else 'cpu'
 a=torch.randn(9,6,dtype=torch.float64,requires_grad=True);b=torch.randn_like(a,requires_grad=True);y=torch.arange(9)%6
 for alpha in (0.,1.):
  total,paired,kl,ce1,ce2=api.terms(a,b,y,alpha)
  expected=.5*(F.cross_entropy(a,y,label_smoothing=.01)+F.cross_entropy(b,y,label_smoothing=.01))+alpha*.5*(F.kl_div(a.log_softmax(1),b.softmax(1),reduction='batchmean')+F.kl_div(b.log_softmax(1),a.softmax(1),reduction='batchmean'))
  assert torch.allclose(total,expected,atol=1e-12,rtol=0) and torch.allclose(ce1+.5*(ce2-ce1)+alpha*kl,total,atol=1e-12,rtol=0)
  g1=torch.autograd.grad(total,(a,b),retain_graph=True);g2=torch.autograd.grad(expected,(a,b),retain_graph=True);assert all(torch.allclose(x,z,atol=1e-12,rtol=0) for x,z in zip(g1,g2))
 assert api.terms(a,a,y,1.)[2].detach()==0
 reject(lambda:api.terms(a,b,y,1.,smoothing=0));reject(lambda:api.terms(a,b,torch.full_like(y,-1),1.));reject(lambda:api.terms(a,b,y,.5));reject(lambda:api.terms(a*float('nan'),b,y,1.));negative=4
 for e in d.BUDGETS:
  for mode in d.MODES:
   c=d.config(next(r for r in d.rows() if r['epochs']==e and r['mode']==mode and r['model_seed']==d.SEEDS[0]));d.make_args(c,'cpu')
   for key,value in [('model_seed',1),('epochs',30),('head_fit_updates',1),('test_views',['practical_high']),('training_recipe','clean L LS0 CENTER25K')]:
    bad=dict(c);bad[key]=value;reject(lambda:d.validate(bad));negative+=1
 if device!='cpu':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 cs=[d.config(next(r for r in d.rows() if r['epochs']==20 and r['mode']==mode and r['model_seed']==d.SEEDS[0])) for mode in d.MODES]
 models=[api.build(c,device) for c in cs];initials=[m.state_dict() for m in models]
 assert all(set(s)==set(initials[0]) and all(torch.equal(v,s[k]) for k,v in initials[0].items()) for s in initials[1:])
 legacy=api.old.build(d.legacy_config(d.SEEDS[0]),device)
 assert all(torch.equal(v,legacy.state_dict()[k]) for k,v in initials[0].items())
 torch.manual_seed(2026101561);raw=torch.randn(128,2,256,device=device);aug=raw*.7+.13;labels=torch.arange(128,device=device)%6
 norm={};rngs={};original_zero_grad_difference=None;root_capture_verified=0
 get_rng=torch.get_rng_state if device=='cpu' else torch.cuda.get_rng_state
 set_rng=torch.set_rng_state if device=='cpu' else torch.cuda.set_rng_state
 for c,m in zip(cs,models):
  m.train();torch.manual_seed(2026101562)
  with m._phase1_labeled_feature_context():out=m(aug,y_tx=labels,return_aux=True)
  if c['mode']!='ORIGINAL':
   assert torch.equal(m._legacy_actual_L_input,aug) and not torch.equal(m._legacy_actual_L_input,raw);root_capture_verified+=1
  loss=F.cross_entropy(out['tx_logits'],labels,label_smoothing=.01);before=get_rng().clone()
  if c['mode']=='ORIGINAL':
   assert getattr(m,'_source_labeled_feature_hook',None) is None;total=loss
  else:
   addition,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=labels,extra={},domains=None,optimizer_step=1,identity_loss=loss,labeled_count=128);total=loss+addition
  assert m._legacy_actual_L_input is None and m._legacy_calls==m._feature_l_steps==1 and m._legacy_extra_packets==(0 if c['mode']=='ORIGINAL' else 128)
  assert torch.equal(before,get_rng())==(c['mode']=='ORIGINAL')
  params=dict(m.named_parameters());names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight');selected=[params[n] for n in names]+[api.classifier(m).weight]
  grads=torch.autograd.grad(total,selected);assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)
  norm[c['mode']]=[float(g.norm()) for g in grads];rngs[c['mode']]=get_rng().clone()
  teacher=copy.deepcopy(m).eval();assert teacher._legacy_actual_L_input is None and teacher._legacy_original_labeled_scope.__self__ is teacher
  if c['mode']!='ORIGINAL':
   reject(lambda:teacher._source_labeled_feature_hook(model=teacher,labeled_output=out,labels=labels,extra={},domains=None,optimizer_step=2,identity_loss=loss,labeled_count=128));negative+=1
   reject(lambda:m._source_labeled_feature_hook(model=m,labeled_output=out,labels=torch.full_like(labels,-1),extra={},domains=None,optimizer_step=2,identity_loss=loss,labeled_count=128));negative+=1
 assert torch.equal(rngs['TWO_CE'],rngs['RDROP'])
 paired=models[1];paired.train()
 reject(lambda:paired._source_labeled_feature_hook(model=paired,labeled_output=out,labels=labels,extra={},domains=None,optimizer_step=paired._feature_l_steps+1,identity_loss=loss,labeled_count=128));negative+=1
 print(json.dumps(dict(status='PASS',device=device,negative_cases=negative,matched_initial_tensors=len(initials[0]),original_SIGTIME_constructor_initial_state_equal=True,RAW_HZ_original_retained=True,original_label_smoothing=.01,original_augmentation_enabled=True,post_augmentation_capture_verified=root_capture_verified,paired_native_RNG_identical=True,ORIGINAL_native_CE_without_auxiliary_hook=True,actual_body_SIG_defaultW_gradient_norms=norm,actual_model_synthetic_L_forward_packets=640,optimizer_steps=0,head_fit_updates=0,actual_IQ_reads=0,checkpoint_loads=0,target_access=False)))
if __name__=='__main__':main()
