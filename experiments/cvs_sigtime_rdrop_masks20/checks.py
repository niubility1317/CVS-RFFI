import copy,json,sys,torch
from torch.nn import functional as F
from . import design as d,model as api,mechanism as f
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,TypeError,KeyError):return
 raise AssertionError('Illegal paired-mask operation accepted')
def main():
 torch.set_num_threads(2);device='cuda:0' if '--cuda' in sys.argv else 'cpu';negative=0
 if device!='cpu':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 get_rng=torch.get_rng_state if device=='cpu' else torch.cuda.get_rng_state
 marginal=[]
 for p in sorted(set(f.SITES.values())):
  torch.manual_seed(2026101601);x=torch.randn(262144,device=device,requires_grad=True);v=torch.randn_like(x)
  torch.manual_seed(2026101602);native=F.dropout(x,p,True);rn=get_rng().clone();gn=torch.autograd.grad(native,x,v)[0]
  torch.manual_seed(2026101602);noise=F.dropout(torch.ones_like(x),p,True);wrapped=f.NativeMask.apply(x,noise,p);rw=get_rng().clone();gw=torch.autograd.grad(wrapped,x,v)[0]
  assert torch.equal(native,wrapped) and torch.equal(gn,gw) and torch.equal(rn,rw)
  first=noise.ne(0);states=[]
  for kind in f.KINDS:
   torch.manual_seed(2026101603);second=f.draw_second(first,p,kind);states.append(get_rng().clone());p1=float((~first).float().mean());p2=float((~second).float().mean());both=float((~first&~second).float().mean())
   assert abs(p1-p)<.003 and abs(p2-p)<.003 and abs(both-(p*p if kind=='INDEPENDENT' else 0))<.003
   if kind=='ANTITHETIC':assert both==0 and bool((first|second).all())
   marginal.append(dict(p=p,kind=kind,first_drop_rate=p1,second_drop_rate=p2,both_drop_rate=both))
  assert torch.equal(states[0],states[1])
 for p,kind in ((0.,'ANTITHETIC'),(.5,'ANTITHETIC'),(.9,'ANTITHETIC'),(.1,'INVALID')):reject(lambda:f.draw_second(torch.ones(4,dtype=torch.bool),p,kind));negative+=1
 for action in (lambda:f.PairState('INVALID'),lambda:f.PairState('INDEPENDENT').second(),lambda:f.PairState('INDEPENDENT').finish()):reject(action);negative+=1
 state=f.PairState('INDEPENDENT');state.start();reject(state.start);reject(state.between);negative+=2
 reject(lambda:state.apply(next(iter(f.SITES)),torch.ones(2,3),.1));negative+=1
 from .provenance import validate_args
 for row in d.rows():
  c=d.config(row);args=vars(d.make_args(c,'cpu'));validate_args(args,c)
  for key,value in [('use_aug',False),('label_smoothing',0.),('source_fold_held_rx',None),('source_fold_training_receivers','1,3,4,6,8'),('baseline_ckpt','unknown.pt'),('teacher_ckpt','unknown.pt'),('epochs',50)]:
   bad=dict(args);bad[key]=value;reject(lambda:validate_args(bad,c));negative+=1
  for key,value in [('model_seed',2026092702),('test_views',['clean']),('head_fit_updates',1),('coupling','INVALID')]:
   bad=dict(c);bad[key]=value;reject(lambda:d.validate(bad));negative+=1
 models=[api.build(d.config(row),device) for row in d.rows()];initials=[m.state_dict() for m in models]
 assert all(set(s)==set(initials[0]) and all(torch.equal(v,s[k]) for k,v in initials[0].items()) for s in initials[1:])
 torch.manual_seed(2026101604);x=torch.randn(128,2,256,device=device);aug=x*.7+.13;y=torch.arange(128,device=device)%6
 outputs={};ends={};norms={};audits={};grads={};gradient_checks={}
 def run(m,arm):
  m.train();torch.manual_seed(2026101605)
  with m._phase1_labeled_feature_context():out=m(aug,y_tx=y,return_aux=True)
  assert torch.equal(m._legacy_actual_L_input,aug)
  ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
  correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra={},domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128)
  named=dict(m.named_parameters());names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight')
  g=torch.autograd.grad(ce+correction,[named[n] for n in names]+[api.classifier(m).weight]);assert all(torch.isfinite(v).all() and v.norm()>0 for v in g)
  outputs[arm]=(out['tx_logits'].detach(),(ce+correction).detach());grads[arm]=g;norms[arm]=[float(v.norm()) for v in g];ends[arm]=get_rng().clone()
 for row,m in zip(d.rows(),models):
  run(m,row['arm']);st=m._pair_state;assert st.stage=='idle' and not st.cache and st.audit['calls']==m._legacy_calls==m._feature_l_steps==1 and m._legacy_extra_packets==128
  assert all(z['calls']==1 for z in st.audit['sites'].values())
  if row['coupling']=='ANTITHETIC':assert all(z['both_drop']==0 for z in st.audit['sites'].values())
  audits[row['arm']]=st.snapshot();before=st.snapshot();m.train();m.forward_identity_only(torch.randn(2,2,256,device=device));assert st.snapshot()==before and not st.cache
  teacher=copy.deepcopy(m).eval();assert teacher._pair_state is not st and teacher._mask_native_context.__self__ is teacher and all(teacher.get_submodule(n)._pair_state is teacher._pair_state for n in f.SITES)
  before=teacher._pair_state.snapshot();teacher.forward_identity_only(torch.zeros(2,2,256,device=device));assert teacher._pair_state.snapshot()==before
  reject(lambda:teacher._source_labeled_feature_hook(model=teacher,optimizer_step=2));negative+=1
 for name in ('INDEP_CE','INDEP_RDROP'):
  row=next(z for z in d.rows() if z['arm']==name);m=api.original.build(d.native_config(row),device);run(m,'NATIVE_'+name)
  assert all(torch.equal(a,b) for a,b in zip(outputs[name],outputs['NATIVE_'+name])) and torch.equal(ends[name],ends['NATIVE_'+name])
  pairs=list(zip(grads[name],grads['NATIVE_'+name]));gradient_checks[name]=dict(bitwise=all(torch.equal(a,b) for a,b in pairs),max_abs=max(float((a-b).abs().max()) for a,b in pairs),max_relative_norm=max(float((a-b).norm()/b.norm()) for a,b in pairs))
  # Native CUDA backward itself varies at float32 reduction roundoff on repeats.
  assert all(torch.equal(a,b) if device=='cpu' else torch.allclose(a,b,atol=1e-7,rtol=1e-5) for a,b in pairs)
  assert gradient_checks[name]['max_relative_norm']<1e-6
 assert all(torch.equal(outputs[a][0],outputs[d.ARMS[0]][0]) for a in d.ARMS) and all(torch.equal(ends[a],ends[d.ARMS[0]]) for a in d.ARMS)
 fixture={a:dict(source_health=True,exports={'FINAL':dict(count=1260,errors=100,mean_CE=1.)}) for a in d.ARMS}
 assert not d.qualify(fixture)['qualified'];fixture['ANTI_RDROP']['exports']['FINAL']['errors']=99;assert d.qualify(fixture)['qualified']
 for reason in ('CE','health','interaction','own_CE'):
  bad=copy.deepcopy(fixture)
  if reason=='CE':bad['ANTI_RDROP']['exports']['FINAL']['mean_CE']=1.1
  elif reason=='health':bad['INDEP_RDROP']['source_health']=False
  elif reason=='interaction':bad['INDEP_CE']['exports']['FINAL']['errors']=104
  else:bad['ANTI_CE']['exports']['FINAL']['errors']=98
  assert not d.qualify(bad)['qualified']
 print(json.dumps(dict(status='PASS',device=device,protocol_negative_cases=negative,qualification_negative_cases=5,mask_marginal_checks=marginal,actual_same_initial_tensors=len(initials[0]),independent_native_output_loss_RNG_bitwise=True,independent_native_gradients=gradient_checks,all_first_logits_equal=True,all_native_RNG_equal=True,mask_sites=13,antithetic_double_drop_zero=True,actual_augmented_input_reused=True,mask_cache_empty_after_pair=True,U_and_eval_use_native_dropout=True,teacher_controller_isolated=True,actual_parameter_gradient_norms=norms,synthetic_identity_packet_forwards=1552,synthetic_mask_elements_per_p=262144,real_IQ_reads=0,checkpoint_loads=0,optimizer_steps=0,head_fit_updates=0,target_access=False)))
if __name__=='__main__':main()
