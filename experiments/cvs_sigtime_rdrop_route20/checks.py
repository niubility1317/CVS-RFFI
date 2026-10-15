import copy,json,sys,torch
from torch.nn import functional as F
from . import design as d,model as api,mechanism as f
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,TypeError,KeyError):return
 raise AssertionError('Illegal RDrop action accepted')
def main():
 torch.set_num_threads(2);torch.manual_seed(17)
 a=torch.randn(9,160,dtype=torch.float64,requires_grad=True);b=torch.randn_like(a,requires_grad=True);W=torch.randn(6,160,dtype=torch.float64,requires_grad=True)
 losses={};gradients={}
 for arm in d.ARMS:
  audit=f.new_audit();loss=f.consistency(a,b,W,arm,audit);g=torch.autograd.grad(loss,(a,b,W),allow_unused=True);losses[arm]=loss.detach();gradients[arm]=g
  assert audit['calls']==1 and audit['max_norm_relative_error']<1e-12
  assert (g[2] is not None)==(arm=='FULL_KL')
  if arm=='DIFFERENCE':assert torch.equal(g[0]+g[1],torch.zeros_like(a)) and audit['max_difference_common_abs']==0
 assert all(torch.equal(x,losses['FULL_KL']) for x in losses.values())
 assert torch.equal(gradients['FULL_KL'][0],gradients['FEATURE_KL'][0]) and torch.equal(gradients['FULL_KL'][1],gradients['FEATURE_KL'][1])
 norm=lambda gs:sum(g.square().sum() for g in gs[:2])
 assert torch.allclose(norm(gradients['SHRINK']),norm(gradients['DIFFERENCE']),atol=1e-15,rtol=1e-12)
 assert norm(gradients['DIFFERENCE'])<=norm(gradients['FEATURE_KL'])
 g1,g2=torch.randn_like(a),torch.randn_like(a);v=torch.randn_like(a)
 before=f.project(g1,g2,'DIFFERENCE');after=f.project(g1+v,g2+v,'DIFFERENCE')
 assert all(torch.allclose(x,y,atol=1e-14,rtol=1e-14) for x,y in zip(before,after))
 assert all(not x.any() for x in f.project(torch.zeros_like(a),torch.zeros_like(a),'SHRINK'))
 reject(lambda:f.consistency(a,b,W,'INVALID',f.new_audit()));reject(lambda:f.project(g1*float('nan'),g2,'DIFFERENCE'));reject(lambda:f.logits(a[:,:159],W));negative=3
 from .provenance import validate_args
 for row in d.rows():
  c=d.config(row);args=vars(d.make_args(c,'cpu'));validate_args(args,c)
  for key,value in [('use_aug',False),('label_smoothing',0.),('source_fold_held_rx',None),('source_fold_training_receivers','1,3,4,6,8'),('baseline_ckpt','unverified.pt'),('teacher_ckpt','unverified.pt'),('epochs',50)]:
   bad=dict(args);bad[key]=value;reject(lambda:validate_args(bad,c));negative+=1
  for key,value in [('model_seed',2026092701),('test_views',['clean']),('head_fit_updates',1),('fold',{})]:
   bad=dict(c);bad[key]=value;reject(lambda:d.validate(bad));negative+=1
 device='cuda:0' if '--cuda' in sys.argv else 'cpu'
 if device!='cpu':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 models=[api.build(d.config(row),device) for row in d.rows()];initials=[m.state_dict() for m in models]
 assert all(set(s)==set(initials[0]) and all(torch.equal(v,s[k]) for k,v in initials[0].items()) for s in initials[1:])
 torch.manual_seed(2026101571);x=torch.randn(128,2,256,device=device);aug=x*.7+.13;y=torch.arange(128,device=device)%6
 get_rng=torch.get_rng_state if device=='cpu' else torch.cuda.get_rng_state
 ends=[];values=[];norms={};Wgrads={};audits={}
 for row,m in zip(d.rows(),models):
  m.train();torch.manual_seed(2026101572)
  with m._phase1_labeled_feature_context():out=m(aug,y_tx=y,return_aux=True)
  assert torch.equal(m._legacy_actual_L_input,aug) and not torch.equal(m._legacy_actual_L_input,x)
  ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
  correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra={},domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128)
  assert m._legacy_actual_L_input is None and m._legacy_calls==m._rdrop_route_calls==m._feature_l_steps==1 and m._legacy_extra_packets==128
  params=dict(m.named_parameters());names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight')
  chosen=[params[n] for n in names]+[api.classifier(m).weight];gs=torch.autograd.grad(ce+correction,chosen)
  assert all(torch.isfinite(g).all() and g.norm()>0 for g in gs) and m._rdrop_route_audit['calls']==1
  norms[row['arm']]=[float(g.norm()) for g in gs];Wgrads[row['arm']]=gs[-1];audits[row['arm']]=m._rdrop_route_audit.copy()
  ends.append(get_rng().clone());values.append((ce+correction).detach())
  teacher=copy.deepcopy(m).eval();assert teacher._legacy_actual_L_input is None and teacher._legacy_original_labeled_scope.__self__ is teacher
  reject(lambda:teacher._source_labeled_feature_hook(model=teacher,labeled_output=out,labels=y,extra={},domains=None,optimizer_step=2,identity_loss=ce,labeled_count=128));negative+=1
 assert all(torch.equal(r,ends[0]) for r in ends[1:]) and all(torch.equal(v,values[0]) for v in values[1:])
 assert all(torch.equal(Wgrads[arm],Wgrads['FEATURE_KL']) for arm in ('SHRINK','DIFFERENCE'))
 assert not torch.equal(Wgrads['FULL_KL'],Wgrads['FEATURE_KL'])
 fixture={arm:dict(source_health=True,removed_common_energy_fraction=.5,exports={'FINAL':dict(count=1260,errors=100,mean_CE=1.)}) for arm in d.ARMS}
 assert not d.qualify(fixture)['qualified'];fixture['DIFFERENCE']['exports']['FINAL']['errors']=99;assert d.qualify(fixture)['qualified']
 for key in ('CE','signal','health','norm_control'):
  bad=copy.deepcopy(fixture)
  if key=='CE':bad['DIFFERENCE']['exports']['FINAL']['mean_CE']=1.1
  elif key=='signal':bad['DIFFERENCE']['removed_common_energy_fraction']=.009
  elif key=='health':bad['FEATURE_KL']['source_health']=False
  else:bad['SHRINK']['exports']['FINAL']['errors']=98
  assert not d.qualify(bad)['qualified']
 print(json.dumps(dict(status='PASS',device=device,negative_cases=negative,gate_negative_cases=5,FP64_forward_KL_identical=True,FP64_feature_projection_and_norm_control=True,actual_same_initial_tensors=len(initials[0]),actual_forward_objectives_bitwise_equal=True,actual_native_logits_bitwise_equal=True,same_native_RNG=True,post_augmentation_capture=True,CE_defaultW_joint_and_unmodified_for_feature_routes=True,actual_parameter_gradient_norms=norms,actual_KL_gradient_audits=audits,synthetic_identity_packet_forwards=1024,real_IQ_reads=0,checkpoint_loads=0,optimizer_steps=0,head_fit_updates=0,target_access=False)))
if __name__=='__main__':main()
