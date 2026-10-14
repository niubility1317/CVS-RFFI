import copy,json,tempfile,math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from . import design as d,model as api
from .observer import Observer
from .objective import consistency,coefficient,ungated,policy
from experiments.cvs_neighbor_pair50.checks import Hidden
from experiments.cvs_neighbor_pair50.sampler import replace
from experiments.cvs_signature_stats50.features import export
def checks():
 torch.set_num_threads(2);torch.manual_seed(7123);configs=[d.config(r) for r in d.training_rows()];models=[api.build(c,'cpu').train() for c in configs]
 assert all(torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items())
 x=torch.randn(8,2,256);y=torch.arange(8)%6;results=[];dataset=Hidden();loader=DataLoader(dataset,batch_size=256,shuffle=True,num_workers=0,drop_last=False)
 def order(l):
  torch.manual_seed(9307);v=torch.cat(list(l));return v,torch.get_rng_state().clone()
 before,rb=order(loader);after,ra=order(replace(loader,'RANDOM'));assert torch.equal(before,after) and torch.equal(rb,ra)
 for c,m in zip(configs,models):
  teacher=copy.deepcopy(m).eval().requires_grad_(False);a=d.make_args(c,'cpu');assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lambda_u,a.lambda_ent,a.label_smoothing)==(50,33,17,.16,.01,.01) and a.use_ema_teacher and a.pseudo_temporal_mode=='batch_neighbor'
  fake=SimpleNamespace(_trusted_pseudo_loss_hook=None,_source_feature_auxiliary_hook=None,_unlabeled_feature_observer_hook=None,_unlabeled_outputs_observer_hook=None,_build_ssdg_wisig_data=lambda:dict(unlabeled_loader=loader))
  with tempfile.TemporaryDirectory() as directory:
   o=Observer(c);o.initialize(m,c,Path(directory))
   with o.native_scope(fake):
    assert fake._build_ssdg_wisig_data()['unlabeled_loader'].sampler.mode=='RANDOM'
    torch.manual_seed(128);l=m.forward_identity_only(x)['tx_logits'];ce=F.cross_entropy(l,y,label_smoothing=.01);zero=ce*0
    returned=o.observe(model=m,args=a,epoch=1,batch_idx=1,loss_l=ce,loss_u=zero,loss_ent=zero,active=False,logits=None,pseudo=None,mask=None);extra,info=o.aux(m,1);assert returned is zero and float(extra)==0 and o.feature_packets==0
    W=m.id_backbone.encoder.core.id_backbone.cls_head.weight;ps=o.params+[W];g0=torch.autograd.grad(ce,ps,retain_graph=True);g1=torch.autograd.grad(ce+extra,ps,retain_graph=True);assert all(torch.equal(u,v) for u,v in zip(g0,g1))
    torch.manual_seed(129);l=m.forward_identity_only(x)['tx_logits'];ce=F.cross_entropy(l,y,label_smoothing=.01)
    with torch.no_grad():weak=teacher.forward_identity_only(x)
    strong=m.forward_identity_only(x+.015);pseudo=weak['tx_logits'].argmax(1);mask=torch.ones(8,dtype=torch.bool);uce=F.cross_entropy(strong['tx_logits'],pseudo);ent=-(strong['tx_logits'].softmax(1)*strong['tx_logits'].log_softmax(1)).sum(1).mean();o.steps=33*222
    rng=torch.get_rng_state().clone();fake._unlabeled_outputs_observer_hook(model=m,args=a,epoch=34,batch_idx=1,teacher_features=weak['id_feat_joint'],student_features=strong['id_feat_joint'],teacher_logits=weak['tx_logits'],student_logits=strong['tx_logits']);returned=o.observe(model=m,args=a,epoch=34,batch_idx=1,loss_l=ce,loss_u=uce,loss_ent=ent,active=True,logits=strong['tx_logits'],pseudo=pseudo,mask=mask);extra,info=o.aux(m,33*222+1)
    assert returned is uce and torch.equal(rng,torch.get_rng_state()) and o.pending is None and o.feature_packets==8 and all(p.grad is None for p in m.parameters())
    if c['arm']==d.PRIMARY:
     gg=torch.autograd.grad(extra,ps,allow_unused=True,retain_graph=True);assert gg[-1] is None and all(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in gg[:-1])
    else:assert float(extra)==0
    base=ce+.16*uce+.01*ent;head0=torch.autograd.grad(base,W,retain_graph=True)[0];head1=torch.autograd.grad(base+extra,W,retain_graph=True)[0];assert torch.equal(head0,head1)
    optimizer=torch.optim.SGD(m.parameters(),lr=.001);old=[p.detach().clone() for p in ps];optimizer.zero_grad();(base+extra).backward();optimizer.step();assert all(not torch.equal(u,v) for u,v in zip(old,ps)) and all(p.grad is None for p in teacher.parameters())
    results.append(dict(arm=c['arm'],original_U_loss_object_preserved=True,no_extra_forward_in_observer=True,auxiliary_W_gradient_absent=True,main_head_gradient_unchanged=True,default_head_t1_t2_SIGprojection_actual_updates=True))
   assert all(getattr(fake,n) is None for n in ('_trusted_pseudo_loss_hook','_source_feature_auxiliary_hook','_unlabeled_feature_observer_hook','_unlabeled_outputs_observer_hook'))
  m.eval().requires_grad_(False)
  with torch.no_grad():
   z=m.forward_identity_only(x)['id_feat_joint'];f=export(m);assert torch.equal(z,f.forward_features(x));cloned=copy.deepcopy(m);cloned.id_backbone.encoder.core.id_backbone.cls_head.weight.zero_();assert torch.equal(z,cloned.forward_identity_only(x)['id_feat_joint'])
  assert sum(p.numel() for p in f.parameters())==195041
 assert coefficient(33,'FEATURE')==0 and math.isclose(coefficient(34,'FEATURE'),.02) and coefficient(38,'FEATURE')==.1 and coefficient(50,'ZERO')==0
 s=torch.randn(5,160,requires_grad=True);t=torch.randn(5,160);tl=torch.zeros(5,6);sl=torch.zeros(5,6);v=consistency(s,t,tl,sl);expected=(1-F.cosine_similarity(s,t)).mean();assert torch.allclose(v,expected,atol=1e-6)
 reject=sl.clone();reject[:,0]=5;assert float(consistency(s,t,tl,reject))==0
 mixed=tl.clone();mixed[:2,0]=6;z=consistency(s,t,mixed,reject);per=.5*(F.normalize(s,dim=1)-F.normalize(t,dim=1)).square().sum(1);assert torch.allclose(z,per[:2].sum()/5) and not torch.allclose(z,per[:2].mean())
 assert torch.equal(policy(tl,sl)['keep'],torch.ones(5,dtype=torch.bool))
 gs=torch.autograd.grad(z,s,retain_graph=True)[0];manual=torch.autograd.grad(per[:2].sum()/5,s)[0];assert torch.equal(gs,manual)
 negative=0
 for ss,tt in [(s,t.requires_grad_(True)),(s.detach(),t.detach()),(s[:,:159],t.detach()[:,:159]),(s,torch.full_like(s,float('nan')))]:
  try:consistency(ss,tt,tl,sl)
  except ValueError:negative+=1
  else:raise AssertionError('Illegal feature graph accepted')
 for key,value in [('epochs',51),('readouts',['FIT_HEAD']),('target_access',True),('model_initialization','target_best'),('confidence_gate',dict(gap_threshold=.01)),('separate_head_training',True)]:
  bad=copy.deepcopy(configs[0]);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Unregistered protocol accepted')
 from .publish import configure
 from experiments.cvs_report_ssl import publish as transport
 configure();assert all(p in transport.PATHS for p in ('experiments/cvs_feature_mean_teacher50','experiments/cvs_sigtime_pair50','experiments/cvs_neighbor_pair50'));compile(transport.REMOTE.replace('CONFIG',repr(dict(project='/tmp',release='r',run='x',archive='a',sha256='s',commit='c'))),'<remote>','exec')
 return dict(status='PASS',protocol_negative_tests=negative,matched_initial_models=True,native_RANDOM_order_RNG_preserved=True,actual_checks=results,feature_parameters=195041,feature_export_exact=True,W_independent=True,zero_pre_pseudo_gradient_unchanged=True,teacher_stopgradient=True,hook_restore=True,no_extra_head_or_modules=True,synthetic_IQ_only=True,query_read=False,remote_compile='PASS')
if __name__=='__main__':print(json.dumps(checks()))
