import copy,io,json,torch
from torch.nn import functional as F
from . import design as d,model as api
from .observer import Observer,objective,capture,deterministic
from experiments.cvs_source_signature_order.objective import descriptors
from experiments.cvs_source_layer_order.objective import relation_indices
from experiments.cvs_source_layer_order import design as sd
from experiments.cvs_source_layer_order.probe import cohorts

def checks():
 torch.set_num_threads(2);configs=[d.config(r) for r in d.training_rows()];models=[api.build(c,'cpu').train() for c in configs];first=models[0].state_dict();assert all(set(m.state_dict())==set(first) and all(torch.equal(v,m.state_dict()[k]) for k,v in first.items()) for m in models)
 a=d.make_args(configs[0],'cpu');assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lambda_u,a.lambda_ent)==(50,33,17,.16,.01) and a.use_ema_teacher and a.pseudo_temporal_mode=='batch_neighbor'
 y=torch.tensor([t for t in range(6) for dd in sd.DAYS for r in sd.RX]);rx=torch.tensor([r for t in range(6) for dd in sd.DAYS for r in sd.RX]);day=torch.tensor([dd for t in range(6) for dd in sd.DAYS for r in sd.RX]);q,groups=relation_indices(y,rx,day,list(range(90)));x=torch.randn(90,2,256,generator=torch.Generator().manual_seed(1887));m=models[0]
 with deterministic(m),capture(m) as maps:out=m.forward_identity_only(x)
 losses,details,clock=objective(maps,q);s=maps['SIG_DESCRIPTOR'];dyn,t=descriptors(s);assert s.shape==(90,112) and dyn.shape==(90,108) and torch.equal(t,torch.ones_like(t))
 for name,v in [('SIG_RAW',s),('SIG_DYNAMIC',dyn)]:
  z=F.normalize(v,dim=1,eps=1e-4);sim=z@z.T;reference=torch.stack([(.1-sim[i,p]+sim[i,n]).relu() for i,p,n in q]).mean();assert torch.allclose(losses[name],reference,atol=1e-7)
 fakeL=dict(y=y.repeat_interleave(70),rx=rx.repeat_interleave(70),day=day.repeat_interleave(70),ids=list(range(6300)),x=x.repeat_interleave(70,dim=0));ix=cohorts(fakeL);warmup=[];updates=[]
 for c,m in zip(configs,models):
  o=Observer();o.c=c;o.model=m;o.extra=0;o.records=[];o.grad=[];o.stream=io.StringIO();o.L=fakeL;o.ix=ix;o.q=q;o.groups=groups;o.params=[dict(m.named_parameters())[n] for n in d.PARAMETERS];o.head=m.id_backbone.encoder.core.id_backbone.cls_head.weight
  torch.manual_seed(377);out=m.forward_identity_only(x);ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);o.steps=4;o.main_CE=ce;rng=torch.get_rng_state().clone();flags=[v.training for v in m.modules()];before={k:v.clone() for k,v in m.state_dict().items()};aux,metrics=o.aux(m,4);assert aux==0 and o.extra==1 and torch.equal(rng,torch.get_rng_state()) and flags==[v.training for v in m.modules()] and all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());warmup.append(torch.autograd.grad(ce+aux,o.params+[o.head]))
  torch.manual_seed(488);out=m.forward_identity_only(x);ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);o.steps=16*222+4;o.main_CE=ce;aux,_=o.aux(m,o.steps);gs=torch.autograd.grad(aux,o.params+[o.head],allow_unused=True,retain_graph=True);assert gs[-1] is None
  if c['arm']=='SIG_TIME':assert all(g is None or not bool(g.any()) for g in gs)
  else:assert all(g is not None and g.norm()>0 for g in gs[:-1])
  cg=torch.autograd.grad(ce,o.head,retain_graph=True)[0];gg=torch.autograd.grad(ce+aux,o.head,retain_graph=True)[0];assert torch.equal(cg,gg);assert all(p.grad is None for p in m.parameters())
  opt=torch.optim.SGD(m.parameters(),lr=.001);opt.zero_grad();(ce+aux).backward();old=[p.detach().clone() for p in o.params+[o.head]];opt.step();assert all(not torch.equal(a,b) for a,b in zip(old,o.params+[o.head]));updates.append(dict(arm=c['arm'],actual_t1_t2_SIGprojection_update=True,default_head_joint_update=True,auxiliary_W_dependency=False))
 assert all(all(torch.equal(a,b) for a,b in zip(warmup[0],v)) for v in warmup[1:]);negative=0
 for key,value in [('margin',.3),('source_role','U_s'),('cohort_packets',128),('coefficient',.2),('view','joint160'),('no_extra_head',False)]:
  bad=copy.deepcopy(configs[0]);bad['signature_order'][key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Changed source recipe accepted')
 for bad in [torch.zeros(2,108),torch.zeros(2,112)]:
  try:descriptors(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid signature accepted')
 from .publish import configure
 from experiments.cvs_report_ssl import publish as t
 configure();compile(t.REMOTE.replace('CONFIG',repr(dict(project='/tmp',release='r',run='x',archive='a',sha256='s',commit='c'))),'<remote>','exec');assert all(p in t.PATHS for p in ['experiments/cvs_signature_order50','experiments/cvs_local_order50','experiments/cvs_source_signature_order','experiments/cvs_source_layer_order'])
 return dict(status='PASS',protocol_negative_tests=negative,actual_native_models=3,initial_parameters_equal=True,warmup_gradients_equal=True,vector_hinge_matches_explicit1800relations=True,native_updates=updates,modes_RNG_buffers_restored=True,gradient_observers_do_not_accumulate=True,synthetic_IQ_only=True,source_IQ_read=False,query_read=False,no_separate_head_fitting=True,remote_compile='PASS')
if __name__=='__main__':print(json.dumps(checks()))
