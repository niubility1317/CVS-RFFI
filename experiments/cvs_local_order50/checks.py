import copy,io,json,torch
from types import SimpleNamespace
from torch.nn import functional as F
from . import design as d,model as api
from .observer import Observer,objective
from experiments.cvs_source_layer_order import design as sd
from experiments.cvs_source_layer_order.objective import capture,views,relation_indices,losses
from experiments.cvs_source_layer_order.probe import cohorts
from experiments.cvs_source_layer_order.checks import checks as source_checks

def checks():
 torch.set_num_threads(2);source=source_checks();configs=[d.config(r) for r in d.training_rows()];models=[api.build(c,'cpu').train() for c in configs];first=models[0].state_dict()
 assert all(set(m.state_dict())==set(first) and all(torch.equal(v,m.state_dict()[k]) for k,v in first.items()) for m in models)
 a=d.make_args(configs[0],'cpu');assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lambda_u,a.lambda_ent)==(50,33,17,.16,.01) and a.use_ema_teacher and a.pseudo_temporal_mode=='batch_neighbor'
 y=torch.tensor([t for t in range(6) for dd in sd.DAYS for r in sd.RX]);rx=torch.tensor([r for t in range(6) for dd in sd.DAYS for r in sd.RX]);day=torch.tensor([dd for t in range(6) for dd in sd.DAYS for r in sd.RX]);q,groups=relation_indices(y,rx,day,list(range(90)));gen=torch.Generator().manual_seed(1887);x=torch.randn(90,2,256,generator=gen)
 m=models[0];saved={k:v.clone() for k,v in m.state_dict().items()}
 from .observer import deterministic
 rng=torch.get_rng_state().clone();modes=[v.training for v in m.modules()]
 with deterministic(m),capture(m) as maps:out=m.forward_identity_only(x)
 z=views(maps,out);order,supcon,details=objective(maps,q,groups);reference=[losses(z[n],q,groups) for n in ('t1','t2')];assert torch.allclose(order,torch.stack([r[0] for r in reference]).mean(),atol=1e-7) and torch.allclose(supcon,torch.stack([r[1] for r in reference]).mean(),atol=1e-6)
 assert torch.equal(rng,torch.get_rng_state()) and modes==[v.training for v in m.modules()] and all(torch.equal(v,m.state_dict()[k]) for k,v in saved.items())
 params=[dict(m.named_parameters())[n] for n in sd.PARAMETERS];head=m.id_backbone.encoder.core.id_backbone.cls_head.weight
 gradients=torch.autograd.grad(order,params+[head],allow_unused=True,retain_graph=True);assert gradients[0].norm()>0 and gradients[1].norm()>0 and gradients[2] is None and gradients[3] is None
 fakeL=dict(y=y.repeat_interleave(70),rx=rx.repeat_interleave(70),day=day.repeat_interleave(70),ids=list(range(6300)),x=x.repeat_interleave(70,dim=0));ix=cohorts(fakeL);warmup_gradients=[];selected_gradients=[]
 for c,m in zip(configs,models):
  obs=Observer();obs.c=c;obs.model=m;obs.out=None;obs.extra=0;obs.records=[];obs.grad=[];obs.stream=io.StringIO();obs.L=fakeL;obs.ix=ix;obs.q=q;obs.groups=groups;obs.params=[dict(m.named_parameters())[n] for n in sd.PARAMETERS];obs.head=m.id_backbone.encoder.core.id_backbone.cls_head.weight
  # Actual native graph, mode/RNG restoration and zero warmup, without a data loader.
  torch.manual_seed(377);out=m.forward_identity_only(x);ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);obs.steps=4;obs.main_CE=ce;rng=torch.get_rng_state().clone();flags=[v.training for v in m.modules()];before={k:v.clone() for k,v in m.state_dict().items()};aux,metrics=obs.aux(m,4);assert aux==0 and obs.extra==1 and torch.equal(rng,torch.get_rng_state()) and flags==[v.training for v in m.modules()] and all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());warmup_gradients.append(torch.autograd.grad(ce+aux,obs.params+[obs.head]))
  torch.manual_seed(488);out=m.forward_identity_only(x);ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);obs.steps=16*222+4;obs.main_CE=ce;aux,_=obs.aux(m,obs.steps);ags=torch.autograd.grad(aux,obs.params+[obs.head],retain_graph=True,allow_unused=True);assert ags[-1] is None and ags[2] is None
  if c['arm']=='SIG_TIME':assert all(g is None or not bool(g.any()) for g in ags)
  else:assert ags[0].norm()>0 and ags[1].norm()>0
  cg=torch.autograd.grad(ce,obs.head,retain_graph=True)[0];gg=torch.autograd.grad(ce+aux,obs.head,retain_graph=True)[0];assert torch.equal(cg,gg)
  opt=torch.optim.SGD(m.parameters(),lr=.001);opt.zero_grad();(ce+aux).backward();oldhead=obs.head.detach().clone();opt.step();assert not torch.equal(oldhead,obs.head) and any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items());selected_gradients.append(dict(arm=c['arm'],auxiliary_W_gradient=False,default_head_joint_update=True))
 assert all(all(torch.equal(x,y) for x,y in zip(warmup_gradients[0],v)) for v in warmup_gradients[1:])
 negatives=0
 for key,value in [('margin',.3),('source_role','U_s'),('cohort_packets',128),('coefficient',.2)]:
  bad=copy.deepcopy(configs[0]);bad['local_order'][key]=value
  try:d.validate(bad)
  except ValueError:negatives+=1
  else:raise AssertionError('Changed recipe accepted')
 from .publish import configure
 from experiments.cvs_report_ssl import publish as t
 configure();compile(t.REMOTE.replace('CONFIG',repr(dict(project='/tmp',release='r',run='x',archive='a',sha256='s',commit='c'))),'<remote>','exec')
 assert 'experiments/cvs_source_layer_order' in t.PATHS
 return dict(status='PASS',protocol_negative_tests=source['protocol_negative_tests']+negatives,models=3,actual_initial_parameters_equal=True,zero_warmup_main_CE_gradients_equal=True,vectorized_SUPCON_matches_source_reference=True,actual_native_source_updates=selected_gradients,auxiliary_modes_RNG_buffers_restored=True,source_IQ=False,query_read=False,no_separate_head_fitting=True,remote_script_compile='PASS')
if __name__=='__main__':print(json.dumps(checks()))
