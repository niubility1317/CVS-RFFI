import copy,json,time
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .objective import validate_layout,variance_terms,calculate
from experiments.cvs_local_order50.observer import deterministic,derivative

def checks():
 torch.set_num_threads(2);started=time.time();configs=[d.config(r) for r in d.training_rows()];m=api.build(configs[0],'cpu').train();a=d.make_args(configs[0],'cpu')
 assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lambda_u,a.lambda_ent,a.label_smoothing)==(50,33,17,.16,.01,.01) and a.use_ema_teacher and a.pseudo_temporal_mode=='batch_neighbor'
 y=torch.tensor([t for t in range(6) for dd in d.DAYS for r in d.RX for j in range(2)]);rx=torch.tensor([r for t in range(6) for dd in d.DAYS for r in d.RX for j in range(2)]);day=torch.tensor([dd for t in range(6) for dd in d.DAYS for r in d.RX for j in range(2)]);ids=list(range(180));validate_layout(y,rx,day,ids)
 negative=0
 for yy,rr,dd,ii,role in [(y,rx,day,ids,'U_s'),(y,rx,day,ids[:-1]+[0],'L_s'),(y,torch.ones_like(rx),day,ids,'L_s'),(y,rx,torch.zeros_like(day),ids,'L_s'),(y[:-1],rx,day,ids,'L_s')]:
  try:validate_layout(yy,rr,dd,ii,role)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid source role/layout accepted')
 for key,value in [('coefficient',.1),('gradient_objects',['logits']),('cohort_packets',90),('every_steps',8)]:
  bad=copy.deepcopy(configs[0]);bad['backbone_fishr'][key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Changed frozen mechanism accepted')
 G=torch.randn(180,4,generator=torch.Generator().manual_seed(771),dtype=torch.double,requires_grad=True);ll,_=variance_terms(G,y,rx,day);scaled,_=variance_terms(G*3,y,rx,day)
 for mode in d.LEVELS:
  assert torch.allclose(ll[mode],scaled[mode],rtol=1e-12,atol=1e-12)
  g=torch.autograd.grad(ll[mode],G,retain_graph=True)[0];eps=1e-5;plus=G.detach().clone();minus=G.detach().clone();plus[0,0]+=eps;minus[0,0]-=eps;fd=(variance_terms(plus,y,rx,day)[0][mode]-variance_terms(minus,y,rx,day)[0][mode])/(2*eps);assert torch.allclose(g[0,0],fd,rtol=1e-5,atol=1e-8)
 gen=torch.Generator().manual_seed(2026211014);x=torch.randn(180,2,256,generator=gen);saved={k:v.clone() for k,v in m.state_dict().items()};flags=[v.training for v in m.modules()];requires=[p.requires_grad for p in m.parameters()];rng=torch.get_rng_state().clone()
 with deterministic(m):losses,detail,matrices=calculate(m,x,y,rx,day,ids)
 assert flags==[v.training for v in m.modules()] and requires==[p.requires_grad for p in m.parameters()] and torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,m.state_dict()[k]) for k,v in saved.items()) and all(p.grad is None for p in m.parameters())
 named=dict(m.named_parameters());params=[named[n] for n in d.PARAMETERS];W=m.id_backbone.encoder.core.id_backbone.cls_head.weight
 actual=[]
 for key in d.LEVELS:
  gg=torch.autograd.grad(losses[key],params+[W],retain_graph=True,allow_unused=True);assert gg[-1] is None and all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in gg[:2]);actual.append(dict(mode=key,loss=float(losses[key].detach()),gradient_norms=[float(g.norm()) if g is not None else None for g in gg],auxiliary_W_gradient=False))
 out=m.forward_identity_only(x[:12]);ce=F.cross_entropy(out['tx_logits'],y[:12],label_smoothing=.01);combined=ce+.01*losses['TX_DAY_CENTERED'];ceW=torch.autograd.grad(ce,W,retain_graph=True)[0];sumW=torch.autograd.grad(combined,W,retain_graph=True)[0];assert torch.equal(ceW,sumW)
 opt=torch.optim.SGD(m.parameters(),lr=1e-4);opt.zero_grad();combined.backward();oldW=W.detach().clone();opt.step();assert not torch.equal(oldW,W) and all(not torch.equal(saved[n],p) for n,p in zip(d.FISHR_PARAMETERS,params[:2]))
 m.eval()
 with torch.no_grad():before=m.forward_identity_only(x[:2])['id_feat_joint'];W_saved=W.clone();W.zero_();after=m.forward_identity_only(x[:2])['id_feat_joint'];W.copy_(W_saved)
 assert torch.equal(before,after)
 return dict(status='PASS',protocol_negative_tests=negative,actual_backbone_second_derivatives=actual,feature_only_W_independence=True,default_head_normal_joint_update=True,head_fit_updates=0,source_data_read=False,query_read=False,auxiliary_mode_RNG_requires_grad_buffers_restored=True,variance_denominator_derivative_finite_difference='PASS',normalized_objective_scale_invariance='PASS',coefficient=d.COEFFICIENT,seconds=time.time()-started)
def integration():
 import io
 from .observer import Observer
 from experiments.cvs_source_layer_order.probe import cohorts
 y90=torch.tensor([t for t in range(6) for dd in d.DAYS for r in d.RX]);rx90=torch.tensor([r for t in range(6) for dd in d.DAYS for r in d.RX]);day90=torch.tensor([dd for t in range(6) for dd in d.DAYS for r in d.RX]);x90=torch.randn(90,2,256,generator=torch.Generator().manual_seed(517))
 L=dict(y=y90.repeat_interleave(70),rx=rx90.repeat_interleave(70),day=day90.repeat_interleave(70),x=x90.repeat_interleave(70,dim=0),ids=list(range(6300)));old=cohorts(L);ix=[torch.cat((old[2*i],old[2*i+1])) for i in range(35)];warm=[];proof=[]
 for arm in ('SIG_TIME','BONE_COND_FISHR'):
  c=d.config(next(r for r in d.training_rows() if r['arm']==arm));m=api.build(c,'cpu').train();obs=Observer();obs.c=c;obs.model=m;obs.L=L;obs.ix=ix;obs.extra=0;obs.records=[];obs.grad=[];obs.stream=io.StringIO();obs.U_steps=obs.U_packets=obs.U_accept=0;obs.params=[dict(m.named_parameters())[n] for n in d.PARAMETERS];obs.head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;a=d.make_args(c,'cpu')
  for epoch in (1,17):
   torch.manual_seed(766);before={k:v.clone() for k,v in m.state_dict().items()};flags=[v.training for v in m.modules()];out=m.forward_identity_only(x90[:12]);ce=F.cross_entropy(out['tx_logits'],y90[:12],label_smoothing=.01);u=ce*0;obs.steps=(epoch-1)*222+15;returned=obs.base(model=m,args=a,epoch=epoch,batch_idx=16,loss_l=ce,loss_u=u,loss_ent=u,active=False,logits=None,pseudo=torch.empty(0,dtype=torch.long),mask=torch.empty(0,dtype=torch.bool));assert returned is u;rng=torch.get_rng_state().clone();aux,detail=obs.aux(m,obs.steps)
   assert torch.equal(rng,torch.get_rng_state()) and flags==[v.training for v in m.modules()] and all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
   if epoch==1:warm.append(torch.autograd.grad(ce+aux,obs.params+[obs.head]));assert aux==0
   else:
    gw=torch.autograd.grad(aux,obs.head,retain_graph=True,allow_unused=True)[0];assert gw is None;assert torch.equal(torch.autograd.grad(ce,obs.head,retain_graph=True)[0],torch.autograd.grad(ce+aux,obs.head,retain_graph=True)[0]);gg=derivative(aux,obs.params)
    if arm=='SIG_TIME':assert all(g.norm()==0 for g in gg)
    else:assert all(g.norm()>0 for g in gg)
    proof.append(dict(arm=arm,effective_gradient_norms=[float(g.norm()) for g in gg],original_U_loss_returned=True))
 assert all(torch.equal(x,y) for x,y in zip(warm[0],warm[1]))
 from .publish import configure
 from experiments.cvs_report_ssl import publish as transport
 configure();compile(transport.REMOTE.replace('CONFIG',repr(dict(project='/tmp',release='r',run='x',archive='a',sha256='s',commit='c'))),'<remote>','exec')
 assert 'experiments/cvs_backbone_fishr50' in transport.PATHS
 return dict(status='PASS',actual_native_hook=proof,zero_warmup_main_gradients_equal=True,original_U_loss_identity=True,cohorts35=True,remote_script_compile='PASS')
if __name__=='__main__':
 value=checks();value['training_hook']=integration();print(json.dumps(value))
