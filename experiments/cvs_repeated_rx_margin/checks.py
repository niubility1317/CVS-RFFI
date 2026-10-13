import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .drift import partition,estimates,refresh,challenges
from .objective import supervised,objective,gradient_diagnostics
from experiments.cvs_finite_margin.objective import robust_loss

def fixture():
 labels=[(tx,r,day,i) for tx in range(6) for r in d.RX for day in d.DAYS for i in range(3)]
 return dict(x=torch.randn(len(labels),2,256),y=torch.tensor([a for a,b,c,i in labels]),rx=torch.tensor([b for a,b,c,i in labels]),day=torch.tensor([c for a,b,c,i in labels]),ids=[f'tx{a}-rx{b}-day{c}-packet{i:03d}' for a,b,c,i in labels])

def _run(device='cpu'):
 torch.set_num_threads(2);torch.manual_seed(7);L=fixture();folds,pdiag=partition(L);c=d.config(next(r for r in d.rows() if r['arm']=='MEASURED_ROBUST'));h=F.normalize(torch.randn(len(folds),160),dim=1)
 stats,diag=estimates(h,L,folds,c,15);changed=h.clone();changed[folds==0]=F.normalize(torch.randn_like(changed[folds==0]),dim=1);other,_=estimates(changed,L,folds,c,15)
 if not torch.equal(stats['delta'][:,:,:,0],other['delta'][:,:,:,0]) or not torch.equal(stats['q'][:,:,:,0],other['q'][:,:,:,0]):raise AssertionError('Anchor fold leaked into drift estimates')
 rc=d.config(next(r for r in d.rows() if r['arm']=='RANDOM_ROBUST'));random,_=estimates(h,L,folds,rc,15)
 if not torch.equal(stats['q'],random['q']) or not torch.allclose(stats['delta'].norm(dim=-1),random['delta'].norm(dim=-1),atol=2e-8) or stats['delta'].norm(dim=-1).max()>.200001:raise AssertionError('Random direction norms/q/radius differ')
 # Perfectly repeated per-cell shifts give q=1; reversing one estimator removes it.
 repeated=torch.zeros_like(h)
 for i,(tx,r,day,f) in enumerate(zip(L['y'],L['rx'],L['day'],folds)):repeated[i,int(tx)]=1.;repeated[i,10]=float(r)/100
 repeated=F.normalize(repeated,dim=1);ss,dd=estimates(repeated,L,folds,c,15)
 if not torch.allclose(ss['q'],torch.ones_like(ss['q']),atol=1e-6):raise AssertionError('Repeated directions not recognized')
 repeated[folds==2,10]*=-1;opposite,_=estimates(repeated,L,folds,c,15)
 if opposite['q'][:,:,:,0].max()>1e-3:raise AssertionError('Contradictory estimators not suppressed')
 m=build(c,device);pg,_=groups(m,c);params=[p for g in pg for p in g['params']];m.train();m.base.eval();modes=[v.training for v in m.modules()];state={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone() if str(device).startswith('cuda') else None
 realstats,realdiag=refresh(m,L,folds,device,c,15)
 if modes!=[v.training for v in m.modules()] or not torch.equal(rng,torch.get_rng_state()) or cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state()) or any(not torch.equal(v,m.state_dict()[k]) for k,v in state.items()):raise AssertionError('Refresh changed model mode/state/RNG')
 x=L['x'][:128].to(device);y=L['y'][:128].to(device);rx=L['rx'][:128].to(device);day=L['day'][:128].to(device);ff=folds[:128].to(device);m.train();out=m.forward_identity_only(x);delta,q=challenges(realstats,y,rx,day,ff,15)
 ce=F.cross_entropy(out['tx_logits'],y);original=torch.autograd.grad(ce,params,allow_unused=True,retain_graph=True)
 for strength,ddelta in [(0.,delta),(.25,torch.zeros_like(delta))]:
  cc,aux,net,detail=supervised(m,out,y,ddelta,'percompetitor worst margin',strength);actual=torch.autograd.grad(cc+net,params,allow_unused=True,retain_graph=True)
  if not torch.equal(ce,cc+net) or any((a is None)!=(b is None) or a is not None and not torch.equal(a,b) for a,b in zip(original,actual)):raise AssertionError('Zero weight/displacement loss or supervised gradient changed')
 candidates=torch.randn(5,17,6,device=device);labels=torch.randint(6,(17,),device=device);env,worst=robust_loss(candidates,labels)
 direct=torch.logsumexp(-worst,dim=1);allce=torch.stack([F.cross_entropy(z,labels,reduction='none') for z in candidates])
 if (direct+1e-5<allce.max(0).values).any():raise AssertionError('Percompetitor envelope failed per-packet CE upper bound')
 records=[]
 for arm in d.ARMS:
  ac=d.config(next(r for r in d.rows() if r['arm']==arm));m=build(ac,device);gp,_=groups(m,ac);ap=[p for g in gp for p in g['params']];m.train();st={k:v.to(device) for k,v in (random if arm=='RANDOM_ROBUST' else ss).items()};loss,metrics,details=objective(m,x,y,rx,None,None,None,None,ac,1333,15,st,ff,day);audit=gradient_diagnostics(loss,details.pop('_net_tensor'),ap,m.head.weight);loss.backward()
  if not all(p.grad is None or torch.isfinite(p.grad).all() for p in ap) or arm!='MATCHED_BASE' and (audit['encoder']['net_added_norm']<=0 or audit['cosine_head']['net_added_norm']<=0):raise AssertionError('Actual encoder/head mechanism gradient missing')
  m.eval()
  with torch.no_grad():
   out=m.forward_identity_only(x[:2]);expected=30*F.linear(F.normalize(out['id_feat_joint'],dim=1,eps=1e-4),F.normalize(m.head.weight,dim=1,eps=1e-4))
  if not torch.equal(out['tx_logits'],expected) or out['id_feat_joint'].shape!=(2,160):raise AssertionError('Ordinary160 cosine interface changed')
  records.append(dict(arm=arm,parameters=sum(p.numel() for p in ap),net_gradient=audit,mechanism=metrics))
 # Actual CE+hidden-U SSL loss/gradient matches the established UNIFORM_SSL objective.
 from experiments.cvs_report_mechanisms.feedback import objective as previous,parameter_selection
 from experiments.cvs_stage2_hybrid.semisupervised import VisitPassBank
 ac=d.config(next(r for r in d.rows() if r['arm']=='MATCHED_BASE'));m=build(ac,device).train();old=copy.deepcopy(m);gp,_=groups(m,ac);op,_=groups(old,ac);ap=[p for g in gp for p in g['params']];bp=[p for g in op for p in g['params']];_,fp=parameter_selection(old,op);optimizer=torch.optim.AdamW(op);ux=torch.randn_like(x);ids=[str(i) for i in range(128)];passes=torch.ones(128,dtype=torch.long);first=VisitPassBank(2,.95);second=VisitPassBank(2,.95);st={k:v.to(device) for k,v in ss.items()}
 torch.manual_seed(12345);value,_,_=objective(m,x,y,rx,ux,ids,passes,first,ac,1333,6,st,ff,day)
 torch.manual_seed(12345);pc=d.b.control(ac['model_seed']);expected,_,_=previous(old,x,y,x[:26],y[:26],ux,ids,passes,second,optimizer,fp,pc,1333)
 ag=torch.autograd.grad(value,ap,allow_unused=True);bg=torch.autograd.grad(expected,bp,allow_unused=True)
 if not torch.equal(value,expected) or any((a is None)!=(b is None) or a is not None and not torch.equal(a,b) for a,b in zip(ag,bg)):raise AssertionError('Matched baseline actual SSL loss/gradient differs')
 bad=copy.deepcopy(c);bad['drift']['radius_max']=.4
 try:d.validate(bad)
 except ValueError:pass
 else:raise AssertionError('Unregistered radius accepted')
 try:partition(dict(L,day=torch.zeros_like(L['day'])))
 except ValueError:pass
 else:raise AssertionError('Illegal source day accepted')
 return dict(status='PASS',records=records,crossfit_anchor_excluded=True,random_norm_q_matched=True,refresh_state_mode_RNG_preserved=True,zero_strength_loss_gradient_exact=True,UNIFORM_SSL_loss_gradient_exact=True,envelope_per_packet_bound=True,query_access=False,U_truth_read=False)
def run(device='cpu'):
 # Deterministic convolution is only a verification setting; the run's original FP32 policy remains fixed.
 with torch.backends.cudnn.flags(deterministic=True):return _run(device)
if __name__=='__main__':print(json.dumps(run()))
