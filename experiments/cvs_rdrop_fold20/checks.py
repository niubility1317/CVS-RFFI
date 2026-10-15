import copy,json,torch
from torch.nn import functional as F
from . import design as d,model as api
def reject(call):
 try:call()
 except (ValueError,RuntimeError):return
 raise AssertionError('Illegal source action accepted')
def main():
 torch.set_num_threads(2);torch.manual_seed(8)
 a=torch.randn(7,6,requires_grad=True);b=torch.randn(7,6,requires_grad=True);y=torch.arange(7)%6
 for alpha in (0.,1.):
  total,paired,kl,ce1,ce2=api.terms(a,b,y,alpha)
  ref=.5*(F.kl_div(F.log_softmax(a,1),F.softmax(b,1),reduction='batchmean')+F.kl_div(F.log_softmax(b,1),F.softmax(a,1),reduction='batchmean'))
  assert torch.allclose(kl,ref,atol=1e-7) and torch.allclose(ce1+.5*(ce2-ce1)+alpha*kl,total)
  g1=torch.autograd.grad(total,(a,b),retain_graph=True);g2=torch.autograd.grad(ce1+.5*(ce2-ce1)+alpha*kl,(a,b),retain_graph=True)
  assert all(torch.allclose(x,z,atol=1e-7) for x,z in zip(g1,g2)) and all(x.norm()>0 for x in g1)
 assert abs(float(api.terms(a,a,y,1.)[2].detach()))<1e-8
 reject(lambda:api.terms(a,b,torch.full_like(y,-1),1.));reject(lambda:api.terms(a,b,y,.5))
 for arm in d.ARMS:
  c=d.config(arm);d.validate(c);d.make_args(c,'cpu')
  for key,value in [('model_seed',2026092702),('head_fit_updates',1),('test_views',['clean']),('fold',{})]:
   bad=dict(c);bad[key]=value;reject(lambda: d.validate(bad))
 c=d.config('RDROP');m=api.build(c,'cpu');m.train();W=api.classifier(m).weight;teacher=copy.deepcopy(m).eval()
 x=torch.randn(128,2,256);labels=torch.arange(128)%6
 with m._phase1_labeled_feature_context():out=m.forward_identity_only(x)
 ce=F.cross_entropy(out['tx_logits'],labels);before=torch.get_rng_state().clone()
 correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=labels,extra={},domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
 assert not torch.equal(before,torch.get_rng_state()) and m._feature_l_steps==m._rdrop_calls==1 and m._rdrop_extra_packets==128
 total=ce+correction;params=dict(m.named_parameters());names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight')
 grads=torch.autograd.grad(total,[params[n] for n in names]+[W]);assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)
 assert api.classifier(m).weight is W and all(p.grad is None for p in m.parameters()) and teacher._rdrop_calls==0
 reject(lambda:teacher._source_labeled_feature_hook(model=teacher,labeled_output=out,labels=labels,extra={},domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x))
 fixtures={arm:dict(source_health=True,exports=dict(FINAL=dict(count=1260,mean_CE=1.,errors=100))) for arm in d.ARMS}
 assert not d.qualify(fixtures)['qualified'];fixtures['RDROP']['exports']['FINAL'].update(errors=99)
 assert d.qualify(fixtures)['qualified'];fixtures['RDROP']['exports']['FINAL']['mean_CE']=1.1;assert not d.qualify(fixtures)['qualified']
 from .resources import choose
 caps={g:dict(pids=set(),free_MiB=24000,uuid=str(g)) for g in range(8)}
 caps[0]['pids']={1};assert choose(caps)==1
 caps[1]['pids']={2};assert choose(caps)==2
 assert choose({0:dict(pids={1,2,3},free_MiB=24000)}) is None
 assert choose({7:dict(pids=set(),free_MiB=11999)}) is None
 assert choose({7:dict(pids=set(),free_MiB=12000)})==7
 print(json.dumps(dict(status='PASS',objective_and_gradient_equivalence=True,negative_cases=11,actual_model_synthetic_L_packets=256,body_and_defaultW_gradients_nonzero=True,head_fit_updates=0,optimizer_steps=0,checkpoint_loads=0,real_IQ_reads=0,query_reads=0)))
if __name__=='__main__':main()
