import copy,json,torch
from types import SimpleNamespace
from torch.nn import functional as F
from torch.utils.data import DataLoader
from . import design as d,model as api
from .observer import Observer
from experiments.cvs_neighbor_pair50.checks import Hidden
from experiments.cvs_neighbor_pair50.sampler import replace
from experiments.cvs_signature_stats50.features import export

def checks():
 torch.set_num_threads(2);dataset=Hidden();base=DataLoader(dataset,batch_size=256,shuffle=True,num_workers=0,drop_last=False);configs=[d.config(r) for r in d.training_rows()];models=[api.build(c,'cpu').train() for c in configs];first=models[0].state_dict();assert all(all(torch.equal(v,m.state_dict()[k]) for k,v in first.items()) for m in models)
 def collect(loader):
  torch.manual_seed(2026101450);parts=list(loader);return torch.cat(parts),torch.get_rng_state().clone(),[len(v) for v in parts]
 original,randomstate,lengths=collect(base)
 pair,_state,_len=collect(replace(base,'PAIRS'));random,rstate,rlengths=collect(replace(base,'RANDOM'));assert torch.equal(original,random) and torch.equal(randomstate,rstate) and torch.equal(randomstate,_state) and lengths==rlengths==_len==[256]*221+[124] and torch.equal(pair.sort().values,torch.arange(56700));physical=torch.tensor(dataset.selected.tolist())[pair].reshape(-1,2);assert (physical.diff(dim=1).abs()==1).all()
 x=torch.randn(12,2,256,generator=torch.Generator().manual_seed(1901));y=torch.arange(12)%6;updates=[]
 for c,m in zip(configs,models):
  a=d.make_args(c,'cpu');assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lambda_u,a.lambda_ent)==(50,33,17,.16,.01) and a.use_ema_teacher and a.pseudo_temporal_mode=='batch_neighbor';o=Observer(c)
  fake=SimpleNamespace(_trusted_pseudo_loss_hook=None,_source_feature_auxiliary_hook=None,_build_ssdg_wisig_data=lambda:dict(unlabeled_loader=base));before_rng=torch.get_rng_state().clone()
  with o.native_scope(fake):
   ctx=fake._build_ssdg_wisig_data();assert ctx['unlabeled_loader'].sampler.mode==c['source_neighbor_sampling']['mode'] and not ctx['unlabeled_loader'].sampler.history
   o.initialize(m,c,None);torch.manual_seed(771);l=m.forward_identity_only(x)['tx_logits'];u=m.forward_identity_only(x+.015)['tx_logits'];ce=F.cross_entropy(l,y,label_smoothing=.01);pseudo=u.detach().argmax(1);mask=torch.ones(12,dtype=torch.bool);uce=F.cross_entropy(u,pseudo);ent=-(u.softmax(1)*u.log_softmax(1)).sum(1).mean();o.steps=33*222;rng=torch.get_rng_state().clone();returned=o.observe(model=m,args=a,epoch=34,batch_idx=1,loss_l=ce,loss_u=uce,loss_ent=ent,active=True,logits=u,pseudo=pseudo,mask=mask);assert returned is uce and torch.equal(rng,torch.get_rng_state()) and all(p.grad is None for p in m.parameters());gs=torch.autograd.grad(ce+.16*uce+.01*ent,o.params,retain_graph=True);after=torch.autograd.grad(ce+.16*returned+.01*ent,o.params,retain_graph=True);assert all(torch.equal(a,b) for a,b in zip(gs,after))
   optimizer=torch.optim.SGD(m.parameters(),lr=.001);old=[p.detach().clone() for p in o.params+[m.id_backbone.encoder.core.id_backbone.cls_head.weight]];optimizer.zero_grad();(ce+.16*returned+.01*ent).backward();optimizer.step();assert all(not torch.equal(a,b) for a,b in zip(old,o.params+[m.id_backbone.encoder.core.id_backbone.cls_head.weight]));updates.append(dict(arm=c['arm'],default_head_and_actual_t1_t2_SIGprojection_update=True,original_U_loss_identity_preserved=True))
  assert fake._trusted_pseudo_loss_hook is None and fake._source_feature_auxiliary_hook is None
  m.eval().requires_grad_(False)
  with torch.no_grad():
   z=m.forward_identity_only(x)['id_feat_joint'];f=export(m);assert torch.equal(z,f.forward_features(x));clone=copy.deepcopy(m);clone.id_backbone.encoder.core.id_backbone.cls_head.weight.zero_();assert torch.equal(z,clone.forward_identity_only(x)['id_feat_joint'])
  assert sum(p.numel() for p in f.parameters())==195041
 negative=0
 for key,value in [('epochs',51),('readouts',['FIT_HEAD']),('target_access',True),('model_initialization','target_best')]:
  bad=copy.deepcopy(configs[0]);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid SSL protocol accepted')
 for mode in ['UNKNOWN']:
  try:replace(base,mode)
  except ValueError:negative+=1
  else:raise AssertionError('Unknown mode accepted')
 dataset.hide=False
 try:replace(base,'PAIRS')
 except ValueError:negative+=1
 else:raise AssertionError('Visible U accepted')
 from .publish import configure
 from experiments.cvs_report_ssl import publish as t
 configure();assert all(p in t.PATHS for p in ['experiments/cvs_sigtime_pair50','experiments/cvs_neighbor_pair50','experiments/cvs_source_neighbor_pairs','experiments/cvs_trusted_weight50','experiments/cvs_source_signature_order']);compile(t.REMOTE.replace('CONFIG',repr(dict(project='/tmp',release='r',run='x',archive='a',sha256='s',commit='c'))),'<remote>','exec')
 return dict(status='PASS',protocol_negative_tests=negative,matched_random_order_and_RNG=True,label_blind_physical_pairs_all56700_once=True,tail124=True,same_initial_fixedSIG_models=True,actual_native_updates=updates,observation_no_forward_or_optimizer_mutation=True,actual_feature_export_exact=True,W_independent=True,feature_parameters=195041,synthetic_IQ_only=True,query_read=False,no_separate_head_fitting=True,remote_compile='PASS')
if __name__=='__main__':print(json.dumps(checks()))
