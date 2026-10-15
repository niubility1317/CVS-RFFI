"""Focused A-F protocol negatives, real-network gradients/cache/EMA, physical AdamW equivalence."""
import copy,json,math,sys
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .optimizer import build,specification,clip_physical,PREFIX,NAMES,QNAMES
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
W='id_backbone.encoder.core.id_backbone.cls_head.weight'
def configuration(arm):return d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==d.SEEDS[0]))
def scalar_equivalence():
 maximum=0.;wrong_errors=[]
 for origin in (25000.,1000000.,19000000.):
  raw=api.build(configuration('C'),'cpu').double();center=api.build(configuration('E'),'cpu').double();rp=dict(raw.named_parameters());qp=dict(center.named_parameters());sm=center.get_submodule(PREFIX[:-1]);origins=(sm.low_origin_hz,sm.band_origin_hz)
  with torch.no_grad():
   for i,(pn,qn,o) in enumerate(zip(NAMES,QNAMES,origins)):rp[pn].fill_(origin+i*7000.);o.copy_(rp[pn]);qp[qn].zero_()
  opts=dict(lr=.0002,eps=1e-8,weight_decay=.03,foreach=False);a=build(raw,list(raw.parameters()),'NORMALIZED_UNITS',**opts);b=build(center,list(center.parameters()),'CENTER_UNITS',**opts);wrong=[o.clone() for o in origins]
  for step in range(100):
   alpha=.0002*(.1+.9*(1+math.cos(math.pi*step/100))/2)
   for opt in (a,b):
    for g in opt.param_groups:g['lr']=alpha
   for i,(pn,qn) in enumerate(zip(NAMES,QNAMES)):
    g=torch.full_like(rp[pn],math.sin(step*.3+i)*1e-5+1e-6);rp[pn].grad=g.clone();qp[qn].grad=g*25000.
   rp[W].grad=torch.full_like(rp[W],.4);qp[W].grad=rp[W].grad.clone()
   na=torch.nn.utils.clip_grad_norm_(raw.parameters(),5.);nb=clip_physical(b,center.parameters(),5.);assert abs(float(na-nb))<1e-11
   a.step();b.step()
   for pn,qn,o in zip(NAMES,QNAMES,origins):maximum=max(maximum,float((rp[pn]-(o+25000.*qp[qn])).abs().max()))
  assert b._sinc_physical_clip_calls==100 and b.param_groups[1]['weight_decay']==0.
  # Analytic missing-origin decay counterexample: same alpha/lambda over 100 steps, no adaptive motion.
  decay=math.prod(1-.0002*(.1+.9*(1+math.cos(math.pi*k/100))/2)*.03 for k in range(100));wrong_errors.append(origin*(1-decay))
 assert maximum<1e-6 and min(wrong_errors)>1.
 return dict(origins_Hz=[25000.,1000000.,19000000.],steps_each=100,dtype='float64',maximum_physical_update_error_Hz=maximum,missing_origin_counterexample_error_Hz=wrong_errors,physical_global_clip_norm_equal=True,cosine_lr_mapping=True)

def run(device='cpu'):
 torch.set_num_threads(2);negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (ValueError,RuntimeError,TypeError,KeyError):negative+=1;return
  raise AssertionError('Illegal matrix accepted')
 cs=[configuration(a) for a in d.ARMS]
 for key,value in [('model_seed',9),('epochs',51),('label_epochs',32),('pseudo_epochs',18),('head_fit_updates',1),('separate_head_training',True),('test_views',['practical_high']),('source_contract','other-source'),('model_initialization','pretrained')]:
  bad=copy.deepcopy(cs[2]);bad[key]=value;reject(lambda:d.validate(bad))
 for key,value in [('alpha',0.),('label_smoothing',0.),('use_aug',False),('sinc_coordinates','OTHER')]:
  bad=copy.deepcopy(cs[2]);bad['training_mechanism'][key]=value;reject(lambda:d.validate(bad))
 for key,value in [('Sinc_optimizer','STACKED_MULTIPLIERS'),('MEAN_VAR',True),('COV',True),('ensemble',True)]:
  bad=copy.deepcopy(cs[4]);bad['combination'][key]=value;reject(lambda:d.validate(bad))
 reject(lambda:d.make_args(cs[0],'cpu'))
 with numerical_context(d.FULL_FP32_POLICY):
  models=[api.build(c,device) for c in cs];original=api.original.build(d.reference_config(d.SEEDS[0]),device);initial=[{n:v.detach().clone() for n,v in m.state_dict().items()} for m in models];ostate=original.state_dict()
  for index in (0,2,3,4):
   state=api.physical_state(initial[index]);assert set(state)==set(ostate) and all(torch.equal(v,ostate[n]) for n,v in state.items())
  for index in (1,5):
   pool=models[index].id_backbone.encoder.sequence_pool
   assert pool.modes==('SIG_TIME',)*3 and len({id(v.local_projection.weight) for v in pool.views})==3 and pool.output_projection.in_features==336
  torch.manual_seed(2026101551);raw=torch.randn(128,2,256,device=device);aug=raw*.7+.13;y=torch.arange(128,device=device)%6;records=[];grads=[];updates=[]
  for c,m,state in zip(cs,models,initial):
   m.train();torch.manual_seed(2026101552)
   class Owner:pass
   owner=Owner();owner.model=m;owner.optimizer=None
   from .optimizer import native_scope
   with native_scope(owner,c['sinc_optimizer_units']['mode']):
    opt=torch.optim.AdamW(list(m.parameters()),lr=.0002,eps=1e-8,weight_decay=.0001,foreach=False)
    with m._phase1_labeled_feature_context():out=m(aug,y_tx=y,return_aux=True)
    assert torch.equal(m._legacy_actual_L_input,aug) and not torch.equal(m._legacy_actual_L_input,raw)
    ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);extra,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra={},domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128);loss=ce+extra;loss.backward()
    assert m._legacy_actual_L_input is None and m._legacy_calls==m._feature_l_steps==1 and m._legacy_extra_packets==128
    if c['arm'] in ('B','F'):assert all(v.local_projection.weight.grad is not None and v.local_projection.weight.grad.norm()>0 for v in m.id_backbone.encoder.sequence_pool.views)
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.);named=dict(m.named_parameters());grads.append({n:p.grad.detach().clone() for n,p in named.items() if p.grad is not None});opt.step();assert opt._sinc_unit_steps==1
   updates.append({n:p.detach().clone() for n,p in named.items()});teacher=copy.deepcopy(m).eval();assert teacher._legacy_original_labeled_scope.__self__ is teacher and teacher._legacy_actual_L_input is None
   # Saved student/ownEMA use identical coordinate keys and origin buffers.
   teacher.load_state_dict(m.state_dict(),strict=True)
   with torch.no_grad():
    m.eval();one=m.forward_identity_only(aug[:2])['id_feat_joint'];ema=teacher.forward_identity_only(aug[:2])['id_feat_joint'];deployed=api.export(m)(aug[:2]);assert torch.equal(one,ema) and torch.equal(one,deployed)
   sm=m.get_submodule(PREFIX[:-1]);leaf=sm.low_coordinate if hasattr(sm,'low_coordinate') else sm.low_hz_
   with torch.no_grad():
    sm._filters(leaf.device,leaf.dtype);leaf.add_(.01 if hasattr(sm,'low_coordinate') else 1000.);cached=sm._filters(leaf.device,leaf.dtype).clone()
    with fresh_parameter_caches(m):fresh=sm._filters(leaf.device,leaf.dtype).clone()
    assert torch.equal(cached,fresh)
   records.append(dict(arm=c['arm'],feature_parameters=sum(p.numel() for n,p in named.items() if n.startswith('id_backbone.encoder.') and n!=W),actual_ownEMA_and_export_equal=True,leaf_version_cache_fresh=True,physical_clip_calls=opt._sinc_physical_clip_calls,Sinc_lr=opt.param_groups[-1]['lr'],Sinc_eps=opt.param_groups[-1]['eps'],Sinc_AdamW_decay=opt.param_groups[-1]['weight_decay'],same_actual_augmented_IQ=True,defaultW_changed=not torch.equal(named[W],state[W])))
  assert max(float((grads[0][n]-grads[2][n]).abs().max()) for n in grads[0])<1e-5
  ordinary=[n for n in grads[2] if n not in NAMES];nonSinc=max(float((updates[2][n]-updates[4][n]).abs().max()) for n in ordinary);assert nonSinc<1e-6
  assert torch.equal(updates[0][W],updates[2][W]) and torch.allclose(updates[2][W],updates[4][W],atol=1e-7,rtol=1e-6)
  opts=dict(lr=.0002,eps=1e-8,weight_decay=.0001);reject(lambda:specification(models[4],list(models[4].parameters()),'NORMALIZED_UNITS',opts));reject(lambda:specification(models[2],list(models[2].parameters()),'CENTER_UNITS',opts));reject(lambda:specification(models[4],list(models[4].parameters())[:-1],'CENTER_UNITS',opts))
 return dict(status='PASS',device=device,negative_cases=negative,original_single_SIGTIME_initial_state_exact=True,physical_ordinary_update_maximum_difference=nonSinc,same_actual_augmented_IQ=True,network_checks=records,scalar_equivalence=scalar_equivalence(),scratch_initialization=True,actual_IQ_reads=0,query_read=False,head_fit_updates=0,default_head_joint_training=True,limitation='Synthetic implementation checks; no network performance claim')
if __name__=='__main__':
 value=run(sys.argv[1] if len(sys.argv)>1 else 'cpu');text=json.dumps(value,indent=2,allow_nan=False)
 if len(sys.argv)>2:open(sys.argv[2],'w',encoding='utf8').write(text+'\n')
 print(text)
