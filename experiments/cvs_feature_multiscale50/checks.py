import json
from copy import deepcopy
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups,MultiscaleFeatureBlock,classifier
from experiments.cvs_e20_cross_style.style import _paths

def check():
 torch.set_num_threads(2);torch.manual_seed(51);x=torch.randn(12,2,256);y=torch.arange(12)%6;records=[];references={}
 original_load=torch.load
 def forbidden(*a,**kw):raise AssertionError('Scratchfeature builder loadedweights')
 torch.load=forbidden
 try:
  for row in d.rows()[:4]:
   c=d.config(row);model=build(c,'cpu');reference=build(d.config(next(r for r in d.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==row['model_seed'])),'cpu')
   assert type(classifier(model)) is type(classifier(reference)) and classifier(model).weight.shape==(6,160) and classifier(model).scale==classifier(reference).scale==30
   assert all(torch.equal(v,classifier(reference).state_dict()[k]) for k,v in classifier(model).state_dict().items())
   q=torch.randn(7,160);assert torch.equal(classifier(model).classify(q),classifier(reference).classify(q))
   changed=model._feature_extractor['changed_layers'];expected=(list(d.TIME_LAYERS) if c['feature_extractor']['time'] else [])+(list(d.PA_LAYERS) if c['feature_extractor']['PA'] else []);assert changed==expected
   modules,paths=_paths(model);b=modules[paths['time_down'].rsplit('.',1)[0]];seen={};handles=[]
   for layer in changed:
    block=getattr(b,layer);assert isinstance(block,MultiscaleFeatureBlock)
    for i,branch in enumerate(block.branches):
     key=f'{layer}:{i}';seen[key]=0
     handles.append(branch.register_forward_hook(lambda m,x,o,k=key:seen.__setitem__(k,seen[k]+1)))
   model.train();out=model.forward_identity_only(x);loss=F.cross_entropy(out['tx_logits'],y);loss.backward()
   for h in handles:h.remove()
   assert out['id_feat_joint'].shape==(12,160) and out['tx_logits'].shape==(12,6) and torch.isfinite(loss) and all(n==1 for n in seen.values())
   for layer in changed:
    for branch in getattr(b,layer).branches:assert branch.weight.grad is not None and torch.isfinite(branch.weight.grad).all() and branch.weight.grad.norm()>0
   args=d.make_args(c,'cpu');assert args.epochs==50 and args.label_epochs==33 and args.pseudo_epochs==17 and args.use_unlabeled and args.use_ema_teacher and args.pseudo_temporal_mode=='batch_neighbor' and args.pseudo_strong_agreement and args.phase1_lr_schedule=='cosine' and args.label_smoothing==.01 and args.lambda_u==.16 and args.lambda_ent==.01 and args.output_dir==c['output_root']
   pg,names=groups(model,c);included={id(p) for group in pg for p in group['params']};assert included=={id(p) for p in model.parameters() if p.requires_grad}
   teacher=deepcopy(model).eval().requires_grad_(False)
   with torch.no_grad():tz=teacher.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   assert tz.shape==(2,6) and torch.isfinite(tz).all() and teacher._identity_input_policy['explicit_CFO_identity_input'] is False
   clone=build(c,'cpu');clone.load_state_dict(model.state_dict(),strict=True);clone.eval();model.eval();assert torch.equal(clone(x),model(x))
   if row['arm']=='OLD_GATE_COS50':references['joint']=out['id_feat_joint'].detach()
   else:assert not torch.allclose(out['id_feat_joint'],references['joint'])
   bad=dict(c,epochs=51)
   try:d.validate(bad)
   except ValueError:pass
   else:raise AssertionError('Epoch51 accepted')
   records.append(dict(arm=row['arm'],trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),changed_layers=changed,all_multiscale_branches_actualforward=True,all_branch_CE_gradients_live=True,classifier_identical_to_same_seed_baseline=True,ordinary_cosine6x160=True))
  from .source import require_budget
  require_budget(11100,11100)
  for logged,actual in ((11100,11099),(11099,11100),(44400,44400)):
   try:require_budget(logged,actual)
   except ValueError:pass
   else:raise AssertionError('Wrong50budget accepted')
  from experiments.cvs_phase1_stack.runtime import blind_sample
  sample=blind_sample((torch.zeros(2,256),3,1,dict(tx=3,tx_i=3,true_tx_i=3,tx_label=3,y_tx=3,pid='source-U-id',rx_i=1,day_i=1)))
  assert sample[1]==-1 and not any(k in sample[3] for k in ('tx','tx_i','true_tx_i','tx_label','y_tx')) and sample[3]['tx_label_visible'] is False
  return dict(status='PASS',records=records,old_gate_EMA_cos_recipe=True,EMA_model_noquery_finite=True,hidden_U_batch_masks_TX=True,wrong_budget_rejected=True,training_checkpoint_loaded=False,epoch51_rejected=True,feature_output_changed=True,target_access=False)
 finally:torch.load=original_load
if __name__=='__main__':print(json.dumps(check()))
