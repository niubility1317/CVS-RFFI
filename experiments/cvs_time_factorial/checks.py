import copy,json,torch
from torch.nn import functional as F
from . import design as d
from .model import build
from experiments.cvs_transfer_features.model import from_phase1
from experiments.cvs_default_head.predict import logits as default_logits

def checks():
 torch.set_num_threads(2);models=[];rng=[];reports=[]
 first=[r for r in d.training_rows() if r['model_seed']==d.SEEDS[0]]
 for row in first:
  torch.manual_seed(4191);c=d.config(row);m=build(c);models.append(m);rng.append(torch.get_rng_state().clone())
 assert all(torch.equal(rng[0],v) for v in rng)
 # Only dilation and activation differ: all tensors, including the default W, match.
 state=models[0].state_dict()
 assert all(all(torch.equal(v,m.state_dict()[k]) for k,v in state.items()) for m in models[1:])
 counts=[sum(p.numel() for p in m.id_backbone.encoder.parameters()) for m in models]
 assert counts==[237697]*4
 from experiments.cvs_feature_multiscale50.model import build as historical_build
 hc=d.old.config(dict(row_id=f"TIME_MS50-s{d.SEEDS[0]}",arm='TIME_MS50',model_seed=d.SEEDS[0]))
 torch.manual_seed(d.SEEDS[0]);historical_build(hc,'cpu');historic_rng=torch.get_rng_state().clone()
 torch.manual_seed(d.SEEDS[0]);build(d.config(first[1]));new_rng=torch.get_rng_state().clone()
 assert not torch.equal(historic_rng,new_rng)
 torch.manual_seed(1071);x=torch.randn(6,2,256);y=torch.arange(6)
 outputs=[]
 for row,m in zip(first,models):
  c=d.config(row);b=m.id_backbone.encoder.core.id_backbone
  for name in ('pa_b1','pa_b2','pa_b3'):assert not hasattr(getattr(b,name),'branches')
  m.train();m.zero_grad(set_to_none=True)
  loss=F.cross_entropy(m.forward_identity_only(x)['tx_logits'],y);loss.backward()
  targets={n:p for n,p in m.named_parameters() if any(n.startswith('id_backbone.encoder.core.id_backbone.'+layer+'.') for layer in ('t1','t2','t3'))}
  assert targets and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in targets.values()),[(n,None if p.grad is None else float(p.grad.norm())) for n,p in targets.items()]
  assert b.cls_head.weight.grad is not None and b.cls_head.weight.grad.norm()>0
  from .model import installed
  with installed(c) as native:
   parameters=native._optimizer_parameters(m,None)
   assert all(any(p is v for v in parameters) for p in targets.values())
   teacher=copy.deepcopy(m).eval().requires_grad_(False);before={n:p.clone() for n,p in teacher.named_parameters() if n in targets}
   opt=torch.optim.AdamW(parameters,lr=.0002,weight_decay=.0001);opt.step()
   after=dict(m.named_parameters());changed={n:float((after[n]-v).detach().abs().max()) for n,v in before.items()}
   assert all(v>0 for v in changed.values())
   native._update_ema_model(teacher,m,.999,versioned=True)
   assert all(not torch.equal(v,dict(teacher.named_parameters())[n]) for n,v in before.items())
  m.eval().requires_grad_(False);features=from_phase1(m).eval().requires_grad_(False)
  with torch.no_grad():
   result=m.forward_identity_only(x);export=features.forward_features(x)
   assert torch.equal(result['id_feat_joint'],export)
   assert torch.equal(default_logits(m,x),result['tx_logits'])
   assert torch.isfinite(default_logits(m,torch.zeros(2,2,256))).all()
  assert features.classifier_parameter_elements_removed==960 if hasattr(features,'classifier_parameter_elements_removed') else not any(n.endswith('cls_head.weight') for n,_ in features.named_parameters())
  outputs.append(result['id_feat_joint'])
  reports.append(dict(arm=row['arm'],identity_parameters=counts[0],changed_TIME_parameter_tensors=len(changed),minimum_numeric_parameter_update=min(changed.values()),optimizer_membership=True,EMA_update=True,joint160_export_exact=True,default_head_joint_gradient=True,separate_head_updates=0))
 assert not torch.equal(outputs[0],outputs[2])
 from .monitor import Monitor
 monitor=Monitor(d.config(first[0]),lambda *a:None)
 values=dict(legacy_unlabeled_active=True,phase='pseudo',epoch=34,mask=torch.tensor([True,False,True,False]),pseudo=torch.tensor([0,1,2,3]),receiver_u=torch.tensor([1,1,3,3]),day_u=torch.tensor([1,1,2,2]),out_s=dict(tx_logits=F.one_hot(torch.tensor([0,2,2,4]),6).float()),y_u=torch.full((4,),-1))
 monitor.observe(values)
 assert sum(r['count'] for r in monitor.epochs[34].values())==4 and sum(r['accepted'] for r in monitor.epochs[34].values())==2
 invalid=dict(values,y_u=torch.zeros(4))
 try:monitor.observe(invalid)
 except ValueError:pass
 else:raise AssertionError('U truth diagnostic accepted')
 negatives=1
 for mutation in [dict(epochs=51),dict(label_epochs=32),dict(readouts=['FRESH_COSINE_ALL_L']),dict(classifier_training='head only'),dict(checkpoint_sources=['unknown']),dict(target_access=True)]:
  c=copy.deepcopy(d.config(first[0]));c.update(mutation)
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Unregistered config accepted')
 return dict(status='PASS',synthetic_only=True,real_IQ_or_truth_read=False,rows=reports,identical_initial_parameter_values=True,outer_CPU_RNG_same=True,historical_TIME_RNG_differs=True,same_initial_function_claim=False,negative_checks=negatives)
if __name__=='__main__':print(json.dumps(checks()))
