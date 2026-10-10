"""Focused scientific-routing, fixed matrix and prediction-boundary checks."""
import json,tempfile
from pathlib import Path
from . import design as d

def checks():
 assert len(d.rows())==48 and len({r['row_id'] for r in d.rows()})==48 and len(d.SEEDS)==3
 for r in d.rows():
  c=d.config(r,.05 if r['arm'] in d.SELECTED_ARMS else None);a=d.make_args(c,'cpu')
  assert not a.use_unlabeled and not a.use_ema_teacher and a.lambda_u==a.lambda_ent==a.label_smoothing==0
  assert a.label_epochs==200 and a.pseudo_epochs==0 and a.phase1_lr_schedule=='cosine'
  assert not a.use_concat_sat_channel_aug and a.sat_training_mode=='disabled' and a.lambda_sat_cls==a.lambda_sat_cons==0
  assert a.from_scratch and not a.baseline_ckpt and not a.teacher_ckpt and not c['arm_plan']['source_u_fit']
  changed=dict(c,features=['pseudo'])
  try:d.validate(changed)
  except ValueError:pass
  else:raise AssertionError('Changed config accepted')
 for ratio in (.03,.05,.1):
  selected=dict(status='SOURCE_SELECTION_FROZEN',selected_ratio=ratio,target_scores_consumed=False)
  assert len(d.test_rows(selected))==36
  search={r['arm'] for r in d.test_rows(selected) if r['arm'] in d.SEARCH_ARMS}
  assert search=={'LTR_S_F'+str(round(ratio*100)),'LTR_S_AF'+str(round(ratio*100))}
 from . import evaluate as e
 with tempfile.TemporaryDirectory() as tmp:
  previous=(d.BASE,e.BASE,e.EVAL_ROW);d.BASE=e.BASE=Path(tmp)
  try:
   r=d.rows()[0];c=d.config(r);e.EVAL_ROW=r['row_id']
   try:e.snapshot(d)
   except FileNotFoundError:pass
   else:raise AssertionError('Missing source freeze accepted')
   d.write(e.BASE/r['row_id']/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,target_access=False))
   assert e.snapshot(d)==[c]
   e.EVAL_ROW='LTR_S_AF3-s'+str(d.SEEDS[0])
   try:e.snapshot(d)
   except FileNotFoundError:pass
   else:raise AssertionError('Candidate target allowed before source selection')
   selection=dict(status='SOURCE_SELECTION_FROZEN',selected_ratio=.05,target_scores_consumed=False)
   d.write(d.BASE/'source_selection.json',selection)
   try:e.snapshot(d)
   except ValueError:pass
   else:raise AssertionError('Unselected source candidate target allowed')
   old=e.physical_ids;e.physical_ids=lambda unused:None
   try:
    try:e.prediction_preflight(d,[c])
    except FileNotFoundError:pass
    else:raise AssertionError('Incomplete predictions accepted before truth')
   finally:e.physical_ids=old
  finally:d.BASE,e.BASE,e.EVAL_ROW=previous
 fixtures=[]
 for arm in ('native','LTR_S_AF5'):
  for seed in d.SEEDS:
   for view in e.VIEWS:
    for dimension in ('overall','receiver'):
     fixtures.append(dict(stage='feature_disentangle',arm=arm,model_seed=seed,row_id=arm+'-s'+str(seed),view=view,
       dimension=dimension,accuracy=.5+(.1 if arm!='native' else 0),macro_f1=.4+(.1 if arm!='native' else 0)))
 assert len([r for r in e.summarize(fixtures) if r['view']=='six_practical_mean'])==2
 pair=next(r for r in e.paired_results(fixtures) if r['treatment']=='LTR_S_AF5' and r['control']=='native' and r['metric']=='accuracy' and r['view']=='six_practical_mean')
 assert abs(pair['mean_difference']-.1)<1e-9 and pair['seed_count']==3
 # Parameter capacity uses active trainable graph, never padding parameters.
 import torch
 from experiments.cvs_multi_action_risk.actions import StateAction,SharedActionCore
 from experiments.cvs_multi_action_risk.receiver import ReceiverDistributionAction
 def size(hidden,shared):
  core=SharedActionCore(hidden) if shared else None
  modules=torch.nn.ModuleList([StateAction('linear',hidden,core),StateAction('temporal',hidden,core),
     ReceiverDistributionAction(shared_core=core if core is not None else SharedActionCore(hidden))])
  return sum(p.numel() for p in modules.parameters())
 independent,shared=size(48,False),size(50,True)
 assert abs(shared-independent)/independent<=d.FEATURE_RECIPE['capacity_relative_tolerance']
 d.require_budget(44400,44400)
 from .publish import REMOTE
 compile(REMOTE.replace('CONFIG',repr(dict(project='example'))),'feature_remote','exec')
 return dict(status='PASS',trained=48,tested=36,source_only=12,seeds=3,source_selection_gate=True,
   target_truth_last=True,pure_ce=True,shared_parameters=shared,independent_parameters=independent,
   shared_difference_fraction=(shared-independent)/independent)

def preflight(output):
 from .runtime_checks import native_smoke
 from .runtime import installed
 import torch
 out=Path(output);out.mkdir(exist_ok=False)
 result=checks()
 c=d.config(d.rows()[0]);a=d.make_args(c,'cpu');a.num_domains=5;a.input_len=256
 with installed(c,training=False) as native:
  model=native.build_baseline_model(a,torch.device('cpu')).eval()
  torch.save(dict(model=model.state_dict(),scratch_only=True,checkpoint_sources=[]),out/'scratch_checkpoint.pth')
  state=torch.load(out/'scratch_checkpoint.pth',map_location='cpu',weights_only=False)
  model.load_state_dict(state['model'],strict=True)
  with torch.no_grad():logits=model(torch.zeros(2,2,256))
  if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Real scratch checkpoint smoke failed')
  del model,state
 result['scratch_checkpoint_no_query_smoke']='PASS'
 # Real architecture scratch checkpoint roundtrip and actual native training,
 # no historical initialization and no query read.
 native_smoke(out/'native','cpu',arms=('native','LTR_S_AF5','joint_shared'))
 for row in d.rows():
  if d.read(d.ROOT/'experiments/cvs_feature_disentangle/configs'/(row['row_id']+'.json'))!=d.config(row):
   raise ValueError('Archived configuration differs')
 d.write(out/'completion.json',dict(status='PASS',target_access=False,results=result))

if __name__=='__main__':print(json.dumps(checks()))
