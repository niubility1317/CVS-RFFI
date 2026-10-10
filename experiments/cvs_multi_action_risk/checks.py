"""Configuration, input role, freeze and scoring-negative checks for this release."""
import json,tempfile
from pathlib import Path
from . import design as d

def checks():
 assert len(d.rows())==48 and len(set(r['row_id'] for r in d.rows()))==48
 assert len(d.SEEDS)==3
 for r in d.rows():
  c=d.config(r);a=d.make_args(c,'cpu')
  assert c['features']==[] and not a.use_unlabeled and not a.use_ema_teacher
  assert a.lambda_u==a.lambda_ent==a.label_smoothing==0 and a.label_epochs==200 and a.pseudo_epochs==0
  assert not a.use_concat_sat_channel_aug and a.sat_training_mode=='disabled' and a.lambda_sat_cls==a.lambda_sat_cons==0
  assert a.phase1_lr_schedule=='cosine' and not a.baseline_ckpt and not a.teacher_ckpt and a.from_scratch
  changed=dict(c,features=['pseudo'])
  try:d.validate(changed)
  except ValueError:pass
  else:raise AssertionError('Changed recipe accepted')
 assert d.plan('LT_label_free')['action_label_free'] and d.plan('LT_label_free_U')['action_label_free']
 assert [a for a in d.ARMS if d.plan(a)['source_u_fit']]==['LT_label_free_U']
 from . import evaluate as e
 fixtures=[]
 for arm in ('native','LTR'):
  for seed in d.SEEDS:
   for view in e.VIEWS:
    for dimension in ('overall','receiver'):
     fixtures.append(dict(stage='multi_action_risk',arm=arm,model_seed=seed,row_id=arm+'-s'+str(seed),
       view=view,dimension=dimension,accuracy=.5+(.1 if arm=='LTR' else 0),macro_f1=.4+(.1 if arm=='LTR' else 0)))
 aggregate=e.summarize(fixtures)
 assert len([r for r in aggregate if r['view']=='six_practical_mean'])==2
 pairs=e.paired_results(fixtures)
 p=next(r for r in pairs if r['treatment']=='LTR' and r['control']=='native' and r['view']=='six_practical_mean' and r['metric']=='accuracy')
 assert abs(p['mean_difference']-.1)<1e-10 and p['seed_count']==3 and p['positive_seeds']==3
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);rid=d.rows()[0]['row_id'];c=d.config(d.rows()[0]);old_base,old_row=e.BASE,e.EVAL_ROW
  try:
   e.BASE=root;e.EVAL_ROW=rid
   try:e.snapshot(d)
   except FileNotFoundError:pass
   else:raise AssertionError('Missing source freeze accepted')
   d.write(root/rid/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,target_access=False))
   assert e.snapshot(d)==[c]
   original=e.physical_ids;e.physical_ids=lambda unused:None
   try:
    try:e.prediction_preflight(d,[c])
    except FileNotFoundError:pass
    else:raise AssertionError('Incomplete prediction accepted before truth')
   finally:e.physical_ids=original
  finally:e.BASE=old_base;e.EVAL_ROW=old_row
 from .publish import REMOTE
 compile(REMOTE.replace('CONFIG',repr(dict(project='example'))),'remote-release','exec')
 return dict(status='PASS',rows=48,seeds=3,baseline='pureCE_noEMA_noU_noSat',changed_config_rejected=True,
             source_freeze_required=True,incomplete_predictions_keep_truth_closed=True)

def preflight(output):
 from .action_checks import run_checks
 from .receiver_checks import checks as rchecks
 from .runtime_checks import native_smoke
 from .diagnostics import run
 out=Path(output);out.mkdir(exist_ok=False)
 results=dict(config=checks(),actions=run_checks('cpu'),receiver=rchecks('cpu'),
   checkpoints=[run(seed,smoke=True) for seed in d.SEEDS])
 native_smoke(out/'native','cpu')
 for row in d.rows():
  if d.read(d.ROOT/'experiments/cvs_multi_action_risk/configs'/(row['row_id']+'.json'))!=d.config(row):raise ValueError('Archived config drift')
 d.write(out/'completion.json',dict(status='PASS',target_access=False,results=results))

if __name__=='__main__':print(json.dumps(checks()))
