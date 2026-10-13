"""Focused negatives for source-only probe fitting and dual-readout truth order."""
import copy,json,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d
from . import score,predict,provenance,probe
from .model import build

def reject(fn,kind=ValueError):
 try:fn()
 except kind:return
 raise AssertionError('Expected protocol rejection')

def check():
 torch.set_num_threads(2)
 for row in d.rows():
  c=d.config(row);d.validate(c);a=d.make_args(c,'cpu')
  assert a.epochs==50 and a.label_epochs==33 and a.pseudo_epochs==17 and a.use_ema_teacher and a.use_unlabeled and a.pseudo_temporal_mode=='batch_neighbor' and a.phase1_lr_schedule=='cosine' and not a.use_concat_sat_channel_aug
  changed=copy.deepcopy(c);changed['probe']['fit_role']='V';reject(lambda:d.validate(changed))
 for row in d.rows()[1:5]:
  with patch('torch.load',side_effect=AssertionError('New backbone must not inherit a checkpoint')):build(d.config(row),'cpu')
 # Gradients must reach W but never the cached/frozen backbone feature producer.
 z=torch.randn(12,160,requires_grad=True);y=torch.arange(12)%6
 head,cost=probe.fit(z,y,6,421,d.PROBE)
 assert z.grad is None and cost['backbone_updates']==0 and head.weight.shape==(6,160) and cost['epochs']==10
 for role in ('V','U_s','query','target'):reject(lambda:probe.fit(z,y,6,421,d.PROBE,role))
 from .dispatch import eligible
 later=d.rows()[5]
 with patch('pathlib.Path.is_file',return_value=False):assert not eligible(later)
 proof_value=dict(leave_RX_mean=.95,leave_RX_worst=.9,full_TX={'accuracy':.99})
 with patch('pathlib.Path.is_file',return_value=True),patch.object(provenance,'source_proof',return_value=Path('.')) as pp,patch.object(probe,'proof',return_value=({},proof_value)),patch.object(d,'read',return_value={'final_training_L_CE':.4}):
  assert eligible(later) and pp.call_count==5
 with patch('pathlib.Path.is_file',return_value=True),patch.object(provenance,'source_proof',return_value=Path('.')),patch.object(probe,'proof',return_value=({},proof_value)),patch.object(d,'read',return_value={'final_training_L_CE':.6}):reject(lambda:eligible(later))
 low=dict(proof_value,leave_RX_mean=.92)
 with patch('pathlib.Path.is_file',return_value=True),patch.object(provenance,'source_proof',return_value=Path('.')),patch.object(probe,'proof',side_effect=[({},proof_value),({},low),({},proof_value),({},proof_value),({},proof_value)]),patch.object(d,'read',return_value={'final_training_L_CE':.4}):reject(lambda:eligible(later))
 with patch.object(predict,'preflight',side_effect=ValueError('Second readout incomplete')),patch('experiments.cvs_report_ssl.evaluate.truth_arrays',side_effect=AssertionError('Premature truth access')):reject(lambda:score.score(d.rows()[0]['row_id']))
 original=d.BASE
 (d.ROOT/'local_artifacts').mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='restitution_closure_',dir=d.ROOT/'local_artifacts') as folder:
  d.BASE=Path(folder)
  try:
   for row in d.rows():
    c=d.config(row);p=d.BASE/row['row_id'];(p/'scores').mkdir(parents=True);(p/'probe').mkdir()
    records=[]
    for readout in d.READOUTS:
     rate=.9 if row['arm']=='BASE_REF' else .8;cm=[[0]*6 for _ in range(6)]
     for k in range(6):cm[k][k]=round(28000*rate);cm[k][(k+1)%6]=28000-cm[k][k]
     records.extend(dict(row_id=row['row_id'],arm=row['arm'],model_seed=row['model_seed'],readout=readout,view='clean',dimension='overall' if i==0 else 'receiver',accuracy=rate,macro_f1=rate,query_count=168000,confusion=cm) for i in range(18))
    d.write(p/'scores/metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'))
    d.write(p/'completion.json',dict(status='TRAINED_PROBED_PREDICTED_SCORED',config=c,views=['clean'],readouts=list(d.READOUTS),count_per_readout=168000,metric_records=36,truth_last=True,independent_recount='VERIFIED'))
    d.write(p/'probe/summary.json',dict(config=c))
   value=score.finish();assert value['rows']==15 and value['readouts']==30 and len(value['records'])==540 and not value['goal_84'] and not value['feature_evidence_positive']
   last=d.BASE/d.rows()[-1]['row_id']/'completion.json';v=d.read(last);v['readouts']=['NATIVE_HEAD'];d.write(last,v);reject(score.finish)
  finally:d.BASE=original
 # Both output keys are required; same-row native logits use the deployed feature.
 c=d.config(d.rows()[4]);m=build(c,'cpu').eval();x=torch.randn(2,2,256)
 with torch.no_grad():v=m.forward_identity_only(x);assert v['id_feat_joint'].shape==(2,160) and torch.allclose(v['tx_logits'],m(x),atol=1e-6,rtol=1e-5)
 return dict(status='PASS',rows=15,new_backbones_no_inheritance=True,oldgate_recipe_fixed=True,probe_L_only=True,frozen_backbone_gradient_isolation=True,firstseed_all5_source_gate=True,incomplete_readout_truth_closed=True,all30readout_closure=True,control_above84_does_not_grant_primary_goal=True,shared_native_forward=True,target_access=False)
if __name__=='__main__':print(json.dumps(check()))
