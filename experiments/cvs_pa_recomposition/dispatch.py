import argparse,os
from pathlib import Path
from . import design as d
from .provenance import source_proof
from experiments.cvs_feature_sequence50 import dispatch as shared
PACKAGE='experiments.cvs_pa_recomposition'
def eligible(row,design=None,proof=None):
 if row['model_seed']==d.SEEDS[0]:return True
 first=[d.config(r) for r in d.rows() if r['model_seed']==d.SEEDS[0]]
 if any(not (Path(c['output_root'])/'completion.json').is_file() for c in first):return False
 evidence={}
 for c in first:
  p=source_proof(c);done=d.read(p/'completion.json');actual=d.read(p/'PA_execution.json')
  if not 0<=done['final_training_L_CE']<=.5 or actual['mean_common_cohort_coverage']<.25:raise ValueError('PA source health/coverage falsification failed:'+c['arm'])
  if c['arm']!='ZERO' and sum(actual['actual_aux_L_feature_gradient_ratios'])/4<1e-5:raise ValueError('PA source learning signal insufficient:'+c['arm'])
  evidence[c['arm']]=dict(metrics=done['final_source_metrics'],CE=done['final_training_L_CE'],learning=actual)
 base=evidence['ZERO']['metrics']
 for arm in ('SAME_RX','CROSS_RX'):
  v=evidence[arm]['metrics']
  if v['source_val_accuracy']<base['source_val_accuracy']-.01 or v['source_val_worst_rx']<base['source_val_worst_rx']-.02:raise ValueError('PA source V health falsification failed:'+arm)
 d.write(d.BASE/'source_seed_expansion_decision.json',dict(source_only=True,target_metrics_read=False,passed=True,evidence=evidence,settings_fixed=True));return True
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:shared.worker(a.worker,design=d,package=PACKAGE)
 elif a.predict:
  from .predict import predict
  predict(a.predict);os.environ['CUDA_VISIBLE_DEVICES']='';shared.replace('--score',a.predict,PACKAGE)
 elif a.score:
  from .score import score
  score(a.score)
 else:shared.eligible=eligible;shared.dispatch(design=d,package=PACKAGE)
