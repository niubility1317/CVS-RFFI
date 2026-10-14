import argparse,os
from pathlib import Path
from . import design as d
from .provenance import source_proof
from experiments.cvs_feature_sequence50 import dispatch as shared
PACKAGE='experiments.cvs_evidence_relation'
def source_criterion(arms):
 configs=[d.config(r) for r in d.rows() if r['model_seed']==d.SEEDS[0] and r['arm'] in arms]
 if any(not (Path(c['output_root'])/'completion.json').is_file() for c in configs):return False
 evidence={}
 for c in configs:
  p=source_proof(c);done=d.read(p/'completion.json');execution=d.read(p/'relation_execution.json');evidence[c['arm']]=dict(metrics=done['final_source_metrics'],CE=done['final_training_L_CE'],mechanism=execution)
  if not 0<=done['final_training_L_CE']<=.5:raise ValueError('Source CE falsification failed:'+c['arm'])
 base=evidence['ZERO']['metrics']
 for arm in arms:
  if arm=='ZERO':continue
  v=evidence[arm];ratios=v['mechanism']['actual_aux_L_feature_gradient_ratios']
  if len(ratios)!=4 or sum(ratios)/4<1e-5 or v['mechanism']['nonzero_relation_fraction_mean']<.5:raise ValueError('Actual relation learnability falsification failed:'+arm)
  if v['metrics']['source_val_accuracy']<base['source_val_accuracy']-.01 or v['metrics']['source_val_worst_rx']<base['source_val_worst_rx']-.02:raise ValueError('Source default-head health falsification failed:'+arm)
 # No target metrics opened; D first waits for source A/B/C, expansion for A-D.
 name='source_BC_decision.json' if len(arms)==3 else 'source_seed_expansion_decision.json'
 d.write(d.BASE/name,dict(source_only=True,target_metrics_read=False,passed=True,evidence=evidence,method_parameters_frozen=True))
 return True
def eligible(row,design=None,proof=None):
 if row['model_seed']==d.SEEDS[0] and row['arm']!='CROSS_RESP':return True
 return source_criterion(('ZERO','ALL_REL','CROSS_REL') if row['model_seed']==d.SEEDS[0] else d.ARMS)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:shared.worker(a.worker,design=d,package=PACKAGE)
 elif a.predict:
  from .predict import predict
  predict(a.predict);os.environ['CUDA_VISIBLE_DEVICES']='';shared.replace('--score',a.predict,PACKAGE)
 elif a.score:
  from .score import score
  score(a.score)
 else:
  shared.eligible=eligible;shared.dispatch(design=d,package=PACKAGE)
