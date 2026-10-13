import argparse
from pathlib import Path
from . import design as d
from .source import train
from .predict import source_proof,predict
from .score import score,finish
from experiments.cvs_pa_cross_delay.dispatch import worker as shared_worker,dispatch as shared_dispatch,replace
MODULE='experiments.cvs_time_factorial.dispatch'
def eligible(row):
 if row['arm']==d.BASELINE_ARM or row['model_seed']==d.SEEDS[0]:return True
 first=[d.config(r) for r in d.training_rows() if r['model_seed']==d.SEEDS[0]]
 if any(not (Path(c['output_root'])/'completion.json').is_file() or not (Path(c['output_root'])/'source_frozen.json').is_file() for c in first):return False
 for c in first:
  source_proof(c)
  value=d.read(Path(c['output_root'])/'completion.json')['final_training_L_CE']
  if not 0<=value<=.5:raise ValueError('Preregistered source L CE first-seed falsification failed: '+c['arm'])
 return True
def dispatch():return shared_dispatch(design=d,eligible_fn=eligible,finish_fn=finish,module=MODULE)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:shared_worker(a.worker,design=d,train_fn=train,proof_fn=source_proof,module=MODULE)
 elif a.predict:predict(a.predict);replace('--score',a.predict,MODULE)
 elif a.score:score(a.score)
 else:dispatch()
