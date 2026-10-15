import argparse
from pathlib import Path
from types import SimpleNamespace
from . import design as d
from .source import train
from .predict import source_proof,predict
from .score import score,finish
from experiments.cvs_signature_stats50.dispatch import worker,dispatch,replace
from experiments.cvs_temporal_rsc50.dispatch import restore_project
MODULE='experiments.cvs_sinc_units50.dispatch'
def registered_design():
 def read(path):
  v=d.read(path)
  return restore_project(v,d.PROJECT) if Path(path)==d.ROOT/'registration/experiment.json' else v
 return SimpleNamespace(**dict(vars(d),read=read))
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:worker(a.worker,design=d,train_fn=train,proof_fn=source_proof,module=MODULE)
 elif a.predict:predict(a.predict);replace('--score',a.predict,MODULE)
 elif a.score:score(a.score)
 else:dispatch(design=registered_design(),finish_fn=finish,module=MODULE)
if __name__=='__main__':main()
