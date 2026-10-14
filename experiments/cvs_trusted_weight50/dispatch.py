"""Reuse the shared resource queue and automatic default-head truth-last pipeline."""
import argparse
from . import design as d
from .source import train
from .predict import source_proof,predict
from .score import score,finish
from experiments.cvs_signature_stats50.dispatch import worker,dispatch,replace
MODULE='experiments.cvs_trusted_weight50.dispatch'
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:worker(a.worker,design=d,train_fn=train,proof_fn=source_proof,module=MODULE)
 elif a.predict:predict(a.predict);replace('--score',a.predict,MODULE)
 elif a.score:score(a.score)
 else:dispatch(design=d,finish_fn=finish,module=MODULE)
if __name__=='__main__':main()
