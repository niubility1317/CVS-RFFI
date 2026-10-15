import argparse,os
from . import design as d
from .source import train
from .predict import source_proof,predict
from .score import score,finish
from experiments.cvs_signature_stats50.dispatch import worker,dispatch,replace
MODULE='experiments.cvs_rx_cagrad_seeds.dispatch'
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict');p.add_argument('--score');a=p.parse_args()
 if a.worker:worker(a.worker,design=d,train_fn=train,proof_fn=source_proof,module=MODULE)
 elif a.predict:predict(a.predict);os.environ['CUDA_VISIBLE_DEVICES']='';replace('--score',a.predict,MODULE)
 elif a.score:os.environ['CUDA_VISIBLE_DEVICES']='';score(a.score)
 else:d.require_source();dispatch(design=d,finish_fn=finish,module=MODULE)
if __name__=='__main__':main()
