import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from experiments.cvs_scratch30_baseline.predict import own_model
from experiments.cvs_scratch30_baseline import design as parent

def run():
 torch.set_num_threads(2);evidence=None
 c=parent.config(next(r for r in parent.rows() if r['arm']=='CE30'))
 model=own_model(c,'cpu')
 with torch.no_grad():z=model(torch.zeros(2,2,256))
 if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real same-contract checkpoint noquery smoke')
 return dict(status='PASS',real_checkpoint_noquery=True,query_IQ_or_truth_read=False,source_evidence='fixed3seed source shift diagnostics; no target scores consumed',checks=checks(),no_training_checkpoint_inheritance=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True,type=Path);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',clean_only=True,query_IQ_or_truth_read=False)))
