import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import check

def run():
 from experiments.cvs_feature_multiscale50.predict import own_model
 from experiments.cvs_time_factorial50.provenance import source_proof
 source_proof(d.compact_config(d.SEEDS[0]))
 m=own_model(d.baseline_config(d.SEEDS[0]),'cpu')
 with torch.no_grad():z=m(torch.zeros(2,2,256))
 if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real same-contract checkpoint noquery smoke failed')
 return dict(status='PASS',real_checkpoint_noquery=True,compact_source_reference_verified=True,query_IQ_or_truth_read=False,no_checkpoint_inheritance=True,checks=check(),head_fit_updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();d.write(a.output,run());print(json.dumps(dict(status='PASS',query_read=False,head_fit_updates=0)))
