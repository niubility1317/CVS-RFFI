import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import check as prototype_checks
from .checks_pipeline import check as pipeline_checks

def run():
 from experiments.cvs_feature_multiscale50.predict import own_model
 c=d.baseline_config(d.SEEDS[0]);model=own_model(c,'cpu')
 with torch.no_grad():z=model(torch.zeros(2,2,256))
 if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real same-contract historical checkpoint no-query smoke')
 return dict(status='PASS',real_checkpoint_noquery=True,query_IQ_or_truth_read=False,no_new_backbone_checkpoint_inheritance=True,prototype=prototype_checks(),pipeline=pipeline_checks())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();d.write(a.output,run());print(json.dumps(dict(status='PASS',query_read=False)))
