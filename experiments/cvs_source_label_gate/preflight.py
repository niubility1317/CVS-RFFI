import argparse,json
from pathlib import Path
import torch
from . import design as d
from .audit import check
from experiments.cvs_feature_multiscale50.predict import own_model
def run():
 model=own_model(d.baseline_config(d.SEEDS[0]),'cpu')
 with torch.no_grad():v=model(torch.zeros(2,2,256))
 if v.shape!=(2,6) or not torch.isfinite(v).all():raise ValueError('Real checkpoint noquery smoke failed')
 return dict(status='PASS',real_checkpoint_noquery=True,query_IQ_or_truth_read=False,no_new_checkpoint_inheritance=True,checks=check())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();d.write(a.output,run());print(json.dumps(dict(status='PASS',query_read=False)))
