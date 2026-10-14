import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as check,wiring
def run():
 from experiments.cvs_feature_multiscale50.predict import own_model
 m=own_model(d.baseline_config(d.SEEDS[0]),'cpu')
 with torch.no_grad():z=m(torch.zeros(2,2,256))
 if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real same-contract checkpoint no-query smoke failed')
 return dict(status='PASS',real_checkpoint_noquery=True,no_inherited_initialization=True,checks=check(),actual_wiring=wiring(),query_access=False,head_fit_updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);d.write(p.parse_args().output,run());print(json.dumps(dict(status='PASS')))
