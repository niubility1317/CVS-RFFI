import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import check_math,check_model,check_pipeline,check_observer
def run():
 from experiments.cvs_feature_multiscale50.predict import own_model
 m=own_model(d.baseline_config(d.SEEDS[0]),'cpu')
 with torch.no_grad():out=m(torch.zeros(2,2,256));z=m.id_backbone.encoder.features(torch.zeros(2,2,256))
 if out.shape!=(2,6) or z.shape!=(2,160) or not torch.isfinite(out).all():raise ValueError('Real same-contract noquery checkpoint smoke failed')
 return dict(status='PASS',real_checkpoint_noquery=True,query_read=False,no_checkpoint_inheritance=True,head_fit_updates=0,math=check_math(),model=check_model(),pipeline=check_pipeline(),observer=check_observer())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();d.write(a.output,run());print(json.dumps(dict(status='PASS',query_read=False,head_fit_updates=0)))
