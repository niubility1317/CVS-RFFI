import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as check
from .provenance import models,source
def run():
 m,t,p=models(d.SEEDS[0])
 with torch.no_grad():a=m.forward_identity_only(torch.zeros(2,2,256));b=t.forward_identity_only(torch.zeros(2,2,256))
 if a['id_feat_joint'].shape!=(2,160) or not all(torch.isfinite(v['tx_logits']).all() for v in (a,b)):raise ValueError('Same-contract real student/EMA no-query smoke failed')
 for seed in d.SEEDS:source(seed)
 return dict(status='PASS',real_checkpoint_noquery=True,source_contract_match=True,source_parents=3,query_access=False,checks=check())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);d.write(p.parse_args().output,run());print(json.dumps(dict(status='PASS')))
