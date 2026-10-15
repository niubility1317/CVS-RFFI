"""One real legal source-checkpoint no-query smoke before scratch workers."""
import argparse,json
from pathlib import Path
import torch
from . import design as d,model
def run():
    from experiments.cvs_crossrx_sharpness20 import design as parent
    from experiments.cvs_crossrx_sharpness20.provenance import source_proof,validate_checkpoint
    c=parent.config(next(r for r in parent.rows() if r['arm']=='BASE_CE' and r['model_seed']==parent.SEEDS[0]))
    source=source_proof(c)
    ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(ck,c)
    m=model.build(d.config(d.rows()[0]),'cpu');m.load_state_dict(ck['model'],strict=True);m.eval()
    with torch.no_grad():v=m.forward_identity_only(torch.zeros(2,2,256))
    if v['id_feat_joint'].shape!=(2,160) or v['tx_logits'].shape!=(2,6) or not torch.isfinite(v['tx_logits']).all():raise ValueError('Real CP/defaulthead noquery smoke failed')
    from .checks_matrix import check
    from .checks_provenance import check as provenance
    return dict(status='PASS',real_checkpoint_noquery=True,source_contract_match=True,
        query_access=False,no_worker_weight_inheritance=True,head_fit_updates=0,
        checks=dict(matrix=check(),provenance=provenance()),
        initialization='Existing samecontract BASECEfinal20 used only in this separate smoke process; workers allfreshscratch20 ownEMA')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d.write(a.output,run());print(json.dumps(dict(status='PASS')))
