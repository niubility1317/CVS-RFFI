import argparse,json
from pathlib import Path
import torch
from . import design as d
from .audit import models
from .signals import checks

def run():
 m,t=models(d.config(d.rows()[0]));m.eval()
 with torch.no_grad():outs=[n.forward_identity_only(torch.zeros(2,2,256)) for n in (m,t)]
 if any(o['id_feat_joint'].shape!=(2,160) or not torch.isfinite(o['tx_logits']).all() for o in outs):raise ValueError('Real frozen checkpoint no-query smoke failed')
 from experiments.cvs_evidence_relation.provenance import source_proof
 for row in d.rows():source_proof(d.config(row)['parent_config'])
 return dict(status='PASS',real_checkpoint_noquery=True,strict_student_ownEMA=True,checkpoint_contract_match=True,source_parents=4,query_access=False,head_fit_updates=0,checks=checks())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);d.write(p.parse_args().output,run());print(json.dumps(dict(status='PASS')))
