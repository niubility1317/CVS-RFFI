"""Real own-checkpoint no-query smoke and actual callback focused checks."""
import argparse
import copy
import json
from pathlib import Path
import torch
from . import design as d,model

def run():
 from experiments.cvs_feature_sequence50.provenance import source_proof
 p=source_proof(d.structural_config(d.SEEDS[0]))
 ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 m=model.build(d.config(d.rows()[0]),'cpu');m.load_state_dict(ck['model'],strict=True)
 t=copy.deepcopy(m);t.load_state_dict(ck['ema_model'],strict=True);m.eval();t.eval()
 with torch.no_grad():outputs=[v.forward_identity_only(torch.zeros(2,2,256)) for v in (m,t)]
 if any(v['id_feat_joint'].shape!=(2,160) or v['tx_logits'].shape!=(2,6)
        or not torch.isfinite(v['tx_logits']).all() for v in outputs):raise ValueError('Real CP/default head noquery smoke failed')
 from .checks_native_extension import check as native
 from .checks_objective import run_checks as loss
 from .checks_credit import check as credit
 from .checks_capture import check as capture
 from .checks_observe import check as observe
 checks=dict(native=native(),loss=loss(),credit=credit(),capture=capture(),observe=observe())
 return dict(status='PASS',real_checkpoint_noquery=True,strict_student_ownEMA=True,
             same_source_contract=True,query_access=False,no_inherited_training=True,
             head_fit_updates=0,checks=checks,
             initialization='Actual training workers independently build scratch models and ownEMA; smoke process weights never enter workers')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();d.write(a.output,run());print(json.dumps(dict(status='PASS')))
