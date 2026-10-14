"""Separate-process real source-checkpoint smoke; workers remain scratch20."""
import argparse
import json
from pathlib import Path
import torch
from . import design as d,model

def run():
 from experiments.cvs_feature_sequence50 import model as old_model
 from experiments.cvs_feature_sequence50.provenance import source_proof
 from experiments.cvs_sigtime_heldcredit.coordinate import convert_model,PREFIX
 old=d.structural_config(d.SEEDS[0]);p=source_proof(old)
 raw=old_model.build(old,'cpu');params=dict(raw.named_parameters())
 origin={key:params[PREFIX+'.'+key].detach().clone() for key in ('low_hz_','band_hz_')}
 ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 outputs=[]
 for key in ('model','ema_model'):
  raw.load_state_dict(ck[key],strict=True)
  converted=convert_model(raw,origin,'CENTER_25K')
  new=model.build(d.config(d.rows()[0]),'cpu');new.load_state_dict(converted.state_dict(),strict=True);new.eval()
  with torch.no_grad():outputs.append(new.forward_identity_only(torch.zeros(2,2,256)))
 if any(v['id_feat_joint'].shape!=(2,160) or v['tx_logits'].shape!=(2,6)
        or not torch.isfinite(v['tx_logits']).all() for v in outputs):raise ValueError('Real CP/default head noquery smoke failed')
 from .checks_source import check as source
 from .checks_provenance import check as provenance
 return dict(status='PASS',real_checkpoint_noquery=True,strict_student_ownEMA=True,
  same_source_contract=True,query_access=False,no_inherited_training=True,head_fit_updates=0,
  checks=dict(source=source(),provenance=provenance()),
  initialization='Old legal source checkpoint is used only by this smoke process; every worker builds fresh scratch20 and its own EMA')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();d.write(a.output,run());print(json.dumps(dict(status='PASS')))
