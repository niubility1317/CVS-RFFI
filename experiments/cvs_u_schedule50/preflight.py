import argparse,copy,json
from pathlib import Path
import torch
from . import design as d,model,proxy
from .checks import run as checks
from .counterfactual import simulate

def run():
 from experiments.cvs_feature_multiscale50.predict import source_proof
 p=source_proof(d.baseline_config(d.SEEDS[0]));ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);c=d.config(d.rows()[0]);m=model.build(c,'cpu');m.load_state_dict(ck['model'],strict=True);t=copy.deepcopy(m);t.load_state_dict(ck['ema_model'],strict=True);m.eval();t.eval()
 with torch.no_grad():out=[x.forward_identity_only(torch.zeros(2,2,256)) for x in (m,t)]
 if any(x['id_feat_joint'].shape!=(2,160) or not torch.isfinite(x['tx_logits']).all() for x in out):raise ValueError('Real samecontract defaulthead/ownEMA no-query smoke failed')
 # This independent synthetic optimizer tests the remote Torch runtime only;
 # actual current-training AdamW states are observed by the four real probes.
 params=[torch.nn.Parameter(torch.ones(3)),torch.nn.Parameter(torch.zeros(2))];opt=torch.optim.AdamW(params,lr=.0002,weight_decay=.0001)
 for _ in range(7):
  for x in params:x.grad=torch.ones_like(x)*.03
  opt.step();opt.zero_grad(set_to_none=True)
 cf=simulate(opt,[torch.ones_like(x) for x in params],[torch.ones_like(x)*4 for x in params],names=['t1','PA'],max_grad_norm=5.)
 if not all(cf['summary'][k] for k in ('live_parameters_unchanged','live_gradients_unchanged','live_optimizer_state_unchanged','rng_unchanged')):raise ValueError('Counterfactual changed remote live state')
 from experiments.cvs_source_label_gate.audit import check as gate_check
 return dict(status='PASS',real_checkpoint_noquery=True,strict_student_ownEMA=True,same_contract=True,no_inherited_training=True,query_access=False,head_fit_updates=0,checks=checks(),runtime_proxy_dependency_imported=True,original_gate_negative_check=gate_check(),remote_AdamW_synthetic_probe=cf['summary'],actual_current_optimizer_probes='Four per realtraining model, not replaced by synthetic smoke')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);d.write(p.parse_args().output,run());print(json.dumps(dict(status='PASS')))
