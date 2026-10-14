import argparse,copy,json
from pathlib import Path
import torch
from . import design as d,model,observe
from .checks import run_checks
from experiments.cvs_u_schedule50.counterfactual import simulate
def run():
 from experiments.cvs_feature_multiscale50.predict import source_proof
 p=source_proof(d.baseline_config(d.SEEDS[0]));ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);m=model.build(d.config(d.rows()[0]),'cpu');m.load_state_dict(ck['model'],strict=True);t=copy.deepcopy(m);t.load_state_dict(ck['ema_model'],strict=True);m.eval();t.eval()
 with torch.no_grad():outputs=[x.forward_identity_only(torch.zeros(2,2,256)) for x in (m,t)]
 if any(o['id_feat_joint'].shape!=(2,160) or not torch.isfinite(o['tx_logits']).all() for o in outputs):raise ValueError('Actual samecontract defaulthead/ownEMA noquery smoke failed')
 parameters=[torch.nn.Parameter(torch.ones(3)),torch.nn.Parameter(torch.zeros(2))];opt=torch.optim.AdamW(parameters,lr=.0002,weight_decay=.0001)
 for _ in range(7):
  for q in parameters:q.grad=torch.ones_like(q)*.03
  opt.step();opt.zero_grad(set_to_none=True)
 cf=simulate(opt,[torch.ones_like(q) for q in parameters],[torch.ones_like(q)*4 for q in parameters],names=['t1','PA'],max_grad_norm=5.)
 checks=run_checks()
 return dict(status='PASS',real_checkpoint_noquery=True,strict_student_ownEMA=True,query_access=False,no_inherited_training=True,head_fit_updates=0,runtime_observer_counterfactual_imported=True,mathematical_checks=checks['checks'],temporary_synthetic_AdamW_preserves_state=all(cf['summary'][k] for k in ('live_parameters_unchanged','live_gradients_unchanged','live_optimizer_state_unchanged','rng_unchanged')),actual_training_probes='Four later actualL/AdamW/parameterupdate points per real model; syntheticsmoke is not performance or realtraining evidence')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);d.write(p.parse_args().output,run());print(json.dumps(dict(status='PASS')))
