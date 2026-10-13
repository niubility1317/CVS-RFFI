import json
import torch
from . import design as d

def main():
 from .integration_checks import checks
 from .checks import checks as math_checks
 from .episode import Objective
 from experiments.cvs_frequency_context50.predict import source_proof,own_model
 from experiments.cvs_source_transfer_probe.run import data
 from experiments.cvs_transfer_features.model import from_phase1
 torch.set_num_threads(2);math_result=math_checks();integration=checks()
 row=d.rows()[0];c=d.baseline_config(row);source_proof(c);model=own_model(c,'cpu')
 with torch.no_grad():
  z=model.forward_identity_only(torch.zeros(2,2,256));f=from_phase1(model);pure=f(torch.zeros(2,2,256))
 if z['tx_logits'].shape!=(2,6) or not torch.isfinite(z['tx_logits']).all() or not torch.equal(pure,z['id_feat_joint']):raise ValueError('Real baseline no-query smoke differs')
 L,V,contract=data();del V
 model.train().requires_grad_(True);before={k:v.clone() for k,v in model.state_dict().items()};proof=[]
 for r in (d.training_rows()[0],d.training_rows()[1]):
  cc=d.config(r);obj=Objective(cc,L,contract)
  for step in (1,2,3):obj(model,step)
  rng=torch.get_rng_state().clone();loss,metric=obj(model,4)
  W=model.id_backbone.encoder.core.id_backbone.cls_head.weight
  grad_W=torch.autograd.grad(loss,W,allow_unused=True,retain_graph=True)[0]
  model.zero_grad(set_to_none=True);loss.backward()
  gradient=sum(float(p.grad.square().sum()) for p in model.id_backbone.parameters() if p.grad is not None)
  if grad_W is not None or not gradient>0 or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Actual source episode gradient/RNG differs')
  proof.append(dict(arm=r['arm'],loss=float(loss),backbone_gradient_l2=gradient**.5,class_W_gradient=None,L_s_only=True))
 if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('No-update smoke changed checkpoint')
 for r in d.rows():d.validate(d.config(r))
 result=dict(status='PASS',real_checkpoint_no_query_smoke=True,old_weights_used='smoke and frozen baseline comparison only; new students scratch',training_inherits_old_weights=False,query_read=False,actual_source_episode_gradient=proof,math=math_result,integration=integration)
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Existing preflight artifact')
 d.write(p,result);print(json.dumps(result))
if __name__=='__main__':main()
