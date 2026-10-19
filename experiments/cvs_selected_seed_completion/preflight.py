"""One real-source-checkpoint compatibility smoke, no source IQ or target queries."""
from pathlib import Path
from . import design as d
def run(r):
 import torch
 from .binding import modules
 from experiments.cvs_equivariant_identity.precision import numerical_context
 b,api=modules(r);c=b['config'](30)
 if r['family']=='pa':parent=Path(c['parent_source'])
 elif r['family']=='clock':parent=d.PROJECT/'runs/20261017-phase1-F-crossRX-clock-factorial-e20-s2026092701-m3-r01/B01-s2026092701/source'
 else:parent=d.PROJECT/'runs'/r['seed01_run']/r['seed01_row']/'source'
 done=d.read(parent/'completion.json');init=d.read(parent/'initialization.json');contract=d.read(parent/'source_contract.json')
 if done['target_access'] or init['target_contact'] or contract['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Real smoke checkpoint source provenance differs')
 # The original matching parents are source-only fixed-final trajectories.
 if not init['scratch_only']:
  for ancestor in init['ancestors']:
   a=Path(ancestor);v=d.read(a/'completion.json');i=d.read(a/'initialization.json')
   if v['target_access'] or i['target_contact'] or not i['scratch_only'] or i['ancestors'] or d.read(a/'source_contract.json')['role_ids']!=contract['role_ids']:raise ValueError('Unverified smoke ancestor')
 cp=torch.load(parent/'final_ssdg.pth',map_location='cpu',weights_only=False)
 torch.set_num_threads(2)
 with numerical_context(d.FULL_FP32_POLICY):
  model=api.build(b['config'](20) if r['family']=='u' else c,'cpu');model.load_state_dict(cp['model'],strict=True);model.eval();before={k:v.clone() for k,v in model.state_dict().items()}
  with torch.no_grad():z=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
  if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Real default-head no-query smoke failed')
  if model._selected_factory_seed!=r['model_seed']:raise ValueError('Factory model seed differs')
 result=dict(status='PASS',row_id=r['row_id'],family=r['family'],actual_factory_seed=r['model_seed'],real_source_checkpoint=True,source_roles='EXACT_MATCH',NN=2,source_IQ=0,target_query=False,optimizer_updates=0,default_head_only=True,state_unchanged=True)
 d.write(d.BASE/'smoke'/(r['row_id']+'.json'),result);print(result,flush=True);return result
