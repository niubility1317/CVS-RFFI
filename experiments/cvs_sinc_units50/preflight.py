import json,torch
from . import design as d
from .checks import checks as component_checks
from .checks_pipeline import checks as pipeline_checks
from experiments.cvs_source_self_challenge.audit import source_model

def main():
 torch.set_num_threads(2);checks=dict(component=component_checks(),pipeline=pipeline_checks())
 model=source_model().eval().requires_grad_(False);before={k:v.clone() for k,v in model.state_dict().items()}
 with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256))
 if out['id_feat_joint'].shape!=(2,160) or out['tx_logits'].shape!=(2,6) or not torch.isfinite(out['tx_logits']).all() or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Real checkpoint no-query smoke failed')
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Immutable preflight exists')
 value=dict(status='PASS',checks=checks,real_checkpoint_no_query_smoke=True,synthetic_IQ_packets=2,source_checkpoint='Predetermined original DRO SIG_TIME firstseed final50, source_proof verified beforeload',training_inherits_checkpoint=False,query_read=False,head_fit_updates=0,real_checkpoint_optimizer_updates=0)
 d.write(p,value);print(json.dumps(value))
if __name__=='__main__':main()
