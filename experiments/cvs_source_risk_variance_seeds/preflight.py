import json,torch
from . import design as d
from .checks import checks
from experiments.cvs_receiver_dro_seeds import predict as api

def main():
 torch.set_num_threads(2);local=checks();d.require_source()
 model=api.own_model(d.source_config(d.rows()[0]),'cpu').eval().requires_grad_(False)
 before={k:v.clone() for k,v in model.state_dict().items()}
 with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256))
 if out['id_feat_joint'].shape!=(2,160) or out['tx_logits'].shape!=(2,6) or not torch.isfinite(out['tx_logits']).all() or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Actual checkpoint no-query smoke failed')
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Immutable preflight exists')
 value=dict(status='PASS',checks=local,original_source_qualification='PASS',real_checkpoint_no_query_smoke=True,
  synthetic_IQ_packets=2,source_only=True,query_read=False,head_fit_updates=0,optimizer_updates=0)
 d.write(p,value);print(json.dumps(value))
if __name__=='__main__':main()
