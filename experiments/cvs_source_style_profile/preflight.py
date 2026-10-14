import json,torch
from . import design as d
from .checks import checks
from experiments.cvs_neighbor_pair50.predict import source_proof,own_model

def main():
 torch.set_num_threads(2);result=checks();c=d.source_config(d.rows()[0]);source_proof(c);model=own_model(c,'cpu');saved={k:v.clone() for k,v in model.state_dict().items()}
 with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256))
 if out['tx_logits'].shape!=(2,6) or out['id_feat_joint'].shape!=(2,160) or not torch.isfinite(out['tx_logits']).all() or not torch.isfinite(out['id_feat_joint']).all():raise ValueError('Actual default no-query checkpoint smoke failed')
 if any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()):raise ValueError('Smoke changed actual state')
 if (d.ROOT/'preflight.json').exists():raise FileExistsError('Existing style profile preflight')
 value=dict(status='PASS',checks=result,real_checkpoint_no_query_smoke=True,source_IQ_read=False,query_read=False,U_truth_read=False,head_fit=False,optimizer_steps=0);d.write(d.ROOT/'preflight.json',value);print(json.dumps(value))
if __name__=='__main__':main()
