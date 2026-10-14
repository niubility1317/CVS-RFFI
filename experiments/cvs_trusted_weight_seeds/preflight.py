import json,torch
from . import design as d
from experiments.cvs_feature_multiscale50.predict import source_proof,own_model

def main():
 from .qualification import check
 qualification=check();d.write(d.ROOT/'source_expansion_evidence.json',qualification)
 torch.set_num_threads(2);old=d.baseline_config(d.rows()[0]);p=source_proof(old);model=own_model(old,'cpu');saved={k:v.clone() for k,v in model.state_dict().items()}
 with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256))
 if out['tx_logits'].shape!=(2,6) or out['id_feat_joint'].shape!=(2,160) or any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()) or not torch.isfinite(out['tx_logits']).all():raise ValueError('Real checkpoint no-query smoke differs')
 value=dict(status='PASS',real_checkpoint_no_query_smoke=True,source_checkpoint=str(p/'final_ssdg.pth'),source_seed=d.SEEDS[0],query_read=False,head_fit=False,training_inherits_checkpoint=False)
 d.write(d.ROOT/'preflight.json',value);print(json.dumps(value))
if __name__=='__main__':main()
