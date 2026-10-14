import json,torch
from . import design as d
from .metrics import observe

def main():
 from . import audit
 from .checks import checks
 torch.set_num_threads(2);v=checks();row=d.rows()[0];c=d.source_config(row);d.model_api(row).source_proof(c);model=d.model_api(row).own_model(c,'cpu').train().requires_grad_(False)
 if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):raise ValueError('Unexpected BatchNorm updates')
 params=[dict(model.named_parameters())[name] for name in d.PARAMETERS]
 for p in params:p.requires_grad_(True)
 x=torch.zeros(4,2,256)
 with torch.no_grad():weak=model.forward_identity_only(x)
 strong=model.forward_identity_only(x);observe(strong['tx_logits'],weak['tx_logits'],strong['id_feat_joint'],weak['id_feat_joint'],weak['tx_logits'].argmax(1),params,'L_s')
 if strong['id_feat_joint'].shape!=(4,160) or (d.ROOT/'preflight.json').exists():raise ValueError('No-query smoke or immutable preflight differs')
 result=dict(status='PASS',checks=v,actual_dispatcher_import=True,real_checkpoint_no_query_smoke=True,source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0)
 d.write(d.ROOT/'preflight.json',result);print(json.dumps(result))
if __name__=='__main__':main()
