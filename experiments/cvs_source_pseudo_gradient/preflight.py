import json,torch
from . import design as d
from .metrics import observe_gradients
def main():
 from .checks import checks
 torch.set_num_threads(2);result=checks()
 for row in d.rows():d.model_api(row).source_proof(d.source_config(row))
 row=d.rows()[0];model=d.model_api(row).own_model(d.source_config(row),'cpu').requires_grad_(False)
 params=[dict(model.named_parameters())[name] for name in d.PARAMETERS]
 for p in params:p.requires_grad_(True)
 value=model.forward_identity_only(torch.zeros(2,2,256))
 pseudo=value['tx_logits'].detach().argmax(1);observe_gradients(value['tx_logits'],pseudo,pseudo,torch.ones(2,dtype=torch.bool),params,'L_s')
 if value['id_feat_joint'].shape!=(2,160) or not torch.isfinite(value['tx_logits']).all():raise ValueError('Real checkpoint no-query smoke failed')
 if (d.ROOT/'preflight.json').exists():raise FileExistsError('Existing preflight')
 result=dict(status='PASS',checks=result,real_checkpoint_no_query_smoke=True,source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0)
 d.write(d.ROOT/'preflight.json',result);print(json.dumps(result))
if __name__=='__main__':main()
