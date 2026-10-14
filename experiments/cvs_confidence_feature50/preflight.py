import copy,json,torch
from . import design as d
from .objective import consistency
from experiments.cvs_source_signature_order.probe import model
def main():
 torch.set_num_threads(2);m=model().train().requires_grad_(True);t=copy.deepcopy(m).eval().requires_grad_(False);saved={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():weak=t(torch.zeros(2,2,256),y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=torch.zeros(2,dtype=torch.long))
 strong=m(torch.full((2,2,256),.015),y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=torch.zeros(2,dtype=torch.long));loss=consistency(strong['id_feat_joint'],weak['id_feat_joint'],weak['tx_logits'],strong['tx_logits']);W=m.id_backbone.encoder.core.id_backbone.cls_head.weight;params=[dict(m.named_parameters())[n] for n in d.PARAMETERS];gs=torch.autograd.grad(loss,params+[W],allow_unused=True)
 if not torch.isfinite(loss) or gs[-1] is not None or any(g is not None and not torch.isfinite(g).all() for g in gs[:-1]) or any(not torch.equal(v,m.state_dict()[k]) for k,v in saved.items()):raise ValueError('Real feature consistency no-query smoke failed')
 v=dict(status='PASS',real_checkpoint_no_query_smoke=True,zero_or_constant_synthetic_IQ_packets=4,actual_joint160_features=True,feature_gradient_W_independent=True,source_checkpoint='predetermined first SIG_TIME final50; source_proof checked by model helper',teacher_for_smoke='frozen copy of source student; training teacher is own scratch EMA',training_inherits_checkpoint=False,query_read=False,head_fit=False,optimizer_updates=0)
 d.write(d.ROOT/'preflight.json',v);print(json.dumps(v))
if __name__=='__main__':main()
