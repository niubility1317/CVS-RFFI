"""One exact source-contract parent,2 synthetic packets,zero query."""
import torch
from . import design as d,model as api
def smoke():
 c=d.config(d.rows()[0]);p=d.parent_proof('B00');cp=d.validate_parent_checkpoint(torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False),c)
 m=api.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.eval()
 with torch.no_grad():o=m.forward_identity_only(torch.zeros(2,2,256))
 if o['tx_logits'].shape!=(2,6) or o['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(o['tx_logits']).all()):raise ValueError('No-query E30 compatibility failed')
 d.write(d.BASE/'preflight.json',dict(status='PASS',real_CP_loads=1,NN_forward_packets=2,real_IQ=0,query_read=False,head_fit_updates=0,new_training_updates=0,source_roles='EXACT_MATCH',fixed_final20_parent=True))
