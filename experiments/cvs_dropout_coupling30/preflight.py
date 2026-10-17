"""Legal original E20 parent/default head, zero source/query IQ."""
from . import design as d
def smoke():
 import torch
 from . import model as api
 c=d.config(d.rows()[0]);p=d.parent_proof('FULL_AS');cp=d.validate_parent_checkpoint(torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False),c)
 m=api.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.eval()
 with torch.no_grad():out=m.forward_identity_only(torch.zeros(2,2,256))
 if out['tx_logits'].shape!=(2,6) or out['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(out['tx_logits']).all()):raise ValueError('Real source parent/default-head smoke failed')
 d.write(d.BASE/'preflight.json',dict(status='PASS',real_CP_loads=1,NN_forward_packets=2,real_IQ=0,query_read=False,optimizer_updates=0,head_fit_updates=0,parent_epoch=20,new_epochs_each=10,cumulative_epochs_each=30,source_roles='EXACT_MATCH',default_head=True))
