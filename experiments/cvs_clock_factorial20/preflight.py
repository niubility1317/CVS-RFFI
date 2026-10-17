"""One matching fixed-final20 checkpoint, compatibility only and zero query."""
import torch
from . import design as d,model as api
from experiments.cvs_fullsource_concat20 import proof as previous_proof

def smoke():
 p=previous_proof.source(d.reference_config());expected=d.read(d.SOURCE)['role_ids']
 if d.read(p/'source_contract.json')['role_ids']!=expected:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['checkpoint_selection']!='final_only' or cp['epoch']!=20 or cp['a1_ema_successful_updates']!=4440:raise ValueError('Reference is not fixed source final20')
 c=d.config(d.rows()[0]);m=api.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.eval()
 with torch.no_grad():o=m.forward_identity_only(torch.zeros(2,2,256))
 if o['tx_logits'].shape!=(2,6) or o['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(o['tx_logits']).all()):raise ValueError('No-query compatibility smoke failed')
 d.write(d.BASE/'preflight.json',dict(status='PASS',real_CP_loads=1,NN_forward_packets=2,real_IQ=0,query_read=False,head_fit_updates=0,new_training_updates=0,checkpoint_used_for_training_initialization=False,source_roles='EXACT_MATCH',reference_fixed_final20=True))
