"""Existing legal E20 source parent and actual default head;no source/query IQ."""
from . import design as d

def control():
 p=d.CONTROL_BASE;c=d.config(d.rows()[0]);v=d.read(p/'completion.json');pred=d.read(p/'prediction/complete.json');r=d.read(p/'source_recovery.json');card=r['card']
 if v['status']!='TRAINED_DEFAULT_PREDICTED_SCORED' or v['views']!=list(d.VIEWS) or v['count']!=168000 or not v['truth_last'] or pred['status']!='PREDICTIONS_COMPLETE' or pred['views']!=list(d.VIEWS) or pred['count']!=168000 or pred['truth_read'] or pred['head_fit_updates'] or not pred['fixed_received_IQ']:raise ValueError('Matched nominalAs control incomplete')
 if r['status']!='SOURCE_BUDGET_AND_FIXED_CHECKPOINT_VERIFIED' or card['row_id']!='FULL_AS-s2026092701' or card['epoch']!=10 or card['successful_updates']!=2220 or card['parent']!=c['parent_source'] or not card['source_roles_match'] or not card['default_head_joint'] or card['head_shape']!=[6,160] or card['final_blocked'] or r['source_roles']!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Nominal control parent/budget/data/head differs')
 return dict(status='VERIFIED',run=d.CONTROL_RUN,row='FULL_AS-s2026092701',reused_existing=True,new_training_updates=0,source_RX_holdout=None)

def smoke():
 import torch
 from . import model as api
 c=d.config(d.rows()[0]);p=d.parent_proof('FULL_AS');cp=d.validate_parent_checkpoint(torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False),c);reference=control()
 if d.read(d.PROJECT+'/runs/'+d.PHYSICAL_RUN+'/result.json')['decision']!='PASS_PHYSICAL_MISMATCH_ONLY':raise ValueError('No fixed source physical evidence')
 m=api.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.eval()
 with torch.no_grad():out=m.forward_identity_only(torch.zeros(2,2,256))
 if out['tx_logits'].shape!=(2,6) or out['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(out['tx_logits']).all()):raise ValueError('Actual source parent/default-head smoke failed')
 d.write(d.BASE/'preflight.json',dict(status='PASS',real_CP_loads=1,NN_forward_packets=2,real_IQ=0,query_read=False,optimizer_updates=0,head_fit_updates=0,parent_epoch=20,new_epochs=10,cumulative_epochs=30,matched_control=reference))
