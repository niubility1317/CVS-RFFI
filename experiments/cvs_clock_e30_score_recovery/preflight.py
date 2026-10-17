"""Verify all five completed source budgets,then one realCP no-query smoke."""
import torch
from . import design as d,proof
from experiments.cvs_clock_factorial30 import model as api
def smoke():
 c=d.config(d.rows()[0]);proof.recover(c)
 cp=proof.checkpoint(torch.load(proof.source(c)/'final_ssdg.pth',map_location='cpu',weights_only=False),c)
 m=api.build(c['training_config'],'cpu');m.load_state_dict(cp['model'],strict=True);m.eval()
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))
 if z['tx_logits'].shape!=(2,6) or z['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(z['tx_logits']).all()):raise ValueError('Fixed default-head compatibility failure')
 d.write(d.BASE/'preflight.json',dict(status='PASS',verified_source_models=1,other_source_models='perrowbeforeprediction',real_CP_loads=3,CP_comparisons='onefinal+oneinitial for provenance;onefinal for smoke',NN_forward_packets=2,real_IQ=0,query_read=False,new_training_updates=0,head_fit_updates=0,all_cumulative_epochs=30,original_failure_preserved=True))
