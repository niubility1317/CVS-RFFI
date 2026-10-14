from . import design as d,model
from experiments.cvs_feature_sequence50 import provenance as shared
def source_proof(c):
 p=shared.source_proof(c,design=d,report_validator=model.validate_report);v=d.read(p/'relation_execution.json')
 if v['auxiliary_calls']!=3774 or v['U_packet_observations']!=963900 or v['anchor_refresh_calls']!=17 or v['extra_L_forward_packets']!=3060 or not v['all_U_used'] or v['U_truth_read'] or v['target_access'] or v['head_fit_updates']:raise ValueError('Actual allU/reference budget differs')
 return p
def own_model(c,device):return shared.own_model(c,device,design=d,proof=source_proof,builder=model.build)
