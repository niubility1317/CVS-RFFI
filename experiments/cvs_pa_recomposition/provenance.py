import json
from . import design as d,model
from experiments.cvs_feature_sequence50 import provenance as shared
def source_proof(c):
 p=shared.source_proof(c,design=d,report_validator=model.validate_report);v=d.read(p/'PA_execution.json')
 if v['auxiliary_calls']!=2498 or v['L_packet_observations']!=1420800 or v['active_L_packet_observations']!=2498*128 or v['optimizer_probe_steps']!=list(d.PROBE_STEPS) or len(v['actual_aux_L_feature_gradient_ratios'])!=4 or v['extra_IQ_forward_packets'] or v['extra_actual_optimizer_steps'] or v['head_fit_updates'] or v['U_truth_read'] or v['target_access'] or v['teacher_modified'] or not v['old_L_CE_unchanged'] or not v['default_head_joint_training']:raise ValueError('Actual PA budget/role/head differs')
 if len(v['U_sampling'])!=17 or any(r['generated_count']!=56700 or r['delivered_count']!=56700 or r['U_truth_read'] for r in v['U_sampling']):raise ValueError('Complete original17U traversal missing')
 rows=[json.loads(z) for z in (p/'source_mechanism_epochs.jsonl').read_text().splitlines()]
 if len(rows)!=50 or [r['epoch'] for r in rows]!=list(range(1,51)) or any(r['U_count']!=(56700 if r['epoch']>=34 else 0) or r['U_truth_read'] or r['head_fit_updates'] for r in rows):raise ValueError('Complete source PA/U records missing')
 updates=[json.loads(z) for z in (p/'actual_PA_optimizer_increment.jsonl').read_text().splitlines()]
 if [r['step'] for r in updates]!=list(d.PROBE_STEPS) or any(r['actual_optimizer_steps']!=1 or r['temporary_steps']!=2 or r['actual_shadow_max_abs_error']>r['actual_shadow_tolerance'] for r in updates):raise ValueError('Actual PA optimizer measurements incomplete')
 return p
def own_model(c,device):return shared.own_model(c,device,design=d,proof=source_proof,builder=model.build)
