import json
from . import design as d,model
from experiments.cvs_feature_sequence50 import provenance as shared

def source_proof(c):
 p=shared.source_proof(c,design=d,report_validator=model.validate_report);v=d.read(p/'calendar_execution.json');actual=d.read(p/'actual_native_args.json');actual=actual.get('args',actual)
 if actual.get('source_unlabeled_epochs')!=c['calendar']['unlabeled_epochs'] or v['observed_optimizer_steps']!=11100 or v['L_packet_observations']!=1420800 or v['U_packet_observations']!=963900 or v['completed_U_epochs']!=c['calendar']['unlabeled_epochs'] or v['optimizer_probe_steps']!=c['calendar']['optimizer_probe_global_steps'] or v['extra_V_forward_packets']!=216000 or v['actual_extra_optimizer_steps'] or v['head_fit_updates'] or v['U_truth_read'] or v['target_access']:raise ValueError('Actual equalU/Lcalendar/optimizer observation differs')
 records=[json.loads(x) for x in (p/'source_mechanism_epochs.jsonl').read_text().splitlines()]
 if len(records)!=50 or [r['epoch'] for r in records]!=list(range(1,51)) or any(r['U_count']!=(56700 if r['epoch'] in c['calendar']['unlabeled_epochs'] else 0) or r['U_truth_read'] or r['gate_changed'] or r['head_fit_updates'] for r in records):raise ValueError('Complete calendar/U epoch record missing')
 if len(v['U_sampling'])!=17 or any(r['generated_count']!=56700 or r['delivered_count']!=56700 or not r['physical_packets_once'] or r['U_truth_read'] for r in v['U_sampling']):raise ValueError('All17 complete physical U traversals missing')
 updates=[json.loads(x) for x in (p/'actual_U_optimizer_increment.jsonl').read_text().splitlines()]
 if len(updates)!=4 or [r['step'] for r in updates]!=c['calendar']['optimizer_probe_global_steps'] or any(r['actual_optimizer_steps']!=1 or not r['real_optimizer_state_preserved_during_shadows'] or r['actual_shadow_max_abs_error']>r['actual_shadow_FP32_tolerance'] for r in updates):raise ValueError('Actual AdamW increment observation missing')
 for ordinal in d.PROBE_U_ORDINALS:
  epoch=c['calendar']['unlabeled_epochs'][ordinal-1];proxy=d.read(p/f'current_teacher_V_epoch{epoch}.json')
  if proxy['source_count']!=27000 or proxy['teacher_successful_updates']!=(epoch-1)*222 or proxy['threshold_fit'] or not proxy['model_buffers_parameters_gradients_unchanged'] or proxy['U_truth_read'] or proxy['target_access']:raise ValueError('Current-stage source teacher proxy differs')
 return p
def own_model(c,device):return shared.own_model(c,device,design=d,proof=source_proof,builder=model.build)
