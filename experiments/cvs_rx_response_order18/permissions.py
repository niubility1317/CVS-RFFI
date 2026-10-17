"""Reuse completed source-only handles; no checkpoint or data revalidation."""
from . import design as d
def validate(old,ores,response,rres):
 c=ores['config'];q=rres['config']
 if old['status']!='ARTIFACTS_COMPLETE_SOURCE_ONLY' or (old['L'],old['V'],old['source_RX_holdout'])!=(6300,27000,None) or old['source_RX']!=list(d.RX) or any(old[k] for k in ('CP','NN','updates','U_used','U_truth_read','target_access')):raise ValueError('Complete original source handle required')
 if not ores['source_roles_exact_match'] or c['source_contract']!=d.SOURCE or c['window']!=[80,160] or c['period']!=20 or c['IQ_points']!=256 or c['RX']!=list(d.RX):raise ValueError('Fixed original source observable differs')
 if response['status']!='ARTIFACTS_COMPLETE_SOURCE_ONLY' or response['decision']!='CLOSED_FIXED_SHARED_RESPONSE' or (response['source_L'],response['source_V'],response['source_RX_holdout'])!=(6300,27000,None) or response['source_RX']!=list(d.RX) or any(response[k] for k in ('raw_IQ_reads','CP','NN','teacher_NN','updates','U_used','U_truth_read','target_access','IQ_views_generated','automatic_training')):raise ValueError('Completed closed source response required')
 if not rres['source_handle_exact_match'] or q['derived_input']!=d.DERIVED or (q['reference_RX'],q['phase_anchor_bin'],q['epsilon'])!=(1,1,1e-12) or q['source_RX_holdout'] is not None:raise ValueError('Exact fixed-response handles differ')
