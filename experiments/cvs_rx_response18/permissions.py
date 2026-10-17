"""Reuse validated source-derived handles; no raw data, U or checkpoint access."""
from . import design as d
def validate(done,resolved):
 c=resolved['config']
 if done['status']!='ARTIFACTS_COMPLETE_SOURCE_ONLY' or done['L']!=6300 or done['V']!=27000 or done['source_RX']!=list(d.RX) or done['source_RX_holdout'] is not None or done['unique_IQ_observations']!=33300 or any(done[k] for k in ('CP','NN','updates','U_used','U_truth_read','target_access')):raise ValueError('Existing exact all-source derived handle required')
 if not resolved['source_roles_exact_match'] or c['source_contract']!=d.SOURCE or c['RX']!=list(d.RX) or c['L']!=6300 or c['V']!=27000 or c['window']!=[80,160] or c['period']!=20 or c['IQ_points']!=256 or c['target_access']:raise ValueError('Original source-derived roles/window differ')
