"""Exact per-scene 0.8 percentage-point allowance against the original R38."""
from tools.r17_compact_acceptance import target_acceptance as validate_target
def target_acceptance(candidate,original,*,reference='user_R38_794529'):
 value=validate_target(candidate,original)
 n=int(candidate['count'])
 for row in value['views'].values():row['accepted']=500*(row['original_correct']-row['candidate_correct'])<=4*n
 value.update(reference=reference,allowance_pp=.8,maximum_correct_loss=4*n//500,accepted=all(r['accepted'] for r in value['views'].values()))
 return value
