"""Fixed source signal test, never target efficacy or a novelty verdict."""
from .design import GATE,ORDINALS

def source_gate(records):
 if [r['ordinal'] for r in records]!=list(ORDINALS):raise ValueError('All four fixed source cohorts required')
 active=[];distinct=[];physical=[]
 for r in records:
  cross=r['modes']['CROSS_GAP']
  if (max(v['gap'] for v in cross['by_RX'].values())>GATE['cross_positive_gap_min']
      and cross['auxiliary_gradient_norm']>GATE['auxiliary_gradient_min']):active.append(r['ordinal'])
  if r['CROSS_GLOBAL_gradient_relative_difference']>=GATE['cross_global_gradient_relative_difference_min']:
   distinct.append(r['ordinal'])
  if r['physical_Sinc']['auxiliary_changed_kernels']>0:physical.append(r['ordinal'])
 return dict(passed=len(active)>=2 and len(distinct)>=2 and len(physical)>=2,
  active_cross_cohorts=active,distinct_cross_global_cohorts=distinct,
  auxiliary_Sinc_kernel_change_cohorts=physical)
