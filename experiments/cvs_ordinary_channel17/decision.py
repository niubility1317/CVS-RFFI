"""Fixed no-both primary;single bypasses are diagnostics,never fallback candidates."""
import math
from . import design as d

def qualify(raw,full,primary,gains):
 c=d.config()['criteria']
 if len(gains)!=4 or not all(math.isfinite(v) for v in gains):raise ValueError('Four complete finite paired draws required')
 acc=100*(primary['accuracy']-full['accuracy']);f1=100*(primary['macro_F1']-full['macro_F1']);racc=100*(primary['accuracy']-raw['accuracy']);rf1=100*(primary['macro_F1']-raw['macro_F1']);conditions=dict(repair_full_accuracy=acc>=c['min_primary_vs_full_accuracy_gain_pp']-1e-12,repair_full_macroF1=f1>=c['min_primary_vs_full_macroF1_gain_pp']-1e-12,positive_draws=sum(v>0 for v in gains)>=c['min_positive_draws'],every_draw_protected=min(gains)>=c['min_each_draw_gain_pp']-1e-12,raw_accuracy=racc>=c['min_primary_vs_raw_accuracy_gain_pp']-1e-12,raw_macroF1=rf1>=c['min_primary_vs_raw_macroF1_gain_pp']-1e-12,raw_correct_retention=primary['raw_correct_retention']>=c['min_raw_correct_retention'])
 for key in ('RX','TX','day'):
  conditions['every_'+key+'_vs_raw']=all(100*(v['accuracy']-raw['by_'+key][k]['accuracy'])>=c['min_each_group_vs_raw_gain_pp']-1e-12 for k,v in primary['by_'+key].items());conditions['every_'+key+'_vs_full']=all(100*(v['accuracy']-full['by_'+key][k]['accuracy'])>=c['min_each_group_vs_full_gain_pp']-1e-12 for k,v in primary['by_'+key].items())
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,primary_vs_full_accuracy_gain_pp=acc,primary_vs_full_macroF1_gain_pp=f1,primary_vs_raw_accuracy_gain_pp=racc,primary_vs_raw_macroF1_gain_pp=rf1,raw_correct_retention=primary['raw_correct_retention'],draw_accuracy_gains_pp=gains,positive_draws=sum(v>0 for v in gains))
