"""Fixed raw identity and light-observation guards;no source accuracy gain claim."""
import math,statistics
from . import design as d

def qualify(raw,noise,draws,stats):
 c=d.config()['criteria'];iq=stats.get('relative_IQ_RMS_change',[])
 if len(draws)!=4 or not iq or not all(math.isfinite(v) and v>=0 for v in iq):raise ValueError('Four complete draws andfinite allpacket IQ statistics required')
 def guard(s):
  conditions=dict(accuracy=100*(s['accuracy']-raw['accuracy'])>=c['min_accuracy_gain_pp']-1e-12,macro_F1=100*(s['macro_F1']-raw['macro_F1'])>=c['min_macroF1_gain_pp']-1e-12,raw_correct_retention=s['raw_correct_retention']>=c['min_raw_correct_retention'])
  for k in ('RX','TX','day'):conditions['every_'+k]=all(100*(v['accuracy']-raw['by_'+k][value]['accuracy'])>=c['min_each_group_accuracy_gain_pp']-1e-12 for value,v in s['by_'+k].items())
  return conditions
 cond=guard(noise);dg=[guard(s) for s in draws];cond.update(each_draw_identity_guard=all(all(z.values()) for z in dg),nonzero_light_noise=statistics.median(iq)>=c['min_IQ_change_median'],bounded_observed_noise=max(iq)<=c['max_IQ_change'])
 return dict(decision='PASS' if all(cond.values()) else 'FAIL',conditions=cond,accuracy_gain_pp=100*(noise['accuracy']-raw['accuracy']),macroF1_gain_pp=100*(noise['macro_F1']-raw['macro_F1']),raw_correct_retention=noise['raw_correct_retention'],draw_accuracy_gains_pp=[100*(s['accuracy']-raw['accuracy']) for s in draws],draw_macroF1_gains_pp=[100*(s['macro_F1']-raw['macro_F1']) for s in draws],draw_guards=dg,IQ_change_median=statistics.median(iq),IQ_change_max=max(iq))
