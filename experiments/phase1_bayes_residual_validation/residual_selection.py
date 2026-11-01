"""Predeclared source selection versus DURA or frozen parent."""
import math
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
REQUIRED={f'{part}:{rx}:{v}' for part,receivers in ((1,(4,6)),(2,(8,))) for rx in receivers for v in VIEWS}

def choose(results,require_iq,reference):
 if reference not in ('baseline','parent'):raise ValueError('Fixed source reference required')
 best=dict(kind='none',validation_net=0)
 for threshold,cells in results.items():
  if set(cells)!=REQUIRED:raise ValueError('All source receiver/scene cells required')
  val=[(k,v) for k,v in cells.items() if k.startswith('1:')]
  if any(v['count']<=0 or v['valid_count']<=0 for _,v in val):raise ValueError('Valid source cells required')
  correct=reference+'_correct';iq=reference+'_iq';net=sum(v['candidate_correct']-v[correct] for k,v in val if not k.endswith(':clean'))
  cls=all(v['candidate_correct']>=v[correct]-max(1,math.ceil(.005*v['valid_count'])) for _,v in val)
  clean=all(v['candidate_iq']<=v[iq]+.001 for k,v in val if k.endswith(':clean'))
  impaired=all(v['candidate_iq']<=1.2*v[iq]+1e-6 for k,v in val if not k.endswith(':clean'))
  if cls and clean and (impaired or not require_iq) and net>=10 and net>best['validation_net']:best=dict(kind='residual',threshold=float(threshold),validation_net=net,reference=reference)
 return best
