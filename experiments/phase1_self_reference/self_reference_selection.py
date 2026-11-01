"""Predeclared DURA-relative, receiver-by-scene source selection."""
import math
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
REQUIRED={f'{part}:{rx}:{view}' for part,receivers in ((1,(4,6)),(2,(8,))) for rx in receivers for view in VIEWS}
def choose(results):
 best=dict(kind='dura',validation_net=0)
 for epoch,combos in results.items():
  for key,cells in combos.items():
   if set(cells)!=REQUIRED:raise ValueError('All source receiver/scene cells required')
   alpha,threshold=map(float,key.split(':'));val=[(k,v) for k,v in cells.items() if k.startswith('1:')]
   if any(v['count']<=0 or v['valid_count']<=0 for _,v in val):raise ValueError('Valid source cases required')
   net=sum(v['candidate_correct']-v['baseline_correct'] for k,v in val if not k.endswith(':clean'))
   cls=all(v['candidate_correct']>=v['baseline_correct']-max(1,math.ceil(.005*v['valid_count'])) for _,v in val)
   iq=all(v['candidate_iq']<=(v['baseline_iq']+.001 if k.endswith(':clean') else 1.2*v['baseline_iq']+1e-6) for k,v in val)
   if cls and iq and net>=10 and net>best['validation_net']:best=dict(kind='self_reference_net',epoch=int(epoch),alpha=alpha,threshold=threshold,validation_net=net)
 return best
