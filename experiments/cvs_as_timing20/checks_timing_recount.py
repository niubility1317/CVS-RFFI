"""Complete synthetic4440-step calendar counter fixtures;0realdata/IQ/NN."""
import copy,json,tempfile
from pathlib import Path
from collections import Counter
from .analyze import timing_recount,ARMS,VIEWS

def run():
 ids=[[1,rx,day,s,t*15000+(1,3,4,6,8).index(rx)*3000+(day-1)*1000+s] for t in range(6) for rx in (3,4,6,8) for day in (1,2,3) for s in range(70)];seq=[dict(role='L',ids=[ids[(i*128+j)%5040] for j in range(128)]) for i in range(4440)];plans={'RAW':[],'ALL':list(range(1,21)),'EARLY7':list(range(1,8)),'LATE7':list(range(14,21))};negative=0
 with tempfile.TemporaryDirectory() as root:
  base=Path(root)
  for arm in ARMS:
   steps=[];g=Counter();u=Counter();epochs=plans[arm]
   for i,batch in enumerate(seq,1):
    epoch=(i-1)//222+1;on=epoch in epochs
    for pid in batch['ids']:
     tx=pid[4]//15000;k=(tx,pid[1],pid[2]);g[k]+=1
     if on:u[k]+=1
    steps.append(dict(As_active=on,extra_As_packets=128*on,As_shadow_generated=128,shadow_raw_MSE=.2,extra_BN_batch_updates=0,extra_input='As(raw)' if on else 'raw',extra_raw_MSE=.2 if on else 0.,shadow_scene_counts={VIEWS[1]:128},shadow_state_counts={'LOS':128},shadow_lock_counts={'locked':128}))
   def cells(c):return [dict(TX=k[0],RX=k[1],day=k[2],scene=VIEWS[1],state='LOS',lock='locked',count=v) for k,v in c.items()]
   audit=dict(generated_coverage=cells(g),steps=4440,generated=568320,consumed=sum(u.values()),timing=dict(active_epochs=epochs),new_parameters=0,extra_U=False,head_fit_updates=0,target_access=False);observed=dict(cells=cells(u),total=sum(u.values()),all_legal_L_supervised=True,confidence_gate=False)
   def write():
    (base/'timing_audit.json').write_text(json.dumps(audit),encoding='utf8');(base/'satellite_supervision_coverage.json').write_text(json.dumps(observed),encoding='utf8')
   write();result=timing_recount(base,arm,seq,steps,[dict(epoch=i) for i in range(1,21)]);assert result['generated_unique_L']==5040 and result['calls']==4440 and result['consumed']==len(epochs)*222*128 and result['consumed_unique_L']==(0 if arm=='RAW' else 5040)
   bad=copy.deepcopy(steps);bad[0]['As_active']=not bad[0]['As_active']
   try:timing_recount(base,arm,seq,bad,[])
   except ValueError:negative+=1
   else:raise AssertionError('Wrong calendar accepted')
   audit['generated_coverage'][0]['count']+=1;write()
   try:timing_recount(base,arm,seq,steps,[])
   except ValueError:negative+=1
   else:raise AssertionError('Wrong generated coverage accepted')
 return dict(status='PASS',complete_synthetic_calendar_steps=17760,protocol_counterexamples=negative,synthetic_L_metadata_exposures=2273280,real_data=0,real_IQ=0,checkpoint_loads=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
