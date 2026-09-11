"""Read-only CPU reproduction of the supplied design's structural examples."""
import json
import sys
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

W = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(W/'code'))
from cvsrffi.cross_response.roles import make_roles
from cvsrffi.cross_response.scheduler import FeedbackScheduler

def pairs(ids, axis):
    actual=set()
    for subset in combinations(ids,4):
        for rotation in range(4):
            role=make_roles(subset if axis=='tx' else (0,1,2,3), subset if axis=='rx' else (1,3,4,6), rotation)
            actual.update(combinations(getattr(role,'query_'+axis),2))
    return actual
tx,rx=tuple(range(6)),(1,3,4,6,8)
tp,rp=pairs(tx,'tx'),pairs(rx,'rx')
c=SimpleNamespace(tx_ids=(0,1,2,3),rx_ids=(1,3,4,6),day_id=1,condition_id=1,k=2,size=32,key='fixture')
records=tuple(SimpleNamespace(tx_id=t,rx_id=r,physical_sample_id=(t,r,k)) for t,r,k in product(c.tx_ids,c.rx_ids,range(2)))
b=SimpleNamespace(candidate=c,roles=make_roles(c.tx_ids,c.rx_ids,0),records=records)
def history(order):
    sch=FeedbackScheduler(update_interval=20)
    sch.history[c.key]=dict(response=0.,decision=0.,reliability=1.,noise=0.)
    for step in range(20):
        for value in (order if step==19 else (0.,0.,0.,0.)):
            sch.commit(b,step,response=value,valid_count=32)
    cov=next(iter(sch.coverage.joint.values()))
    return {'history_response':sch.history[c.key]['response'],'rectangle_valid_count':cov['valid_count'],'rectangle_physical_unique':len(cov['physical'])}
out={'baseline_commit':'f9677eaa451bf2dce4ddf254f6d1b555b7213984','scope':'synthetic CPU logic only; no model training or target scoring',
 'roles':{'query_tx_pairs':len(tp),'missing_tx_pairs':sorted(set(combinations(tx,2))-tp),'query_rx_pairs':len(rp),'missing_rx_pairs':sorted(set(combinations(rx,2))-rp),'theoretical_query_rectangles_3days':len(tp)*len(rp)*3,'observed_rectangle_universe_3days':15*10*3,'proposed_4x4_directed_role_plans':6*6},
 'late_nonzero':history((0.,0.,0.,10.)),'early_nonzero':history((10.,0.,0.,0.)),
 'expected_single_flush_history':.2*10/80,'expected_rectangle_record_exposures':80*8}
assert out['roles']['query_tx_pairs']==11 and out['roles']['query_rx_pairs']==6
assert abs(out['late_nonzero']['history_response']-2)<1e-12
assert out['late_nonzero']['rectangle_valid_count']==80*32
path=Path(__file__).with_suffix('.json')
path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
