from pathlib import Path
import torch,json
from . import design as d
from .readout import scores
from experiments.cvs_stochastic_ensemble import design as s
from experiments.cvs_quality_film import design as q
from experiments.cvs_stochastic_ensemble.summarize import analyze as existing
def run():
 existing(d.REFERENCE)
 for rx in d.RX:
  data=[];first=None
  for m in range(3):
   cfg=q.config(next(r for r in q.rows() if r['held_rx']==rx and r['arm']=='CE30')) if m==0 else s.config(next(r for r in s.rows() if r['held_rx']==rx and r['member']==m));p=Path(cfg['output']);raw=torch.load(p/'source_predictions.pt',map_location='cpu',weights_only=False)
   if raw['target_access'] or (first is not None and raw['ids']!=first):raise ValueError('Frozen sourceID differs')
   first=raw['ids'];data.append(raw['logits'])
  zz=torch.stack(data)
  for row in [r for r in d.rows() if r['held_rx']==rx]:
   c=d.config(row);p=Path(c['output']);p.mkdir(parents=True,exist_ok=False);z=scores(zz,c['arm']);torch.save(dict(scores=z,ids=first,config=c,target_access=False,new_training_updates=0),p/'source_predictions.pt');d.write(p/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,count=5400,new_training_updates=0,target_access=False));print('SOURCE_FUSION '+json.dumps(dict(row_id=c['row_id'],count=5400)),flush=True)
 from .summarize import finish
 finish();d.write(d.BASE/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows=15,parents=15,new_training_updates=0,source_predictions=81000,target_access=False))
