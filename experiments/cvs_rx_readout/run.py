import json,time
from pathlib import Path
import torch
from . import design as d
from .source import own
from .readout import probabilities
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_equivariant_identity.precision import numerical_context
def run():
 torch.set_num_threads(2);L,U,V,_=load_source(d.config(d.rows()[0])['source_loader_config'])
 if U is not None:raise ValueError('U muststayclosed')
 with numerical_context(d.FULL_FP32_POLICY),torch.no_grad():
  for rx in d.RX:
   rows=[r for r in d.rows() if r['held_rx']==rx];held=V['rx']==rx;x=V['x'][held];ids=[V['ids'][i] for i in held.nonzero().flatten().tolist()];outputs={}
   for parent in d.config(rows[0])['parents']:
    m=own(parent,L,'cuda:0');before={k:v.clone() for k,v in m.state_dict().items()};aa=[];bb=[]
    for i in range(0,len(x),256):
     a,b=m.training_logits(x[i:i+256].to('cuda:0'));aa.append(a.cpu());bb.append(b.cpu())
    if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozenparentstate changed')
    outputs[parent['arm']]=(torch.cat(aa),torch.cat(bb));del m
   for row in rows:
    c=d.config(row);p=Path(c['output']);p.mkdir(parents=True,exist_ok=False);pred={arm:probabilities(a,b,c['readout']) for arm,(a,b) in outputs.items()};torch.save(dict(ids=ids,probabilities=pred,config=c,target_access=False,parameters_buffers_unchanged=True,new_training_updates=0),p/'source_predictions.pt');d.write(p/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,new_training_updates=0,held_V_count=5400,source_only=True,target_access=False,parameters_buffers_unchanged=True));print('SOURCE_READOUT '+json.dumps(dict(row_id=c['row_id'],count=5400,parents=2)),flush=True)
 from .summarize import finish
 finish();d.write(d.BASE/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows=15,parents=10,new_training_updates=0,source_predictions=162000,target_access=False))
