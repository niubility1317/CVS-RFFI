"""Source-only parent and endpoint metadata check; no target artifact access."""
import argparse
from pathlib import Path
import torch
from . import design as d


def preflight():
 from experiments.cvs_e20_foundation.verify import verify
 from experiments.cvs_e20_budget_fishr import design as parent
 from experiments.cvs_e20_budget_fishr.evaluate import source_provenance,snapshot
 from experiments.cvs_e20_budget_fishr.source import load_source
 spec=d.read(d.ROOT/'registration/experiment.json')
 assert spec['run_id']==d.RUN and [r['config'] for r in spec['rows']]==[d.config(r) for r in d.rows()]
 torch.set_num_threads(2);evidence=[]
 for row in d.rows():
  c=d.validate(d.config(row));proof=verify(c['model_seed'],build_smoke=True)
  L,U,V,contract=load_source(c['parent_build_config'])
  assert [len(x['ids']) for x in (L,U,V)]==[6300,56700,27000] and 'y' not in U
  endpoints=[]
  for arm in c['endpoint_arms']:
   ec=parent.config(next(r for r in parent.rows() if r['arm']==arm and r['model_seed']==row['model_seed']))
   assert snapshot(ec['row_id'])[0]==ec
   path=source_provenance(ec)/'final_ssdg.pth'
   assert path.is_file()
   endpoints.append(dict(arm=arm,checkpoint=str(path),source_metadata_verified=True))
  evidence.append(dict(row_id=row['row_id'],E20=proof,source_contract=contract,endpoints=endpoints))
 return dict(status='PASS',target_access=False,U_truth=False,V_backward=False,identity_updates=0,rows=evidence)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
 d.write(a.output,preflight())
