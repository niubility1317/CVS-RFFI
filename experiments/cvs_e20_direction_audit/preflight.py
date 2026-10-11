import argparse,torch
from . import design as d

def preflight():
 from .run import load_L
 from experiments.cvs_e20_foundation.verify import verify
 from experiments.cvs_e20_cross_style.evaluate import snapshot,source_provenance
 from experiments.cvs_e20_cross_style.model import build
 spec=d.read(d.ROOT/'registration/experiment.json')
 assert spec['run_id']==d.RUN and [r['config'] for r in spec['rows']]==[d.config(r) for r in d.rows()]
 torch.set_num_threads(2);rows=[]
 for row in d.rows():
  c=d.config(row);proof=verify(c['model_seed'],build_smoke=False);L,contract=load_L(c);pc=c['parent_build_config']
  assert snapshot(pc['row_id'])[0]==pc;endpoint=source_provenance(pc)/'final_ssdg.pth';assert endpoint.is_file()
  model=build(pc,'cpu',checkpoint=endpoint).eval()
  with torch.no_grad():logits=model(L['x'][:2])
  assert logits.shape==(2,6) and torch.isfinite(logits).all()
  cells=[int(((L['y']==y)&(L['rx']==rx)&(L['day']==day)).sum()) for y in range(6) for rx in (1,3,4,6,8) for day in (1,2,3)]
  assert len(cells)==90 and min(cells)>=2
  rows.append(dict(row_id=row['row_id'],P0=proof,CE20=str(endpoint),source_contract=contract,L=6300,cells=cells,real_source_smoke_shape=list(logits.shape)))
 return dict(status='PASS',rows=rows,identity_updates=0,target_access=False,V_forward=0,U_forward=0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();d.write(a.output,preflight())
