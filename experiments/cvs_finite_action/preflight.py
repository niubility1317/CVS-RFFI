import argparse,torch
from . import design as d
from .observations import load,extract
from experiments.cvs_nocfo_engineering.teacher import load as reference
from .state import snapshot
from experiments.cvs_equivariant_identity.precision import numerical_context
def preflight():
 spec=d.read(d.ROOT/'registration/experiment.json');assert [r['config'] for r in spec['rows']]==[d.config(r) for r in d.rows()];torch.set_num_threads(2);records=[]
 with numerical_context(d.FULL_FP32_POLICY):
  for row in d.rows():
   c=d.config(row);L,U,contract=load(c);model,proof=reference(c['model_seed'],'cpu');before=snapshot(model)
   small=lambda cache:{k:(v[:8] if k!='role' else v) for k,v in cache.items() if k in ('x','y','rx','day','ids','role')}
   lo=extract(model,small(L),c,'cpu','L_s');uo=extract(model,small(U),c,'cpu','U_s');after=snapshot(model)
   assert all(torch.equal(before[k],after[k]) for k in before) and lo['counts']['packet_occurrences']==120 and uo['counts']['packet_occurrences']==56
   records.append(dict(row_id=row['row_id'],checkpoint_provenance=proof,source_contract=contract,real_L=120,real_U=56,U_truth=False,reference_state_unchanged=True,actual_layers=list(lo['h']),identity_updates=0,target_access=False))
 return dict(status='PASS',rows=records,identity_updates=0,target_access=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--remote-preflight',action='store_true');d.write(p.parse_args().output,preflight())
