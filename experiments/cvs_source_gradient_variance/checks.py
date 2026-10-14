import json,torch
from . import design as d
from .probe import select,statistic,smoke,objects

def checks():
 y=[];rx=[];day=[]
 for tx in range(6):
  for dd in d.DAYS:
   for receiver in d.RX:
    y.extend([tx]*70);rx.extend([receiver]*70);day.extend([dd]*70)
 L=dict(y=torch.tensor(y),rx=torch.tensor(rx),day=torch.tensor(day),ids=[('synthetic',i) for i in range(6300)])
 ix=select(L);assert len(ix)==180 and torch.equal(ix,select(L));negatives=0
 for role in ('U_s','target','V'):
  try:select(L,role)
  except ValueError:negatives+=1
  else:raise AssertionError('Unauthorized gradient-source role accepted')
 yy,rr,dd=(L[k][ix] for k in ('y','rx','day'))
 G=torch.arange(180,dtype=torch.double).unsqueeze(1).repeat(1,4)
 result=statistic(G,yy,rr,dd)
 assert result['raw']['normalized_coordinate_variance_discrepancy']==0
 assert result['TX_day_centered']['normalized_coordinate_variance_discrepancy']==0
 assert all(abs(v['variance_trace']-4*9/35)<1e-12 for v in result['TX_day_centered']['records'])
 for invalid in (G[:179],G*float('nan')):
  try:statistic(invalid,yy,rr,dd)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid source derivative matrix accepted')
 from experiments.cvs_feature_multiscale50.model import build
 model=build(d.source_config(),'cpu').eval();native=smoke(model)
 try:objects(model.train())
 except ValueError:negatives+=1
 else:raise AssertionError('Train-mode observer accepted')
 return dict(status='PASS',synthetic_only=True,balanced_subset_and_unbiased_variance=True,protocol_negatives=negatives,native_no_query_derivative_smoke=native,source_IQ_read=False,query_read=False,head_fit=False,optimizer_steps=0)
if __name__=='__main__':print(json.dumps(checks()))
