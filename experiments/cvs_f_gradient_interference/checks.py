import math
from .core import measure,decision

def run():
 for a,b in [([1.,2.],[2.,-1.]),([2.,0.],[-3.,0.]),([0.,0.],[1.,0.]),([1e-8,2e-8],[3e-8,1e-8])]:
  p2=sum(x*x for x in a);s2=sum(x*x for x in b);dot=sum(x*y for x,y in zip(a,b));r=measure(p2,s2,dot)
  assert math.isclose(r['joint_norm'],math.sqrt(sum((x+y)**2 for x,y in zip(a,b))),rel_tol=1e-9,abs_tol=1e-14)
 for bad in [(-1.,1.,0.),(1.,1.,2.),(float('nan'),1.,0.),(1.,float('inf'),0.)]:
  try:measure(*bad)
  except ValueError:pass
  else:raise AssertionError('Bad gradient scalar accepted')
 rows=[]
 for model in ('C0','C1'):
  for rep in (0,1):
   for batch in range(40):
    for second in ('same_ordinary','As'):rows.append(dict(checkpoint=model,rep=rep,batch=batch,secondary=second,group='whole',**measure(1.,4. if second=='As' else 1.,-1. if second=='As' else 1.)))
 assert decision(rows)['status']=='SUPPORTED_FOR_SEPARATE_SOURCE_EXPERIMENT_ONLY'
 rows[-1]['opposed']=False
 # Still passes aggregate; losing a whole replicate's effect must reject.
 for r in rows:
  if r['checkpoint']=='C1' and r['rep']==1 and r['secondary']=='As':r['opposed']=False
 assert decision(rows)['status']=='RATIONALE_REJECTED'
 try:decision(rows[:-1])
 except ValueError:pass
 else:raise AssertionError('Partial source geometry accepted')
 return dict(status='PASS',analytic_gradient_pairs=4,scalar_negatives=4,all_replica_rule_checks=3,IQ_reads=0,NN=0,updates=0)

def model_checks():
 import torch
 from . import audit,design as d
 from experiments.cvs_f_crossrx_practical20 import model as api
 torch.set_num_threads(2);torch.manual_seed(1841)
 m=api.build(d.prior.config(d.prior.rows()[0]),'cpu');m.train();before={k:v.clone() for k,v in m.state_dict().items()};named=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
 assert not any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in m.modules())
 x=torch.randn(8,2,256);y=torch.tensor([0,0,1,1,2,2,3,3]);rx=torch.tensor([3,4]*4);domains=torch.tensor([3,6]*4)
 p,terms,logits=audit.primary(m,x,y,domains,rx,named,1)
 g1,t1,l1=audit.secondary(m,x,y,domains,named,1);g2,t2,l2=audit.secondary(m,x,y,domains,named,1)
 assert all(torch.equal(g1[k],g2[k]) for k in g1) and torch.equal(l1,l2)
 assert all(not v.requires_grad for v in p.values()) and all(v>0 for v in (float(p['default_head'].norm()),float(p['SIG_TIME'].norm()),float(p['Sinc_Hz'].norm())))
 geom=audit.geometry(p,g1);assert set(geom)==set(audit.GROUPS)|{'whole'}
 assert math.isclose(geom['whole']['primary_squared'],sum(geom[k]['primary_squared'] for k in audit.GROUPS),rel_tol=1e-12)
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(v.grad is None for _,v in named)
 q=torch.nn.Parameter(torch.tensor([2.]));g=audit.gradients((3*q).sum(),[('foo.sinc.low_coordinate',q)])
 assert torch.allclose(g['Sinc_Hz'],torch.tensor([3/25000],dtype=torch.float64),rtol=1e-7,atol=1e-12)
 c=d.config(d.rows()[0]);negatives=0
 for key,val in [('target_access',True),('optimizer_updates',1),('head_fit_updates',1),('V_values_used',True),('U_values_used',True),('source_L_count',6300),('model_seed',2026092702),('batch_size',64),('replicates',[0]),('held_RX1_used',True)]:
  import copy
  bad=copy.deepcopy(c);bad[key]=val
  try:d.validate(bad)
  except ValueError:negatives+=1
  else:raise AssertionError('Protocol/config mutation accepted')
 return dict(status='PASS',synthetic_model_packets=48,real_IQ_reads=0,checkpoint_loads=0,optimizer_updates=0,protocol_negatives=negatives,default_head_SIG3_Sinc_gradients=True,physical_Sinc_oracle=True,same_secondary_RNG_and_gradients=True,parameters_buffers_unchanged=True)

def serialization_checks():
 import json
 from . import design as d
 from dataclasses import asdict,dataclass
 c=d.config(d.rows()[0]);assert json.loads(json.dumps(c))==c and '\\' not in c['source_preparation'] and all('\\' not in p for p in c['references'].values())
 @dataclass
 class Config:values:tuple=(1.,2.)
 assert json.loads(json.dumps(asdict(Config())))=={'values':[1.,2.]}
 assert json.loads(json.dumps(asdict(Config())))!={'values':[1.,3.]}
 return dict(status='PASS',portable_config_roundtrip=True,tuple_JSON_equivalence=True,actual_value_difference_rejected=True)
