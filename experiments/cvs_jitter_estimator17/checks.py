"""Focused fixed-scope negatives plus exact arithmetic counterexamples."""
import inspect,json
from copy import deepcopy
import numpy as np
from . import design as d
from .estimator import estimate,derivative,symcross,pair_basis,solve_moments
from .synthetic import template

def main():
 negatives=0
 for key,value in [('packets_per_cell',71),('cycles',5),('period',21),('draws',33),('data_seed',1),('cells',1),('scope','source_data'),('on_fail','retry')]:
  c=deepcopy(d.config());c[key]=value
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError(key)
 for key in ['real_IQ_reads','checkpoints','NN','updates','teacher_forwards','head_fit_updates','GPU','target_access']:
  c=deepcopy(d.config());c['cost'][key]=1
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError(key)
 z,dz=template('multitone_quadratic');y=np.broadcast_to(z,(70,4,20)).copy()
 invalid=[y[:69],y[:,:3],y[:,:,:19],y.real.copy()]
 for item in (complex('nan'),complex('inf')):
  q=y.copy();q[0,0,0]=item;invalid.append(q)
 for q in invalid:
  try:estimate(q)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid observations accepted')
 scientific={}
 scientific['no_oracle_or_metadata_argument']=list(inspect.signature(estimate).parameters)==['observations']
 before=y.copy();v=estimate(y);scientific['input_unchanged']=np.array_equal(y,before)
 scientific['noiseless_covariance_zero']=np.array_equal(np.array(v['rhs']),np.zeros(4))
 scientific['noiseless_full_rank']=v['status']=='IDENTIFIABLE' and max(map(abs,v['variances']))==0
 scientific['analytic_periodic_derivative']=np.allclose(derivative(z),dz,atol=2e-14,rtol=2e-14)
 a=np.array([1+2j,3-4j]);b=np.array([2-3j,4+1j]);cross=symcross(a,b)
 explicit=np.array([[.5*(np.array([x.real,x.imag])[:,None]@np.array([w.real,w.imag])[None,:]+np.array([w.real,w.imag])[:,None]@np.array([x.real,x.imag])[None,:]) for x,w in zip(a,b)]])[0]
 scientific['cross_outer_exact']=np.allclose(cross,np.stack([explicit[:,0,0],np.sqrt(2.)*explicit[:,0,1],explicit[:,1,1]],axis=1),atol=1e-15)
 G=np.array(v['Gram']);scientific['noiseless_Gram_exact']=np.allclose(G,pair_basis(y,derivative(y),(0,1))[0].T@pair_basis(y,derivative(y),(0,1))[0],atol=1e-11)
 toy=np.diag([2.,3.,4.,5.]);true=np.array([.01,.02,.03,.04]);fit=solve_moments(toy,toy*toy,toy@true)
 scientific['exact_four_variance_recovery']=np.allclose(fit['variances'],true,atol=1e-14)
 for name in ['tone','constant','zero']:
  z,_=template(name);res=estimate(np.broadcast_to(z,(70,4,20)))
  scientific[name+'_unidentifiable']=res['status'].startswith('UNIDENTIFIABLE')
 assert all(scientific.values()),scientific
 print(json.dumps(dict(status='PASS',protocol_negatives=negatives,science_checks=scientific,synthetic_noiseless_estimator_calls=4,synthetic_noiseless_estimator_observations=22400,real_IQ_reads=0,checkpoint_loads=0,NN=0,updates=0,target_access=False)))
if __name__=='__main__':main()
