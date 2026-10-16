"""Metadata-only scientific threshold tests,0 IQ/model/updates."""
import copy
from . import design as d
from .decision import qualify
def run():
 s=dict(accuracy=.99,by_RX={str(r):dict(accuracy=.99) for r in d.FIT},by_TX={str(t):dict(accuracy=.99) for t in range(6)},by_day={str(t):dict(accuracy=.99) for t in (1,2,3)})
 scores={v['view']:copy.deepcopy(s) for v in d.views()};keys=set(scores)-{'identity'};k={n:dict(perturbation_energy=.01,kernel_relative_RMSE=.1,grid_response_max_abs_dB=.49) for n in keys};diag={n:dict(count=5040,finite_PSD_relative_RMSE=.1,boundary_error_fraction=.1,interior_relative_MSE_max=1e-25,RMS_relative_error_max=1e-8) for n in keys};stat=dict(relative_MSE=.5,better_folds=18,prediction_SSE=1.,control_SSE=2.);assert qualify(stat,k,diag,scores)['decision']=='PASS';negative=0
 name=next(n for n in keys if n.startswith('structured'))
 for kind in ('identity','TX','fidelity','crossfit','control'):
  a,b,c,e=copy.deepcopy((stat,k,diag,scores))
  if kind=='identity':e[name]['accuracy']=.98
  elif kind=='TX':e[name]['by_TX']['2']['accuracy']=.97
  elif kind=='fidelity':c[name]['finite_PSD_relative_RMSE']=.36
  elif kind=='crossfit':a['better_folds']=14
  else:a['control_SSE']=.9
  assert qualify(a,b,c,e)['decision']=='FAIL';negative+=1
 return dict(status='PASS',threshold_negative_cases=negative,source_IQ_reads=0,checkpoint_loads=0,model_forwards=0,optimizer_updates=0)
if __name__=='__main__':
 value=run();d.write(d.ROOT/d.LOCAL/'decision_checks.json',value);print(value)
