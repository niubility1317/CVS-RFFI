from pathlib import Path
from . import design as d
from .model import installed
from experiments.cvs_pa_cross_delay.source import train as shared_train,source_validation

def details(model):return dict(identity_input_policy=model._identity_input_policy,actual_time_factorial=model._time_factorial_contract)
def epoch_diagnostic(c,state,row):
 import torch
 m=state['model'];current=dict(m.named_parameters());movement={}
 for layer in ('t1','t2','t3'):
  values=[(current[n].detach().cpu(),v) for n,v in m._time_initial_parameters.items() if '.'+layer+'.' in n]
  delta=sum((a-b).square().sum() for a,b in values).sqrt();origin=sum(b.square().sum() for a,b in values).sqrt()
  movement[layer]=dict(delta_L2=float(delta),relative_L2=float(delta/origin.clamp_min(1e-12)))
 epoch=int(row['epoch']);record=dict(epoch=epoch,layer_parameter_movement=movement)
 if epoch in (33,34):
  was_training=m.training
  import random,numpy as np
  python_rng=random.getstate();numpy_rng=np.random.get_state();loader=state['ctx']['val_loader'];generator=getattr(loader,'generator',None);loader_rng=generator.get_state() if generator is not None else None
  try:
   with torch.random.fork_rng(devices=[torch.cuda.current_device()]):record['source_V_RX_health']=source_validation(m,loader,'cuda:0')
  finally:
   random.setstate(python_rng);np.random.set_state(numpy_rng)
   if generator is not None:generator.set_state(loader_rng)
   m.train(was_training)
 state.setdefault('time_diagnostics',[]).append(record)
 d.write(Path(c['output_root'])/'source_TIME_epoch_diagnostic.json',dict(records=state['time_diagnostics'],new_V_diagnostic_passes=sum('source_V_RX_health' in v for v in state['time_diagnostics']),head_fit=False,source_only=True,checkpoint_selection='final_only'))
def train(c):return shared_train(c,design=d,installer=installed,details=details,epoch_callback=epoch_diagnostic)
