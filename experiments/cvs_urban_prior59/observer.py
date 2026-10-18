"""Reuse original optimizer/EMA/ordinary schedule and actual six-forward counters."""
import json,sys
from pathlib import Path
from . import design as d,model
from .source import parent_payload,segment_train
from experiments.cvs_noise_shape_factorial43.observer import Observer as Original

class Observer(Original):
 def __init__(self,c):self.c=d.validate(c);self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c)
 def initialize(self,m,c,out):
  super().initialize(m,c,out)
  actual=dict(config=c,channel=m._full_channel.audit(),modules={k:str(v.__file__) for k,v in sys.modules.items() if k in ['experiments.cvs_urban_prior59.channel','experiments.cvs_noise_shape_factorial43.model','SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual']},q=None)
  d.write(self.out/'urban_prior_actual.json',actual);print(json.dumps(dict(stage='ACTUAL_MODULES_AND_SCENE_PRIOR',**actual)),flush=True)
 def finish(self):
  result=super().finish();coverage=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in sorted(self.model._full_coverage.items())]
  result.update(new_epochs=10,parent_epochs=20,cumulative_epochs=30,new_updates=2220,parent_updates=4440,cumulative_updates=6660,scene_prior=self.c['scene_prior'],TX_RX_scene_state_lock=coverage,q=None,source_RX_holdout=None)
  if result['L_NN']!=1704960 or result['L_As']!=142080 or result['Ag_packets'] or result['extra_U_NN'] or result['head_fit_updates'] or result['U_truth_read']:raise ValueError('Unchanged sixL/allLAs/oldU/defaulthead budget differs')
  d.write(self.out/'result.json',result);return result

def train(c):
 observer=Observer(c);segment_train(c,design=d,model_api=model,observer=observer);return observer.finish()
