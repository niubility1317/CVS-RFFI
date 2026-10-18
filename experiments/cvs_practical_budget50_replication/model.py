"""Original seed-aware F factory and graph; unchanged complete As at E31..50."""
from pathlib import Path
import torch
from . import design as d
from .channel import Practical
from .protocol import cumulative,check_checkpoint,check_source_metadata
from experiments.cvs_As_replicate49 import model as original
classifier,source_domain_count=original.classifier,original.source_domain_count

class Shifted:
 def __init__(self,obj):self.obj=obj
 def __getattr__(self,k):
  if k=='obj':raise AttributeError(k)
  return getattr(self.obj,k)
 def apply(self,x,meta,epoch,step):
  epoch,step=cumulative(epoch,step);return self.obj.apply(x,meta,epoch=epoch,step=step)

def original_config(c):return d.reference_config(c,30)
def attach(m,c):
 d.validate(c)
 if m._replication_factory_seed!=c['model_seed']:raise ValueError('Actual original F factory seed differs')
 m._full_channel=Shifted(Practical('train')) if c['As'] else original.impl.NoAs();m._shape_offset=6660
 m._feature_learning_report['E50_budget_extension']=dict(parent=c['parent_source'],parent_epochs=30,new_epochs=20,cumulative_epochs=50,epoch_offset=30,step_offset=6660,optimizer='resetAdamW/cos20',EMA='strictownFINAL30',same_original_dropout=True,same_original_satellite_KL=True,no_new_parameters=True,actual_F_factory_seed=m._replication_factory_seed)
 return m

def install(m,c):return attach(original.install(m,original_config(c)),c)
def build(c,device='cpu'):return attach(original.build(original_config(c),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 parent_payload(c);p=Path(c['output_root'])
 check_source_metadata(d.read(p/'completion.json'),d.read(p/'source_frozen.json'),d.read(p/'initialization.json'),d.read(p/'source_contract.json'),c,d.read(d.SOURCE)['role_ids'],20,4440)
 done=d.read(p/'completion.json')
 if done['cumulative_epochs']!=50 or done['cumulative_optimizer_steps']!=11100:raise ValueError('Only fixed cumulativeFINAL50')
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 check_checkpoint(cp,seed=c['model_seed'],run=d.RUN,row=c['row_id'],epoch=20,updates=4440,baseline=c['parent_source']+'/final_ssdg.pth')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
