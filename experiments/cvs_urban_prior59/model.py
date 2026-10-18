"""Install the unchanged P43 F/default head and six-forward objective."""
from pathlib import Path
import torch
from . import design as d
from .channel import Practical
from experiments.cvs_noise_shape_factorial43 import model as original
from experiments.cvs_clock_factorial30.model import Shifted
classifier,source_domain_count=original.classifier,original.source_domain_count

def attach(m,c):
 d.validate(c);m._full_channel=Shifted(Practical());m._feature_learning_report['urban_prior59']=c['scene_prior'];return m

def install(m,c):return attach(original.install(m,d.old_config()),c)
def build(c,device='cpu'):return attach(original.build(d.old_config(),device),c)

def own_model(c,device='cpu'):
 from .source import parent_payload
 from .protocol import final_metadata,checkpoint
 d.validate(c);parent=parent_payload(c);p=Path(c['output_root']);final_metadata(c,d.read(p/'completion.json'),d.read(p/'source_frozen.json'),d.read(p/'initialization.json'),d.read(p/'source_contract.json'),d.read(d.SOURCE)['role_ids'])
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);checkpoint(c,cp)
 if set(cp['model'])!=set(parent['model']) or set(cp['ema_model'])!=set(parent['ema_model']):raise ValueError('Own defaulthead/student/EMA keys differ')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
