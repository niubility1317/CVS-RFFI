from types import SimpleNamespace
import torch
from . import design as d
from experiments.cvs_stage2_rx_risk.model import OriginalIdentity
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_stage2_decouple.common import parameter_groups

def build(c,device,checkpoint=None):
 d.validate(c)
 # Build from static native arguments; no parent checkpoint is opened.
 args=d.baseline.make_args(c['reference_config'],str(device));args.num_domains=5;args.input_len=256
 torch.manual_seed(c['model_seed'])
 with installed(c) as native:base=native.build_baseline_model(args,torch.device(device))
 base._stage2_baseline_args=vars(args).copy()
 model=OriginalIdentity(base).to(device)
 if c['multilevel']['enabled']:
  from .multilevel import MultiLevel
  model.multilevel=MultiLevel(base,seed=c['model_seed']).to(device)
 if checkpoint is not None:
  ck=torch.load(checkpoint,map_location=device,weights_only=False)
  expected=dict(config=c,epoch=20,optimizer_steps=4440,parent_checkpoint=None,checkpoint_selection='final_only',target_access=False)
  if any(ck.get(k)!=v for k,v in expected.items()):raise ValueError('Own scratch checkpoint identity differs')
  model.load_state_dict(ck['model'],strict=True)
 return model

def groups(model,c):
 pg,names=parameter_groups(model.base)
 for group in pg:group['lr']=c['optimizer']['peak_lr']
 names={key:['base.'+n for n in value] for key,value in names.items()}
 if c['multilevel']['enabled']:
  pg.append(dict(name='multilevel_heads',params=list(model.multilevel.parameters()),lr=c['optimizer']['peak_lr']))
  names['multilevel_heads']=['multilevel.'+n for n,p in model.multilevel.named_parameters()]
 return pg,names
