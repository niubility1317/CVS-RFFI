import torch
from copy import deepcopy
from . import design as d
from experiments.cvs_nocfo_foundation import design as n
from experiments.cvs_nocfo_foundation.model import groups as native_groups
from experiments.cvs_stage2_rx_risk.model import OriginalIdentity
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_source_identity_readout.input_policy import apply

def build_member(c,device):
 # Source-loader config identifies roles/architecture. A member's actual
 # initialization and dropout seed is separately frozen by the ensemble config.
 nc=c['source_loader_config'];n.validate(nc)
 reference=deepcopy(nc['reference_config']);reference['model_seed']=c['member_init_seed']
 args=n.baseline.make_args(nc['reference_config'],str(device));args.num_domains=5;args.input_len=256
 args.from_scratch=True;args.baseline_ckpt='';args.teacher_ckpt=''
 torch.manual_seed(c['member_init_seed'])
 with installed(reference) as native:base=native.build_baseline_model(args,torch.device(device))
 base._stage2_baseline_args=vars(args).copy();model=OriginalIdentity(base).to(device)
 model._identity_input_policy=apply(model);model._identity_input_policy.update(historical_checkpoint_training_policy=None,function_policy='no_explicit_CFO_training_and_prediction',source_training_policy='no_explicit_CFO')
 model._current_q=None;model.layer_info=[dict(level='native_ensemble_member',member=c['member'],actual_initialization_seed=c['member_init_seed'])]
 return model
def build(c,fit_x,device):d.validate(c);return build_member(c,device)
def groups(model,c):return native_groups(model,c['source_loader_config'])
def probabilities(logits):
 if len(logits)!=3 or any(x.shape!=logits[0].shape or x.ndim!=2 or x.shape[1]!=6 or not torch.isfinite(x).all() for x in logits):raise ValueError('Three fixed six-class finite logits required')
 return torch.stack([x.softmax(1) for x in logits]).mean(0)
