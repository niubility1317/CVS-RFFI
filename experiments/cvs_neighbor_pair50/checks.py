import copy,json
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from . import design as d
from .sampler import replace
from .model import build

class Index:
 def __init__(self,rx,day,sig):self.rx_i=rx;self.day_i=day;self.eq_i=0;self.sig_i=sig
 @property
 def tx_i(self):raise AssertionError('U TX truth accessed')
class Hidden(Dataset):
 hide=True;split_source='source_unlabeled'
 def __init__(self):
  self.selected=np.concatenate([np.arange(70,700)+i*1000 for i in range(90)]);self.index=[Index([1,3,4,6,8][(i//3)%5],i%3+1,sig) for i in range(90) for sig in range(70,700)]
 def __len__(self):return len(self.selected)
 def __getitem__(self,i):return i

def checks():
 torch.set_num_threads(2);ds=Hidden();base=DataLoader(ds,batch_size=256,shuffle=True,drop_last=False,num_workers=0)
 rng=torch.random.get_rng_state().clone();random=replace(base,'RANDOM');paired=replace(base,'PAIRS');assert torch.equal(rng,torch.random.get_rng_state())
 def collect(loader):
  torch.manual_seed(2026101450);values=list(loader);return torch.cat(values),torch.random.get_rng_state().clone(),[len(v) for v in values]
 old,old_rng,lengths=collect(base);new,new_rng,newlengths=collect(random);pairs,pair_rng,pairlengths=collect(paired)
 assert torch.equal(old,new) and torch.equal(old_rng,new_rng) and torch.equal(old_rng,pair_rng)
 assert lengths==newlengths==pairlengths==[256]*221+[124]
 assert torch.equal(pairs.sort().values,torch.arange(56700))
 physical=torch.tensor(ds.selected.tolist(),dtype=torch.long)[pairs].reshape(-1,2);assert (physical.diff(dim=1).abs()==1).all()
 assert random.sampler.history[-1]['delivered_count']==paired.sampler.history[-1]['delivered_count']==56700
 from experiments.cvs_phase1_overlay.model import native_modules
 native=native_modules();L=DataLoader(torch.arange(6300),batch_size=128,shuffle=True,drop_last=True)
 before=len(paired.sampler.history);label_batches=list(native._muse_epoch_pairs(L,paired,use_muse=False,use_unlabeled_step_budget=True))
 assert len(label_batches)==222 and all(u is None for l,u in label_batches) and len(paired.sampler.history)==before
 c=d.config(d.training_rows()[0]);torch.manual_seed(c['model_seed']);model=build(c);model.eval().requires_grad_(False)
 from experiments.cvs_feature_multiscale50.model import build as original
 oldmodel=original(d.baseline_config(c),'cpu').eval().requires_grad_(False)
 assert set(model.state_dict())==set(oldmodel.state_dict()) and all(torch.equal(v,oldmodel.state_dict()[k]) for k,v in model.state_dict().items())
 x=torch.randn(2,2,256)
 from experiments.cvs_transfer_features.model import from_phase1
 with torch.no_grad():
  out=model.forward_identity_only(x);other=oldmodel.forward_identity_only(x);f=from_phase1(model).eval().requires_grad_(False)
  assert torch.equal(out['tx_logits'],other['tx_logits']) and torch.equal(out['id_feat_joint'],other['id_feat_joint']) and torch.equal(f.forward_features(x),out['id_feat_joint'])
  clone=copy.deepcopy(model);clone.id_backbone.encoder.core.id_backbone.cls_head.weight.zero_();assert torch.equal(clone.forward_identity_only(x)['id_feat_joint'],out['id_feat_joint'])
 assert sum(p.numel() for p in model.id_backbone.parameters())==177025 and sum(p.numel() for p in f.parameters())==176065
 negative=0
 for key,value in [('epochs',51),('readouts',['FIT_HEAD']),('target_access',True),('model_initialization','old_target_best')]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid training protocol accepted')
 for mode in ['UNKNOWN']:
  try:replace(base,mode)
  except ValueError:negative+=1
  else:raise AssertionError('Unknown U order accepted')
 ds.hide=False
 try:replace(base,'PAIRS')
 except ValueError:negative+=1
 else:raise AssertionError('Visible U labels accepted')
 return dict(status='PASS',same_initial_all_states_and_default_features=True,actual_feature_export_exact=True,W_independent=True,identity_parameters=177025,feature_parameters=176065,matched_random_loader_exact=True,outside_sampler_RNG_identical=True,hidden_TX_access_trap_pass=True,source_U_pool=56700,batches=222,tail=124,label_epoch_never_fetches_U=True,negative_checks=negative,synthetic_only=True,no_real_IQ_or_query=True,no_head_fit=True)
if __name__=='__main__':print(json.dumps(checks()))
