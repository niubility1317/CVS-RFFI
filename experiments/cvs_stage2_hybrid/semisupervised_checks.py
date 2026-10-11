"""Meaningful no-truth checks for the next separately registered SSL run."""
import argparse
from pathlib import Path
import torch
from .semisupervised import UnlabeledStream,VisitPassBank,confidence_mask,EMAIdentity,validate_unlabeled,UNLABELED_KEYS
from . import design as d


def check(device='cpu'):
 torch.set_num_threads(2)
 s=UnlabeledStream(11,13);ix,visits=s.batch(24)
 assert sorted(ix[:11].tolist())==list(range(11)) and sorted(ix[11:22].tolist())==list(range(11))
 assert visits.tolist()==[1]*11+[2]*11+[3]*2
 bank=VisitPassBank();pred=torch.tensor([0,1]);conf=torch.tensor([.99,.99]);ids=['physical_A','physical_B']
 mask,_=bank.update(ids,pred,conf,torch.ones(2,dtype=torch.long));assert not mask.any()
 mask,_=bank.update(ids,pred,conf,torch.ones(2,dtype=torch.long)*2);assert mask.all()
 mask,_=bank.update(ids,torch.tensor([2,1]),conf,torch.ones(2,dtype=torch.long)*3);assert mask.tolist()==[False,True]
 # Repetition inside one pass must never manufacture an additional streak.
 fresh=VisitPassBank();fresh.update(ids,pred,conf,torch.ones(2,dtype=torch.long));mask,_=fresh.update(ids,pred,conf,torch.ones(2,dtype=torch.long));assert not mask.any()
 cache={k:torch.zeros(2,dtype=torch.long) for k in ('rx','day','domain','eq','sig','base_index')}
 cache.update(x=torch.randn(2,2,256),ids=ids,role='U_s',preprocessing={});assert validate_unlabeled(cache)==2
 for forbidden in ('y','label','tx','parent_margin','parent_feature'):
  try:validate_unlabeled(dict(cache,**{forbidden:torch.zeros(2)}))
  except ValueError:pass
  else:raise AssertionError('U truth/derived-supervision accepted')
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_stage2_decouple.design import make_args
 from experiments.cvs_stage2_geometry.features import identity
 from experiments.cvs_phase1_overlay.model import native_modules
 pc=d.parent_config(d.SEEDS[0]);args=make_args(pc,device);args.num_domains,args.input_len=5,256
 with installed(pc) as native:model=native.build_baseline_model(args,torch.device(device))
 ema=EMAIdentity(model,decay=.5);x=torch.randn(8,2,256,device=device);ema.logits(x)
 key='id_backbone.encoder.core.id_backbone.cls_head.weight';weight=dict(ema.model.named_parameters())[key];version=weight._version;before=weight.clone()
 with torch.no_grad():dict(model.named_parameters())[key].add_(torch.randn_like(weight)*.1)
 expected=.5*before+.5*dict(model.named_parameters())[key].detach();ema.update(model)
 assert weight._version>version and torch.allclose(weight,expected) and ema.updates==1
 cached=ema.logits(x)
 with torch.enable_grad():fresh_logits=identity(ema.model,x)[0]
 torch.testing.assert_close(cached,fresh_logits,atol=3e-5,rtol=3e-6)
 assert not any(p.requires_grad for p in ema.model.parameters()) and not any(p.grad is not None for p in ema.model.parameters())
 confidence=torch.tensor([.91,.93,.99,.95,.96]);domains=torch.tensor([0,0,0,1,1]);n=native_modules()
 from types import SimpleNamespace
 legacy=SimpleNamespace(tau_min=.92,tau_max=.97,pseudo_quantile=.86,pseudo_threshold_mode='rx_day_quantile')
 assert torch.equal(confidence_mask(confidence,domains),n._threshold_mask(confidence,domains,legacy))
 return dict(status='PASS',device=device,synthetic_only=True,target_access=False,checks=['full_U_permutation_before_reuse','per_physical_ID_pass_history','same_pass_cannot_advance_streak','prediction_change_resets_streak','U_truth_schema_rejected','EMA_versions_increment','native_cached_vs_fresh_teacher_logits_match','EMA_no_gradient','confidence_gate_matches_repaired_native'])


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);args=p.parse_args();r=check(args.device);d.write(args.output,r);print('PASS truth-blind SSL history and native EMA-cache invalidation')
