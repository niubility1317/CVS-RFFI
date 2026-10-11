"""Bounded native-architecture integration using synthetic IQ only."""
import argparse
from pathlib import Path
import tempfile
import torch
from . import design as d
from .source import forward,save_completion,checkpoint,SourceIdentityView
from .optim import TemporarySAM,WeightAverage
from . import actions as a
from experiments.cvs_stage2_decouple.common import parameter_groups
from experiments.cvs_stage2_decouple.design import make_args
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_stage2_geometry.receiver_checks import synthetic_cache
from experiments.cvs_stage2_geometry.receiver import fit_receiver,load_receiver
from experiments.cvs_stage2_decouple.source_counterfactual_audit import RNGState


def check(device):
 torch.set_num_threads(2);seed=d.SEEDS[0];pc=d.parent_config(seed);args=make_args(pc,device);args.num_domains,args.input_len=5,256
 with numerical_context(d.FULL_FP32_POLICY),installed(pc) as native:model=native.build_baseline_model(args,torch.device(device))
 model._stage2_baseline_args=vars(args)
 assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
 groups,names=parameter_groups(model);opt=torch.optim.AdamW(groups);model.train()
 cache=synthetic_cache();cache['x']=torch.randn(360,2,256);cache['state']=cache['state'].float()
 bank=load_receiver(fit_receiver(cache,seed));r=bank.sample_relations(torch.Generator().manual_seed(127));ix=torch.cat((torch.arange(96),r['indices']))
 payload=dict(version=a.VERSION,contract=a.contract(),seed=seed,networks={k:a.new_net(k,a.a._generator(seed+i),device).state_dict() for i,k in enumerate(a.a.KINDS)})
 actions=a.FrozenBalanced(payload,device)
 records=[];calls=[];hook=model.dom_backbone.register_forward_hook(lambda *args:calls.append(1))
 try:
  for arm in d.ARMS:
   c=d.config(dict(arm=arm,model_seed=seed,row_id=f'{arm}-s{seed}'));opt.zero_grad(set_to_none=True);rng=RNGState()
   loss,w,parts,detail=forward(c,model,cache,ix,r,888,device,actions)
   assert torch.isfinite(loss)
   if arm=='H0':assert torch.equal(loss,parts['ce']) and not any(k in detail for k in ('contrastive','actions','cross_rx_mix','manifold_mix'))
   loss.backward()
   if c['enabled']['sam']:
    before={n:p.detach().clone() for n,p in model.named_parameters()};after=RNGState();sam=TemporarySAM(groups,.02);sam.perturb();opt.zero_grad(set_to_none=True);rng.restore()
    second,*_=forward(c,model,cache,ix,r,888,device,actions);second.backward();sam.restore();after.restore()
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters()) and after.matches()
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for g in groups for p in g['params'])
   records.append(dict(arm=arm,loss=float(loss.detach()),sam=c['enabled']['sam']))
  assert not calls
  opt.step()
  avg=WeightAverage(model,[n for values in names.values() for n in values])
  for epoch in [32,34,36,38,40]:avg.add(model,epoch)
  avg.apply(model,[32,34,36,38,40])
  c=d.config(dict(arm='H9',model_seed=seed,row_id=f'H9-s{seed}'))
  from .evaluate import checkpoint_provenance
  with tempfile.TemporaryDirectory(prefix='hybrid-native-') as folder:
   out=Path(folder)/'source';out.mkdir()
   path=checkpoint(c,model,opt,out,averaging=dict(count=avg.count,epochs=avg.epochs));checkpoint_provenance(c,torch.load(path,map_location='cpu',weights_only=False))
   save_completion(c,out,dict(status='fixture'));f=d.read(Path(folder)/'source_frozen.json')
   assert f['selection']=='fixed_tail_average'
  from experiments.cvs_feature_disentangle.validation import evaluate_source_stress
  tiny=[(cache['x'][:5].to(device),cache['y'][:5].to(device),torch.zeros(5,dtype=torch.long),dict(rx_i=torch.tensor([1,3,4,6,8]),day_i=torch.zeros(5,dtype=torch.long)))]
  with tempfile.TemporaryDirectory(prefix='hybrid-stress-') as folder:
   try:evaluate_source_stress(SourceIdentityView(model),tiny,device,c,folder)
   except ValueError as error:assert str(error)=='Incomplete source V pressure validation',str(error)
   else:raise AssertionError('Incomplete V accepted')
 finally:hook.remove()
 return dict(status='PASS',device=device,target_access=False,synthetic_IQ=True,native_forward_backward_11_arms=records,SAM_same_rng_exact_restore=True,average_checkpoint_provenance=True,source_diagnostic_interface=True,domain_forwards=0)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);args=p.parse_args()
 result=check(args.device);d.write(args.output,result);print('PASS native hybrid graph, 11 arms, SAM, average, source stress')
