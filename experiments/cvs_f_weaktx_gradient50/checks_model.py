"""One actual same-contract BASE checkpoint, synthetic no-query feature VJPs."""
import argparse,json,torch
from collections import Counter
from core import *
from loaders import imports
def run(a):
 d,api,validate,base_penalty,tail,precision,configs=imports(a);torch.set_num_threads(2)
 with precision(d.FULL_FP32_POLICY):
  c=configs[0];cp=torch.load(c['output_root']+'/final_ssdg.pth',map_location='cpu',weights_only=False);validate(cp,c);m=api.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.eval();parameters=select(m);assert any('.sequence_pool.views.0.' in n for n,p in parameters), 'EXPECTED_RED: deployed SIG_TIME projection missing from feature scope';before={n:v.clone() for n,v in m.state_dict().items()};modes=[x.training for x in m.modules()];rng=torch.get_rng_state().clone();maps={};handles=[]
  for level in ('time_down','t2.pool'):
   handles.append(m.get_submodule(PREFIX+level).register_forward_hook(lambda module,args,out,level=level:maps.__setitem__(level,out)))
  try:
   gen=torch.Generator().manual_seed(2026102007);x=torch.randn(18,2,256,generator=gen);y=torch.arange(18)//3;rx=torch.tensor([1,3,4]*6);out=m.forward_identity_only(x);values,info=objectives(out,maps,y,rx,base_penalty,tail);W=m.get_submodule(PREFIX+'cls_head').weight
   assert all(float(v.detach())>0 for v in values[6:]);assert torch.autograd.grad(sum(values[6:]),W,retain_graph=True,allow_unused=True)[0] is None
   sinc=m.get_submodule(PREFIX+'sinc').low_coordinate;assert torch.autograd.grad(sum(values[6:]),sinc,retain_graph=True)[0].abs().sum()>0
   raw,unused=vectors(m,values,parameters);gg=grams(raw,parameters);assert all(bool(torch.isfinite(v['total']).all()) for v in gg.values());assert raw.shape[0]==11 and raw.abs().sum()>0
  finally:
   for h in handles:h.remove()
  maps={};handles=[]
  for level in ('time_down','t2.pool'):handles.append(m.get_submodule(PREFIX+level).register_forward_hook(lambda module,args,out,level=level:maps.__setitem__(level,out)))
  try:
   output2=m.forward_identity_only(x.flip(0));values2,_=objectives(output2,maps,y.flip(0),rx.flip(0),base_penalty,tail);raw2,_=vectors(m,values2,parameters);assert bool(torch.isfinite(raw2).all())
  finally:
   for h in handles:h.remove()
  assert all(torch.equal(v,m.state_dict()[n]) for n,v in before.items()) and all(p.grad is None for p in m.parameters()) and modes==[x.training for x in m.modules()] and torch.equal(rng,torch.get_rng_state())
  counts=Counter(n.split('.')[0] for n,p in m.named_parameters());outside=[n for n,p in parameters if not n.startswith(PREFIX)]
  return dict(status='PASS',actual_original_F_checkpoint=True,no_query_smoke=True,synthetic_examples=36,repeated_forward_VJP_graph=True,feature_parameters=sum(p.numel() for _,p in parameters),total_parameters=sum(p.numel() for p in m.parameters()),full_feature_VJP_shape=list(raw.shape),objective_names=list(OBJECTIVES),state_grad_RNG_modes_unchanged=True,default_W_aux_gradient=None,unused_selected_parameters=unused,outside_identity_prefix=outside,parameter_root_counts=dict(counts),IQ_reads=0,target_reads=0,optimizer_updates=0,head_fit_updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-run',required=True);p.add_argument('--source-release',required=True);p.add_argument('--profile',required=True);a=p.parse_args();print(json.dumps(run(a)))
