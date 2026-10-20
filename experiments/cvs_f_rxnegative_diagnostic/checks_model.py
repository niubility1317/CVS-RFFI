"""One true original checkpoint, synthetic no-query full-feature gradients."""
import argparse,json,torch
from core import *
from loaders import imports
def run(a):
 d,api,validate,precision,configs=imports(a);torch.set_num_threads(2)
 with precision(d.FULL_FP32_POLICY):
  c=configs[0];cp=torch.load(c['output_root']+'/final_ssdg.pth',map_location='cpu',weights_only=False);validate(cp,c);m=api.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.eval();params=select(m);before={n:v.clone() for n,v in m.state_dict().items()};modes=[x.training for x in m.modules()];rng=torch.get_rng_state().clone();maps={};handles=[]
  for level in LEVELS:handles.append(m.get_submodule(PREFIX+level).register_forward_hook(lambda module,args,out,level=level:maps.__setitem__(level,out)))
  try:
   gen=torch.Generator().manual_seed(PARTITION_SEED);x=torch.randn(18,2,256,generator=gen);y=torch.arange(18)//3;rx=torch.tensor([1,3,4]*6);out=m.forward_identity_only(x);values,info=objectives(out,maps,y,rx,1);head=[p for n,p in m.named_parameters() if n.startswith(PREFIX+'cls_head.')];assert len(head)==6 and all(g is None for g in torch.autograd.grad(sum(values[3:]),head,allow_unused=True,retain_graph=True));sinc=m.get_submodule(PREFIX+'sinc').low_coordinate;assert torch.autograd.grad(sum(values[3:]),sinc,retain_graph=True)[0].norm()>0;raw=vectors(values,params);gg=summary(raw,params);assert raw.shape[0]==6 and all(all(x is not None for x in row) for v in gg.values() for row in v['cosine'])
  finally:
   for h in handles:h.remove()
  assert all(torch.equal(v,m.state_dict()[n]) for n,v in before.items()) and all(p.grad is None for p in m.parameters()) and modes==[x.training for x in m.modules()] and torch.equal(rng,torch.get_rng_state())
  return dict(status='PASS',real_checkpoint_no_query_smoke=True,source_release_commit=REF_COMMIT,total_parameters=sum(p.numel() for p in m.parameters()),selected_feature_parameters=sum(p.numel() for n,p in params),default_all6_head_aux_gradients_None=True,Sinc_feature_aux_gradient_nonzero=True,full_feature_VJP_objectives=list(OBJECTIVES),synthetic_forward_examples=18,state_grad_RNG_modes_unchanged=True,IQ_reads=0,target_reads=0,optimizer_updates=0,head_fit_updates=0,loaded_checkpoint_discarded_not_inherited=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-run',required=True);p.add_argument('--source-release',required=True);p.add_argument('--profile',required=True);a=p.parse_args();print(json.dumps(run(a)))
