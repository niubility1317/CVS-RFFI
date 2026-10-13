"""Actual kernels and frozen source branch removal, never fit a classifier."""
from contextlib import contextmanager
import time
import torch
from torch.utils.data import DataLoader
from . import design as d
from .provenance import source_proof,own_model
from experiments.cvs_feature_restitution.features import layout
from experiments.cvs_feature_restitution.source_geometry import analyze
def equivalent_kernel(block):
 if getattr(block,'nonlinear',False):raise ValueError('Nonlinear branches cannot be folded')
 ci=block.branches[0].in_channels;co=block.fuse.out_channels;settings=getattr(block,'settings',getattr(block,'_settings',None));radius=max((k-1)//2*r for k,r in settings)
 kernel=block.fuse.weight.new_zeros(co,ci,2*radius+1,dtype=torch.float64)
 for j,(branch,(k,rate)) in enumerate(zip(block.branches,settings)):
  p=block.fuse.weight[:,j*ci:(j+1)*ci,0].double();w=branch.weight[:,0].double()
  for t in range(k):kernel[:,:,radius+(t-(k-1)//2)*rate]+=p*w[:,t][None,:]
 kernel[:,:,radius]+=block.skip.weight[:,:,0].double();return kernel

def kernel_report(model):
 core,_,_=layout(model);out={}
 for layer in ('t1','t2','t3'):
  block=getattr(core,layer)
  if not hasattr(block,'branches'):out[layer]=dict(kind='native depthwise+pointwise',parameters=sum(p.numel() for p in block.parameters()));continue
  if getattr(block,'nonlinear',False):out[layer]=dict(kind='nonlinear uncollapsible',parameters=sum(p.numel() for p in block.parameters()));continue
  k=equivalent_kernel(block).detach();r=k.shape[-1]//2;energy=k.square().sum((0,1));sv=torch.linalg.svdvals(k.permute(1,0,2));prob=sv/(sv.sum(1,keepdim=True)+1e-30);erank=(-(prob*prob.clamp_min(1e-30).log()).sum(1)).exp()
  out[layer]=dict(kind='linear equivalent preGN kernel',parameters=sum(p.numel() for p in block.parameters()),offsets=list(range(-r,r+1)),energy=energy.tolist(),far_energy_fraction=float(energy[torch.arange(-r,r+1).abs()>2].sum()/energy.sum()),numerical_rank_per_input=(sv>sv[:,0:1]*1e-10).sum(1).tolist(),effective_rank_per_input=erank.tolist(),singular_values=sv.tolist(),not_joint160_rank=True)
 return out
@contextmanager
def remove(block,j):
 handle=block.branches[j].register_forward_hook(lambda m,a,o:o*0)
 try:yield
 finally:handle.remove()
def measure_update(model,before):
 for n,v in model.state_dict().items():
  if not torch.equal(v.detach().cpu(),before[n]):raise ValueError('Frozen source diagnostic mutated model: '+n)
def run(row,config=None,output=None,proof=source_proof,load=own_model,args_factory=d.make_args):
 c=config or d.config(next(r for r in d.rows() if r['row_id']==row));proof(c);out=output or d.BASE/row/'diagnostics'
 if out.exists():raise FileExistsError(out)
 torch.set_num_threads(2);start=time.perf_counter()
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_phase1_stack.fast_execution import configure
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from cvsrffi.xuc_fusion.native import role_ids_from_native
 with numerical_context(d.FULL_FP32_POLICY):
  model=load(c,'cuda:0');before={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};core,_,_=layout(model);removals=[(l,j) for l in ('t1','t2','t3') for j in range(3)] if c['arm']!='BASE_REF' else []
  aggregates={f'{l}/branch{j}':dict(count=0,corrected_by_full=0,introduced_by_full=0,changed=0,margin_delta_sum=0.,embedding_delta_sum=0.) for l,j in removals};zz=[];yy=[];rr=[]
  with installed(c) as n:
   ctx=n._build_ssdg_wisig_data(configure(args_factory(c)),torch.device('cuda:0'))
   if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=d.read(d.SOURCE)['role_ids']:raise ValueError('Exact legal source roles required')
   for role,key,count in [('L_s','train_loader',6300),('V','val_loader',27000)]:
    native=ctx[key];loader=DataLoader(native.dataset,batch_size=256,shuffle=False,drop_last=False,num_workers=0,collate_fn=native.collate_fn,generator=torch.Generator().manual_seed(812))
    if len(loader.dataset)!=count:raise ValueError('Source shape differs')
    with torch.no_grad():
     for x,y,domain,meta in loader:
      x=x.to('cuda:0');y=y.to('cuda:0');v=model.forward_identity_only(x);logits=v['tx_logits'];correct=logits.argmax(1)==y;other=logits.clone();other.scatter_(1,y[:,None],-torch.inf);margin=logits.gather(1,y[:,None])[:,0]-other.max(1).values
      if role=='L_s':zz.append(v['id_feat_joint'].cpu());yy.append(y.cpu());rr.append(meta['rx_i'].cpu())
      if role=='V':
       for layer,j in removals:
        with remove(getattr(core,layer),j):alt=model.forward_identity_only(x)
        al=alt['tx_logits'];ac=al.argmax(1)==y;ao=al.clone();ao.scatter_(1,y[:,None],-torch.inf);am=al.gather(1,y[:,None])[:,0]-ao.max(1).values;g=aggregates[f'{layer}/branch{j}']
        g['count']+=len(x);g['corrected_by_full']+=int((correct&~ac).sum());g['introduced_by_full']+=int((~correct&ac).sum());g['changed']+=int((logits.argmax(1)!=al.argmax(1)).sum());g['margin_delta_sum']+=float((margin-am).sum());g['embedding_delta_sum']+=float((v['id_feat_joint']-alt['id_feat_joint']).norm(dim=1).sum())
  measure_update(model,before)
  if any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('Diagnostic enabled optimization')
  for g in aggregates.values():g['mean_margin_delta']=g['margin_delta_sum']/g['count'];g['mean_embedding_delta']=g['embedding_delta_sum']/g['count']
  result=dict(status='SOURCE_DEFAULT_DIAGNOSTICS_COMPLETE',config=c,source_roles=['L_s','V'],fit_roles=[],head_fit_updates=0,backbone_updates=0,model_params_buffers_unchanged=True,target_access=False,truth_read=False,kernels=kernel_report(model),branch_removal=aggregates,geometry=analyze(torch.cat(zz),torch.cat(yy),torch.cat(rr),role='L_s'),interpretation='Frozen branch removal can disrupt coadaptation; not scratch ablation or TX causal proof; preGN equivalent-kernel rank is not joint160 rank',sample_feature_cache_written=False,wall_seconds=time.perf_counter()-start)
  out.mkdir(parents=True,exist_ok=False);d.write(out/'summary.json',result);print('SOURCE_DEFAULT_DIAGNOSTICS_COMPLETE '+row,flush=True)
  if c.get('baseline_reuse'):historical(c['model_seed'])
  return result

def historical(seed):
 from experiments.cvs_feature_multiscale50 import design as old
 from experiments.cvs_feature_multiscale50.predict import source_proof as proof,own_model as load
 c=old.config(next(r for r in old.rows() if r['arm']=='TIME_MS50' and r['model_seed']==seed))
 return run(c['row_id'],config=c,output=d.BASE/'historical_time_diagnostics'/c['row_id'],proof=proof,load=load,args_factory=old.make_args)
