"""Frozen legal-source geometry, trained kernels and actual W-independent export."""
import time
import torch
from torch.utils.data import DataLoader
from . import design as d,model as api
from .provenance import source_proof,own_model
from experiments.cvs_feature_restitution.features import layout
from experiments.cvs_feature_restitution.source_geometry import analyze

def kernel_report(model):
 core,_,_=layout(model);out={}
 for layer in ('t1','t2','t3'):
  block=getattr(core,layer)
  if not isinstance(block,api.SingleBasisBlock):out[layer]=dict(kind='native rank1 depthwise-pointwise');continue
  k=api.equivalent_kernel(block).detach().cpu();sv=torch.linalg.svdvals(k.permute(1,0,2));ranks=(sv>sv[:,0:1]*1e-10).sum(1)
  if int(ranks.max())>2:raise ValueError('Single temporal basis preGN rank exceeded2')
  out[layer]=dict(kind='single k5 temporal basis plus instantaneous projection',parameters=sum(p.numel() for p in block.parameters()),preGN_rank_per_input=ranks.tolist(),singular_values=sv.tolist(),not_joint160_rank=True)
 return out

def run(row):
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source=source_proof(c);out=d.BASE/row/'diagnostics'
 if out.exists():raise FileExistsError(out)
 torch.set_num_threads(2);started=time.perf_counter()
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_phase1_stack.fast_execution import configure
 from experiments.cvs_equivariant_identity.precision import numerical_context
 with numerical_context(d.FULL_FP32_POLICY):
  model=own_model(c,'cuda:0');before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};zz=[];yy=[];rr=[];checked=False
  with installed(c) as n:
   from cvsrffi.xuc_fusion.native import role_ids_from_native
   ctx=n._build_ssdg_wisig_data(configure(d.make_args(c)),torch.device('cuda:0'))
   if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=d.read(d.SOURCE)['role_ids']:raise ValueError('Exact legal source roles required')
   native=ctx['train_loader'];loader=DataLoader(native.dataset,batch_size=256,shuffle=False,drop_last=False,num_workers=0,collate_fn=native.collate_fn,generator=torch.Generator().manual_seed(812))
   if len(loader.dataset)!=6300:raise ValueError('L source count differs')
   with torch.no_grad():
    for x,y,domain,meta in loader:
     x=x.to('cuda:0');v=model.forward_identity_only(x);z=v['id_feat_joint']
     if not checked:
      if not torch.equal(model.id_backbone.encoder.features(x),z):raise ValueError('Actual feature export differs')
      h=api.classifier(model);w=h.weight.detach().clone();h.weight.zero_()
      if not torch.equal(model.id_backbone.encoder.features(x),z):raise ValueError('W modification changed z')
      del h.weight
      if not torch.equal(model.id_backbone.encoder.features(x),z):raise ValueError('W deletion changed z')
      h.weight=torch.nn.Parameter(w,requires_grad=False);checked=True
     zz.append(z.cpu());yy.append(y.cpu());rr.append(meta['rx_i'].cpu())
  if sum(len(z) for z in zz)!=6300 or any(not torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items()) or any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('Frozen source diagnostic state/count differs')
  core,prefix,_=layout(model);initial=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'];updates={}
  for layer in ('t1','t2','t3'):
   entries=[(n,p.detach().cpu()) for n,p in model.named_parameters() if n.startswith(prefix+'.'+layer+'.')]
   updates[layer]={n:dict(changed_values=int((p!=initial[n]).sum()),l2_change=float((p-initial[n]).norm())) for n,p in entries}
  result=dict(status='SOURCE_DEFAULT_DIAGNOSTICS_COMPLETE',config=c,source_roles=['L_s'],fit_roles=[],head_fit_updates=0,backbone_updates=0,model_params_buffers_unchanged=True,target_access=False,truth_read=False,kernels=kernel_report(model),branch_removal={},parameter_updates=updates,trained_export=dict(original_feature_only_equal=True,W_modified_deleted_invariant=True),geometry=analyze(torch.cat(zz),torch.cat(yy),torch.cat(rr),role='L_s'),sample_feature_cache_written=False,wall_seconds=time.perf_counter()-started,interpretation='Parameter-matched different function families; kernel rank is not feature rank or transfer evidence')
  out.mkdir(parents=True,exist_ok=False);d.write(out/'summary.json',result)
 print('SOURCE_DEFAULT_DIAGNOSTICS_COMPLETE '+row,flush=True);return result
