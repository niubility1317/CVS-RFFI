"""Read-only source geometry and local-order checks; zero fitted heads."""
import time
import torch
from torch.utils.data import DataLoader
from . import design as d
from .provenance import source_proof,own_model
from experiments.cvs_feature_restitution.source_geometry import analyze

def run(row):
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source_proof(c);out=d.BASE/row/'diagnostics'
 if out.exists():raise FileExistsError('Existing source diagnostics; preserve')
 start=time.perf_counter();torch.set_num_threads(2)
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_phase1_stack.fast_execution import configure
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from cvsrffi.xuc_fusion.native import role_ids_from_native
 with numerical_context(d.FULL_FP32_POLICY):
  model=own_model(c,'cuda:0');a=configure(d.make_args(c));expected=d.read(d.SOURCE)
  pool=next((m for m in model.modules() if type(m).__name__=='SequencePool'),None)
  values={};order_total=0;order_changed=0;order_max=0.;order_hits=0
  with installed(c) as n:
   ctx=n._build_ssdg_wisig_data(a,torch.device('cuda:0'))
   if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids']:raise ValueError('Exact source roles/no target loaders required')
   for role,key,count in [('L_s','train_loader',6300),('V','val_loader',27000)]:
    loader=ctx[key];loader=DataLoader(loader.dataset,batch_size=256,shuffle=False,drop_last=False,num_workers=0,collate_fn=loader.collate_fn)
    if len(loader.dataset)!=count:raise ValueError('Source diagnostic count differs')
    zz=[];yy=[];rr=[]
    with torch.no_grad():
     for x,y,domain,meta in loader:
      x=x.to('cuda:0');v=model.forward_identity_only(x);zz.append(v['id_feat_joint'].cpu());yy.append(y.cpu());rr.append(meta['rx_i'].cpu())
      if role=='V' and pool is not None:
       def shuffle(module,args):
        h=args[0];b,ch,t=h.shape;length=t//4
        order=torch.randperm(length,generator=torch.Generator().manual_seed(2026101329)).to(h.device)
        return (h.reshape(b,ch,4,length)[:,:,:,order].reshape(b,ch,t),)
       handle=pool.register_forward_pre_hook(shuffle)
       try:alternative=model.forward_identity_only(x)
       finally:handle.remove()
       diff=(v['id_feat_joint']-alternative['id_feat_joint']).abs().max(1).values
       order_max=max(order_max,float(diff.max()));order_total+=len(x)
       order_changed+=int((v['tx_logits'].argmax(1)!=alternative['tx_logits'].argmax(1)).sum())
       order_hits+=int((alternative['tx_logits'].argmax(1).cpu()==y).sum())
    values[role]=dict(z=torch.cat(zz),y=torch.cat(yy),rx=torch.cat(rr))
  L=values['L_s'];geometry=analyze(L['z'],L['y'],L['rx'],role='L_s')
  if any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('Source diagnostic updated model')
  if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):raise ValueError('Temporary local-order hook retained')
  if pool is not None and c['arm'] in ('MEAN','MEAN_VAR','NONLINEAR_MEAN','RFF','COV') and order_max>1e-4:raise ValueError('Order-independent descriptor changed under within-window permutation')
 result=dict(status='SOURCE_DEFAULT_DIAGNOSTICS_COMPLETE',config=c,source_roles=['L_s','V'],fit_roles=[],head_fit_updates=0,backbone_updates=0,target_access=False,truth_read=False,geometry=geometry,order_diagnostic=dict(scope='within each fixed window; only new middle-feature branch shuffled; actual IQ and original paths unchanged',count=order_total,max_embedding_change=order_max,changed_default_head_decisions=order_changed,permuted_default_head_accuracy=order_hits/order_total if order_total else None),sample_feature_cache_written=False,wall_seconds=time.perf_counter()-start)
 out.mkdir(parents=True,exist_ok=False);d.write(out/'summary.json',result);print('SOURCE_DEFAULT_DIAGNOSTICS_COMPLETE '+row,flush=True);return result
