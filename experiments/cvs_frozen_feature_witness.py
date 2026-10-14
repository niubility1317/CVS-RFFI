"""Read-only same-source-input functional witness for already frozen models.

This analysis neither trains nor selects a model. It does not replace the
registered default-head test or assert new-TX/Phase2 transfer performance.
"""
import torch
import torch.nn.functional as F

def analyze(model,initial,x):
 if model.training or initial.training or x.ndim!=3 or tuple(x.shape[1:])!=(2,256):raise ValueError('Frozen eval models and original source IQ required')
 if any(p.grad is not None for m in (model,initial) for p in m.parameters()):raise ValueError('Frozen analysis cannot contain training gradients')
 before=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in (model,initial)]
 from experiments.cvs_feature_multiscale50.model import classifier
 head=classifier(model);weight=head.weight
 with torch.no_grad():
  output=model.forward_identity_only(x);z=output['id_feat_joint'];z0=initial.forward_identity_only(x)['id_feat_joint'];pure=model.id_backbone.encoder.features(x)
  if z.shape!=(len(x),160) or not torch.isfinite(z).all() or not torch.isfinite(z0).all():raise ValueError('Actual deployed joint160 invalid')
  full_pure_error=float((z-pure).abs().max())
  try:
   head.weight=torch.nn.Parameter(torch.zeros_like(weight),requires_grad=False);zero=model.id_backbone.encoder.features(x)
   del head.weight;removed=model.id_backbone.encoder.features(x)
  finally:head.weight=weight
  zero_error=float((z-zero).abs().max());removed_error=float((z-removed).abs().max())
  for test in (pure,zero,removed):
   if not torch.allclose(z,test,atol=1e-5,rtol=1e-5):raise ValueError('Class W changes actual deployed source features')
  delta=(z-z0).norm(dim=1);cos=F.cosine_similarity(z,z0,dim=1,eps=1e-4)
 for m,state in zip((model,initial),before):
  if any(not torch.equal(v,m.state_dict()[k]) for k,v in state.items()) or any(p.grad is not None for p in m.parameters()):raise ValueError('Read-only analysis changed parameter/buffer/gradient')
 summary=dict(status='SOURCE_FUNCTIONAL_WITNESS_COMPLETE',source_count=len(x),forward_packets=5*len(x),source_eval_only=True,feature='actualjoint160',full_vs_pure_max_abs_error=full_pure_error,W_zero_max_abs_error=zero_error,W_removed_max_abs_error=removed_error,feature_delta_from_own_scratch_initial_mean=float(delta.mean()),feature_delta_from_own_scratch_initial_max=float(delta.max()),feature_cosine_from_own_scratch_initial_mean=float(cos.mean()),features_changed_from_own_initial=bool((delta>1e-5).any()),parameters_buffers_gradients_unchanged=True,head_fit_updates=0,optimizer_updates=0,EMA_updates=0,U_truth_read=False,target_access=False,limitation='Same90 legal balancedL inputs atowninitial andfixedfinal checkpoint; functionalchange andW-independence only, not causalauxbenefit or unseenTX transfer.')
 return summary,dict(final=z.detach().cpu(),initial=z0.detach().cpu())

def run(request):
 from pathlib import Path
 from importlib import import_module
 package=request['package']
 if package not in ('experiments.cvs_u_schedule50','experiments.cvs_pa_recomposition'):raise ValueError('Only the two registered source-only analyses')
 d=import_module(package+'.design');proof=import_module(package+'.provenance');model_api=import_module(package+'.model')
 row=next(r for r in d.rows() if r['row_id']==request['row_id']);c=d.config(row);p=proof.source_proof(c)
 out=p.parent/'source_feature_witness'
 if out.exists():raise FileExistsError('Existing source analysis; preserve/reuse rather than overwrite')
 from experiments.cvs_receiver_residual_capacity4.control import lock
 from experiments.cvs_multi_state_action.dispatch import capacity
 import time
 torch.set_num_threads(2)
 while True:
  with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
   states=capacity();ready=[(len(s['pids']),-s['free_mb'],g) for g,s in states.items() if len(s['pids'])<3 and s['free_mb']>=12000]
   if ready:
    gpu=min(ready)[2];device=torch.device(f'cuda:{gpu}');torch.zeros(1,device=device);torch.cuda.synchronize();break
  time.sleep(3)
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_phase1_stack.fast_execution import configure
 with numerical_context(d.FULL_FP32_POLICY):
  m=proof.own_model(c,device);initial=model_api.build(c,device);payload=torch.load(p/'initial_smoke.pt',map_location='cpu',weights_only=False)
  if payload.get('scratch_only') is not True:raise ValueError('Own scratch initial provenance absent')
  initial.load_state_dict(payload['model'],strict=True);initial.eval().requires_grad_(False)
  with installed(c) as n:
   args=configure(d.make_args(c,str(device)));ctx=n._build_ssdg_wisig_data(args,device)
   from cvsrffi.xuc_fusion.native import role_ids_from_native
   if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=d.read(d.SOURCE)['role_ids']:raise ValueError('Only exact original source roles allowed')
  ds=ctx['train_loader'].dataset
  if len(ds)!=6300 or getattr(ds,'hide',True):raise ValueError('Only legal visible L6300 for balance')
  cells={}
  for i in range(len(ds)):
   _,y,_,meta=ds[i];key=(int(y),int(meta['rx_i']),int(meta['day_i']));cells.setdefault(key,i)
  expected={(y,r,day) for y in range(6) for r in (1,3,4,6,8) for day in (1,2,3)}
  if set(cells)!=expected:raise ValueError('Source cell shape differs')
  samples=[ds[cells[k]] for k in sorted(cells)];x=torch.stack([s[0] for s in samples]).to(device);summary,features=analyze(m,initial,x)
  summary.update(run_id=d.RUN,row_id=row['row_id'],source_execution_commit=d.read(p/'resolved_config.json')['commit'],analysis_code_commit=request['analysis_code_commit'],role='L_s',source_current_final_epoch=50,source_initial='Own scratch initial_smoke only; no inherited training',physical_ids=[[int(s[3][k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')] for s in samples],TX_labels_used_for_balance_only=True,source_cells=90,GPU=gpu)
  out.mkdir();torch.save(features,out/'same_L90_features.pt');d.write(out/'summary.json',summary)
 return summary
