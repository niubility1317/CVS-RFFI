"""Actual frozen source f_theta derivatives. No backward accumulation or optimizer."""
import copy,json,math,time
from pathlib import Path
import torch
from types import SimpleNamespace
from torch.nn import functional as F
from . import design as d
from experiments.cvs_neighbor_pair50.predict import source_proof,own_model
from experiments.cvs_source_gate_audit.gates import physical_metadata,masks
from experiments.cvs_source_neighbor_pairs.order import order as make_order
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_equivariant_identity.precision import numerical_context
P='id_backbone.encoder.core.id_backbone.'
W=P+'cls_head.weight'
LAYERS=('time_fuse','t1','t2','t3','f1','f2','f3','pa_b1','pa_b2','pa_b3')

def group(name):
 if not name.startswith('id_backbone.') or name==W:raise ValueError('Gradient object must be actual feature parameter, not class W or auxiliary network')
 if not name.startswith(P):return 'reference'
 name=name[len(P):]
 for key,prefixes in [('Sinc_Hz',('sinc.',)),('time_shallow',('time_fuse.','t1.')),('time_middle',('t2.','t3.')),('time_projection',('t_proj.',)),('frequency',('f1.','f2.','f3.','f_proj.','freq_gate.','freq_stats_proj.')),('PA_shallow',('pa_gate.','pa_b1.','pa_stats_proj.')),('PA_middle',('pa_b2.','pa_b3.')),('PA_projection',('pa_proj.',)),('fusion',('fuse.',)),('feature_norm_gain',('cls_head.',))]:
  if name.startswith(prefixes):return key
 raise ValueError('Unregistered actual feature parameter group: '+name)

def objects(model):
 if model.training or any(m.training for m in model.id_backbone.modules()):raise ValueError('Source gradient observer requires eval')
 head=model.id_backbone.encoder.core.id_backbone.cls_head
 if type(head).__name__!='ResidualPhysicalHead' or head.weight.shape!=(6,160) or head.scale!=30:raise ValueError('Default cosine classifier differs')
 params=[(n,p) for n,p in model.named_parameters() if n.startswith('id_backbone.') and n!=W]
 if sum(p.numel() for n,p in params)!=176065:raise ValueError('Actual f_theta boundary differs')
 groups={}
 for i,(n,p) in enumerate(params):groups.setdefault(group(n),[]).append(i)
 return params,groups

def capture(model,x):
 backbone=model.id_backbone.encoder.core.id_backbone;values={};handles=[]
 def hook(name):
  def store(module,args,output):
   if name in values or not isinstance(output,torch.Tensor) or not output.requires_grad or output.shape[0]!=len(x):raise ValueError('Actual feature activation call/shape differs')
   values[name]=output
  return store
 try:
  for name in LAYERS:handles.append(getattr(backbone,name).register_forward_hook(hook(name)))
  out=model.forward_identity_only(x);values['joint160']=out['id_feat_joint']
  if set(values)!=set(LAYERS)|{'joint160'} or values['joint160'].shape!=(len(x),160):raise ValueError('Actual deployed feature was not traced')
  if any(not torch.isfinite(a).all() for a in values.values()):raise ValueError('Nonfinite actual feature activation')
  return out,values
 finally:
  for h in handles:h.remove()

def derivatives(loss,params,activations,retain=False):
 if not torch.isfinite(loss):raise ValueError('Nonfinite source diagnostic objective')
 names=list(activations);grads=torch.autograd.grad(loss,[p for n,p in params]+[activations[n] for n in names],retain_graph=retain,allow_unused=True)
 pg=[torch.zeros_like(p) if g is None else g.detach() for (n,p),g in zip(params,grads[:len(params)])]
 if any(g is None for g in grads[len(params):]):raise ValueError('Traced activation is disconnected from the default objective')
 ag={n:g.detach() for n,g in zip(names,grads[len(params):])}
 if any(not torch.isfinite(g).all() for g in [*pg,*ag.values()]):raise ValueError('Nonfinite actual source derivative')
 return pg,ag

def parameter_record(groups,l,u,e):
 records=[]
 for name,indices in groups.items():
  a=torch.cat([l[i].double().flatten() for i in indices]);b=torch.cat([u[i].double().flatten() for i in indices]);q=torch.cat([e[i].double().flatten() for i in indices]);s=b+q
  na=float(a.norm());nb=float(b.norm());ne=float(q.norm());ns=float(s.norm())
  records.append(dict(path=name,L_CE_gradient_L2=na,weighted_pseudo_CE_gradient_L2=nb,weighted_entropy_gradient_L2=ne,combined_U_gradient_L2=ns,combined_U_over_L=ns/na if na else None,L_U_cosine=float(torch.dot(a,s))/(na*ns) if na and ns else None,elements=a.numel()))
 return records

def activation_record(clean,strong,lg,ug,eg,rx,mask):
 result=[];n=len(rx);accepted=int(mask.sum())
 for name in clean:
  a=lg[name].double().reshape(n,-1).norm(dim=1)*n
  b=ug[name].double().reshape(n,-1).norm(dim=1)*accepted
  q=eg[name].double().reshape(n,-1).norm(dim=1)*n
  for receiver in (1,3,4,6,8):
   keep=rx==receiver;sel=keep&mask
   if keep.any():result.append(dict(path=name,receiver=receiver,count=int(keep.sum()),accepted=int(sel.sum()),per_packet_L_gradient_L2_sum=float(a[keep].sum()),per_accepted_weighted_pseudo_gradient_L2_sum=float(b[sel].sum()),per_packet_weighted_entropy_gradient_L2_sum=float(q[keep].sum()),scope='Mean-reduction multiplier removed: L and entropy times batch N, pseudo times selected N. Per-example eval sensitivity, not native train gradient or cross-layer importance.'))
 return result

def actual_arguments(root):
 values=d.read(root/'resolved_native_args.json');values=values.get('args',values);args=SimpleNamespace(**values)
 if args.pseudo_domain_gate or not args.pseudo_temporal_gate or not args.pseudo_strong_agreement or args.pseudo_temporal_mode!='batch_neighbor' or args.pseudo_temporal_window!=2 or args.lambda_u!=.16 or args.lambda_ent!=.01 or args.strong_noise_std!=.015 or args.use_mixstyle:raise ValueError('Actual old source gate/identity loss differs')
 return args

def measure(c,L,meta):
 d.validate(c)
 if len(L['ids'])!=6300 or set(meta)!={'base_index','rx_i','day_i','eq_i','sig_i'}:raise ValueError('Only approved full labeled source and allowed metadata')
 root=source_proof(c['source_config']);student=own_model(c['source_config'],'cpu').eval().requires_grad_(True);teacher=copy.deepcopy(student)
 ck=torch.load(root/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100 or ck['ema_model'] is None:raise ValueError('Actual ownEMA fixed final origin differs')
 teacher.load_state_dict(ck['ema_model'],strict=True);teacher.eval().requires_grad_(False);del ck
 saved={k:v.clone() for k,v in student.state_dict().items()};saved_t={k:v.clone() for k,v in teacher.state_dict().items()};params,groups=objects(student)
 args=actual_arguments(root);native=native_modules()
 order=make_order(meta,c['shuffle_seed'],'PAIRS')
 if not torch.equal(order.sort().values,torch.arange(6300)):raise ValueError('L source pool not exactly once')
 batches=[];start=time.time()
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(c['strong_seed'])
  for begin in range(0,6300,256):
   ix=order[begin:begin+256];x=L['x'][ix];bm={k:v[ix] for k,v in meta.items()};rx=L['rx'][ix];domain=(torch.searchsorted(torch.tensor([1,3,4,6,8]),rx)*3+L['day'][ix]-1).long()
   with torch.no_grad():weak=teacher.forward_identity_only(x)['tx_logits']
   out_l,clean=capture(student,x);out_s,strong=capture(student,native._strong_augment(x,.015))
   with torch.no_grad():pseudo,selected,detail=masks(native,weak,out_s['tx_logits'].detach(),bm,domain,args);mask=selected['FULL_OLD']
   # Gates are fixed before the allowed L labels enter supervised objective/observer.
   y=L['y'][ix];LL=F.cross_entropy(out_l['tx_logits'],y,label_smoothing=.01)
   UU=.16*(F.cross_entropy(out_s['tx_logits'][mask],pseudo[mask]) if mask.any() else out_s['tx_logits'].sum()*0.)
   prob=out_s['tx_logits'].softmax(1);EE=.01*(-prob*prob.clamp_min(1e-8).log()).sum(1).mean()
   lp,la=derivatives(LL,params,clean);up,ua=derivatives(UU,params,strong,retain=True);ep,ea=derivatives(EE,params,strong)
   batches.append(dict(batch=begin//256+1,count=len(ix),accepted=int(mask.sum()),accepted_correct=int((mask&(pseudo==y)).sum()),teacher_correct=int((pseudo==y).sum()),L_CE=float(LL.detach()),weighted_pseudo_CE=float(UU.detach()),weighted_entropy=float(EE.detach()),geometry_all_visible=bool(selected['GEOMETRIC_NEIGHBOR'].all()),parameter_paths=parameter_record(groups,lp,up,ep),activation_paths=activation_record(clean,strong,la,ua,ea,rx,mask)))
 if any(not torch.equal(v,student.state_dict()[k]) for k,v in saved.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in saved_t.items()):raise ValueError('Source derivative observer changed parameter/buffer')
 if any(p.grad is not None for m in (student,teacher) for p in m.parameters()):raise ValueError('Unexpected accumulated gradients')
 if len(batches)!=25 or sum(b['count'] for b in batches)!=6300 or any(not b['geometry_all_visible'] for b in batches):raise ValueError('Actual source batch/neighbor coverage differs')
 return dict(status='VERIFIED',run_id=d.RUN,row_id=c['row_id'],arm=c['arm'],seed=c['model_seed'],source_roles='EXACT_MATCH',source_role='L_s',source_packets=6300,source_batches=25,student_and_ownEMA=True,source_checkpoint_budget=dict(epochs=50,updates=11100),parameters_buffers_unchanged=True,accumulated_gradients=False,feature_parameters=176065,class_W_excluded_elements=960,head_fit=False,optimizer_steps=0,backbone_training_epochs=0,U_IQ_read=False,U_truth_read=False,V_forward=False,V_gradients=False,V_cache_materialized=True,query_read=False,source_observer_labels_after_masks=True,mode='eval; no dropout; diagnostic derivatives, not actual recorded train gradients',batches=batches,packet_forwards=18900,seconds=time.time()-start,interpretation='Same feature path source L/U-proxy gradient norm ratios/cosines only. Sinc Hz separately grouped; do not compare raw units across paths, infer U accuracy/newTX transfer, or pick target-driven lambda.')

def main():
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 torch.set_num_threads(2)
 if d.BASE.exists():raise FileExistsError('Existing immutable source derivative root')
 L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None
 # Existing validated source loader returns V; V is never accessed by derivative/forward code.
 del V
 meta=physical_metadata(L,'L_s');d.BASE.mkdir(parents=True,exist_ok=False)
 try:
  with numerical_context(d.FULL_FP32_POLICY):
   for row in d.rows():
    c=d.config(row);p=Path(c['output_root']);p.mkdir(exist_ok=False);d.write(p/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip()))
    result=measure(c,L,meta);d.write(p/'results.json',result);d.write(p/'completion.json',dict(status='ANALYZED',source_only=True,source_packets=6300,head_updates=0,optimizer_steps=0,query_read=False,U_truth_read=False));print(json.dumps(dict(row_id=row['row_id'],status='ANALYZED',seconds=result['seconds'])),flush=True)
  d.write(d.BASE/'completion.json',dict(status='ANALYZED',models=3,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0))
 except Exception as e:
  d.write(d.BASE/'failure.json',dict(status='FAILED',exception_type=type(e).__name__,message=str(e)));raise

if __name__=='__main__':main()
