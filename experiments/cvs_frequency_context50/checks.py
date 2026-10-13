import copy,json,torch
from . import design as d
from .model import FrequencyContext,build

def checks():
 torch.set_num_threads(2);torch.manual_seed(715)
 models={a:build(d.config(next(r for r in d.rows() if r['arm']==a and r['model_seed']==d.SEEDS[0]))).eval() for a in d.ARMS}
 info={};x=torch.randn(6,2,256);y=torch.arange(6)
 for a,m in models.items():
  b=m.id_backbone.encoder.core.id_backbone;z=m.forward_identity_only(x)['tx_logits'];assert z.shape==(6,6) and torch.isfinite(z).all()
  alone=torch.cat([m.forward_identity_only(v[None])['tx_logits'] for v in x]);assert torch.allclose(z,alone,atol=2e-5,rtol=2e-5)
  loss=torch.nn.functional.cross_entropy(z,y);loss.backward();extra=[p for n,p in m.id_backbone.named_parameters() if 'qkv' in n]
  if a!='LEGACY50':assert extra and all(p.grad is not None and p.grad.norm()>0 for p in extra)
  info[a]=dict(identity_parameters=sum(p.numel() for p in m.id_backbone.parameters()),qkv_gradients=[float(p.grad.norm()) for p in extra],head_type=type(b.cls_head).__name__,head_shape=list(b.cls_head.weight.shape),source_U=True,EMA=True)
 local,global_=(models[a].id_backbone.encoder.core.id_backbone for a in ('LOCAL_FREQ','GLOBAL_FREQ'))
 assert info['LOCAL_FREQ']['identity_parameters']==info['GLOBAL_FREQ']['identity_parameters']
 for a in ('LOCAL_FREQ','GLOBAL_FREQ'):
  b=models[a].id_backbone.encoder.core.id_backbone;base=models['LEGACY50'].id_backbone.encoder.core.id_backbone
  for name in ('t1','t2','t3','pa_b1','pa_b2','pa_b3','freq_gate','pa_gate','cls_head'):
   left=getattr(b,name).state_dict();right=getattr(base,name).state_dict();assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
  for name in ('f1','f2'):
   assert all(torch.equal(v,getattr(base,name).state_dict()[k]) for k,v in getattr(b,name)[0].state_dict().items())
 assert all(torch.equal(v,global_.state_dict()[k]) for k,v in local.state_dict().items())
 # Attention location test: distant bins can affect a global output; the local mask excludes them.
 for mode in ('local','global'):
  torch.manual_seed(67);block=FrequencyContext(16,mode).double();v=torch.randn(2,16,16,dtype=torch.float64,requires_grad=True);out=block(v);g=torch.autograd.grad(out[0,:,0].square().sum(),v)[0]
  assert g[1].abs().max()==0
  if mode=='local':assert g[0,:,2:].abs().max()==0
  else:assert g[0,:,15].norm()>0
 # New CPU module initialization must preserve both CPU and CUDA random streams.
 from .model import install
 from experiments.cvs_phase1_stack.runtime import installed as native_installed
 c=d.config(next(r for r in d.rows() if r['arm']=='GLOBAL_FREQ' and r['model_seed']==d.SEEDS[0]));a=d.make_args(c,'cpu');a.num_domains=5;a.input_len=256
 with native_installed(c) as native:raw=native.build_baseline_model(a,torch.device('cpu'))
 torch.manual_seed(1707);cpu_before=torch.random.get_rng_state();cuda_before=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
 install(raw,c)
 assert torch.equal(cpu_before,torch.random.get_rng_state())
 if cuda_before:assert all(torch.equal(x,y) for x,y in zip(cuda_before,torch.cuda.get_rng_state_all()))
 # Native versioned EMA must update the added feature parameters and invalidate eval caches.
 from experiments.cvs_phase1_overlay.model import native_modules
 teacher=copy.deepcopy(models['GLOBAL_FREQ']).eval().requires_grad_(False);student=models['GLOBAL_FREQ'];n=native_modules()
 name=next(k for k in dict(student.named_parameters()) if 'qkv.weight' in k);tp=dict(teacher.named_parameters())[name];before=tp.detach().clone();version=tp._version
 with torch.no_grad():dict(student.named_parameters())[name].add_(.1)
 n._update_ema_model(teacher,student,.999,versioned=True)
 assert tp._version>version and torch.allclose(tp,before+.0001,atol=1e-7,rtol=1e-7)
 # Source-only recipe is the user-selected historical pseudo/EMA recipe, not CE-only.
 for row in d.rows():
  c=d.config(row);a=d.make_args(c,'cpu');assert a.epochs==50 and a.label_epochs==32 and a.pseudo_epochs==18
  assert a.use_ema_teacher and a.use_unlabeled and a.pseudo_temporal_mode=='batch_neighbor' and a.pseudo_temporal_gate and a.pseudo_strong_agreement
  assert a.label_smoothing==.01 and a.phase1_lr_schedule=='cosine' and not a.use_concat_sat_channel_aug
 negative=0
 for key,value in [('epochs',51),('label_epochs',31),('features',[]),('explicit_CFO_identity_input',True),('legacy_recipe',dict(d.LEGACY_FIELDS,use_ema_teacher=False))]:
  c=copy.deepcopy(d.config(d.rows()[0]));c[key]=value
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Changed contract accepted: '+key)
 return dict(status='PASS',models=info,negative_cases=negative,batch_independent=True,local_global_dependency_verified=True,legacy_EMA_versioned_added_params=True,extra_initialization_preserves_CPU_CUDA_RNG=True,target_or_source_IQ_read=False)
if __name__=='__main__':print(json.dumps(checks()))
