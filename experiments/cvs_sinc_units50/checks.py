import ast,copy,json
import torch
from torch.nn import functional as F
from .optimizer import build,native_scope,NAMES
from experiments.cvs_temporal_rsc50 import model as api,design as d
from experiments.cvs_temporal_rsc50.checks import invoke,native_call
from experiments.cvs_phase1_overlay.model import native_modules

def normalized_coordinate_check(model):
 model=copy.deepcopy(model).double();named=dict(model.named_parameters());ps=[named[k] for k in NAMES];fs=25000000.
 phi=[torch.nn.Parameter(p.detach().clone()/fs) for p in ps]
 opt=build(model,[p for p in model.parameters() if p.requires_grad],'NORMALIZED_UNITS',lr=.0002,weight_decay=.0001,foreach=False)
 ref=torch.optim.AdamW(phi,lr=.0002,weight_decay=.0001,foreach=False)
 gen=torch.Generator().manual_seed(9504)
 max_error=0.
 for step in range(4):
  lr=.0002/(step+1)
  for group in opt.param_groups:group['lr']=lr
  ref.param_groups[0]['lr']=lr
  for p,q in zip(ps,phi):
   g=torch.randn(p.shape,dtype=torch.float64,generator=gen)*1e-8;p.grad=g;q.grad=g*fs
  opt.step();ref.step()
  max_error=max(max_error,max(float((p-q*fs).detach().abs().max()) for p,q in zip(ps,phi)))
 assert max_error<3e-8
 return dict(maximum_absolute_coordinate_error_Hz=max_error,steps=4,preclip_gradients_supplied=True,global_clip_equivalence_claim=False)

def checks():
 torch.set_num_threads(2);c=d.config(next(r for r in d.training_rows() if r['arm']=='SIG_TIME'));a=d.make_args(c,'cpu');original=api.build(c,'cpu').train();standard=copy.deepcopy(original);unit=copy.deepcopy(original)
 native=native_modules();params=lambda m:native._optimizer_parameters(m,None)
 plain_opt=torch.optim.AdamW(params(original),lr=.0002,weight_decay=.0001)
 from types import SimpleNamespace
 owner=SimpleNamespace(model=standard,optimizer=None)
 old_class=torch.optim.AdamW
 with native_scope(owner,'STANDARD'):standard_opt=torch.optim.AdamW(params(standard),lr=.0002,weight_decay=.0001)
 assert torch.optim.AdamW is old_class and standard_opt is owner.optimizer
 unit_opt=build(unit,params(unit),'NORMALIZED_UNITS',lr=.0002,weight_decay=.0001)
 x=torch.randn(12,2,256,generator=torch.Generator().manual_seed(9501));y=torch.arange(12)%6;rx=torch.tensor([1,3,4,6,8,1]*2)
 rng=torch.get_rng_state().clone();outputs=[];before_sinc={k:dict(unit.named_parameters())[k].detach().clone() for k in NAMES}
 for model in (original,standard,unit):
  torch.set_rng_state(rng);out=invoke(model,a,x,y,rx);outputs.append(out['tx_logits'].detach());loss=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);loss.backward();torch.nn.utils.clip_grad_norm_(params(model),5.)
 assert all(torch.equal(outputs[0],q) for q in outputs[1:])
 for key,p in original.named_parameters():
  for m in (standard,unit):
   q=dict(m.named_parameters())[key]
   assert (p.grad is None)==(q.grad is None) and (p.grad is None or torch.equal(p.grad,q.grad))
 plain_opt.step();standard_opt.step();unit_opt.step()
 assert all(torch.equal(v,standard.state_dict()[k]) for k,v in original.state_dict().items())
 for s,t in zip(plain_opt.state.values(),standard_opt.state.values()):assert set(s)==set(t) and all(torch.equal(v,t[k]) if torch.is_tensor(v) else v==t[k] for k,v in s.items())
 for key,v in original.state_dict().items():
  if key not in NAMES:assert torch.equal(v,unit.state_dict()[key])
 differences={k:dict(changed_elements=int((dict(unit.named_parameters())[k].detach()!=before_sinc[k]).sum()),maximum_abs_Hz=float((dict(unit.named_parameters())[k].detach()-before_sinc[k]).abs().max())) for k in NAMES}
 assert all(r['changed_elements']>1 for r in differences.values())
 original.eval();unit.eval()
 with torch.no_grad():z0=original.forward_identity_only(x)['id_feat_joint'];z1=unit.forward_identity_only(x)['id_feat_joint']
 dz=float((z0-z1).abs().max());assert dz>1e-8
 _,_,tree=native_call();ema=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_update_ema_model');namespace={'torch':torch};exec(compile(ast.Module(body=[ema],type_ignores=[]),'<actual-native-EMA>','exec'),namespace)
 teacher0=copy.deepcopy(standard);teacher1=copy.deepcopy(standard)
 namespace['_update_ema_model'](teacher0,original,.999);namespace['_update_ema_model'](teacher1,unit,.999)
 assert any(not torch.equal(teacher0.state_dict()[k],teacher1.state_dict()[k]) for k in NAMES)
 for key,v in teacher0.state_dict().items():
  if key not in NAMES:assert torch.equal(v,teacher1.state_dict()[key])
 normalized=normalized_coordinate_check(original)
 flat=params(original);negative=[lambda:build(original,flat,'UNKNOWN'),lambda:build(original,flat[:-1],'NORMALIZED_UNITS'),lambda:build(original,flat+flat[:1],'NORMALIZED_UNITS'),lambda:build(original,flat,'NORMALIZED_UNITS',lr=0),lambda:build(original,flat,'NORMALIZED_UNITS',eps=0),lambda:build(original,flat,'NORMALIZED_UNITS',weight_decay=-1),lambda:build(original,flat,'NORMALIZED_UNITS',amsgrad=True)]
 for bad in negative:
  try:bad()
  except (ValueError,RuntimeError):pass
  else:raise AssertionError('Invalid native optimizer recipe accepted')
 return dict(status='PASS',negative_cases=len(negative),actual_native_L_forward=True,actual_native_parameter_collection=True,same_initial_logits_and_clipped_gradients=True,STANDARD_native_AdamW_and_moments_exact=True,nonsinc_including_default_W_first_update_exact=True,actual_joint160_feature_difference_max=dz,synthetic_first_step_Sinc=differences,original_ownEMA_path_tracks_Sinc=True,normalized_coordinates=normalized,native_cosine_schedule_overwrite_handled=True,model_architecture_or_parameterization_changed=False,real_IQ_or_checkpoint_read=False,formal_training_started=False,head_fit_updates=0,scope='Synthetic local one-step mechanics; not evidence of real trainedfilter adaptation, source/target accuracy or DG gain')
if __name__=='__main__':print(json.dumps(checks()))
