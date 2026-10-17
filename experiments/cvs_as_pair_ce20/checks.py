"""Focused equivalence/difference checks on real F with synthetic IQ only."""
import copy,json,math,random,subprocess,sys
from pathlib import Path
import torch
from torch.nn import functional as F

def run():
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
 from SSDG import train_ssdg as native
 from . import design as d,model as api
 from .objective import pair
 from experiments.cvs_scene_cycle20 import model as prior
 from experiments.cvs_crossrx_satmix20 import supcon
 from experiments.cvs_iq_timing20.objective import rd
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (ValueError,KeyError):negative+=1
  else:raise AssertionError('Forbidden contract accepted')
 c=d.config(d.rows()[0])
 for key,value in [('epochs',40),('model_seed',2026092702),('model_initialization','parent'),('classifier','other'),('test_views',['clean'])]:
  bad=copy.deepcopy(c);bad[key]=value;reject(lambda bad=bad:d.validate(bad))
 for section,key,value in [('as_pair_objective','applied_KL_weight',.5),('as_pair_objective','policy','RDROP'),('as_pair_objective','new_NN_forwards',2),('as_pair_objective','U_changed',True),('additive','extra_weight',.5),('scene_schedule','mode','balanced'),('training_mechanism','As',False),('training_mechanism','Ag',True)]:
  bad=copy.deepcopy(c);bad[section][key]=value;reject(lambda bad=bad:d.validate(bad))
 g=torch.Generator().manual_seed(202610177011);a=torch.randn(8,6,generator=g,requires_grad=True);b=torch.randn(8,6,generator=g,requires_grad=True);y=torch.arange(8)%6
 loss,ce,diag=pair(a,b,y);expected=.5*(F.cross_entropy(a,y,label_smoothing=.01)+F.cross_entropy(b,y,label_smoothing=.01))
 assert torch.equal(loss,expected) and torch.equal(ce,expected) and not diag.requires_grad and diag.grad_fn is None and float(diag)>0
 ga=torch.autograd.grad(loss,(a,b),retain_graph=True);gb=torch.autograd.grad(expected,(a,b));assert all(torch.equal(x,z) for x,z in zip(ga,gb))
 for bady in (torch.full((8,),-1,dtype=torch.long),torch.full((8,),6,dtype=torch.long),y.float()):reject(lambda bady=bady:pair(a,b,bady))
 reject(lambda:pair(a,b[:7],y));reject(lambda:pair(a*float('nan'),b,y))
 args=d.make_args(c,'cpu');oldargs=vars(d.old.make_args(d.legacy_config(),'cpu'));newargs=vars(args)
 changed={k for k,v in oldargs.items() if v!=newargs.get(k) and not(isinstance(v,float) and isinstance(newargs.get(k),float) and math.isnan(v) and math.isnan(newargs[k]))};assert changed=={'run_id','candidate_id','output_dir'},changed
 x=torch.randn(128,2,256,generator=g);saved=x.clone();labels=torch.arange(128)%6;domains=3+torch.arange(128)%12;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains];meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.zeros(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128));base=native.build_aug_base_cfg(args);augment=native.make_augmentor(base);native.configure_augmentor_for_epoch(augment,base,6,args);random.seed(202610177012);torch.manual_seed(202610177012);ordinary=augment(x.clone(),labels=labels,no_pa=False)
 def execute(m):
  m._feature_l_steps=m._joint_calls=m._legacy_calls=1110;seen=[];outputs=[];orig=m.id_backbone.encoder._view_features;original=m.forward_identity_only
  def trace(value):seen.append(value.detach().clone());return orig(value)
  def forward(value,*aa,**kw):out=original(value,*aa,**kw);outputs.append(out['tx_logits']);return out
  m.id_backbone.encoder._view_features=trace;m.forward_identity_only=forward
  torch.manual_seed(202610177013);random.seed(202610177013)
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=F.cross_entropy(out['tx_logits'],labels,label_smoothing=.01);correction,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=labels,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  m.id_backbone.encoder._view_features=orig;m.forward_identity_only=original
  return ce+correction,seen,outputs[-2:],(torch.get_rng_state().clone(),random.getstate())
 torch.manual_seed(202610177014);old=prior.build(d.legacy_config(),'cpu').train();old_init={k:v.detach().clone() for k,v in old.state_dict().items()};oldloss,oldinputs,oldz,oldrng=execute(old);pars=list(old.parameters());oldgrads=torch.autograd.grad(oldloss,pars,retain_graph=True,allow_unused=True);_,_,oldKL=rd(*oldz,labels);klgrads=torch.autograd.grad(.25*oldKL,pars,allow_unused=True)
 torch.manual_seed(202610177014);new=api.build(c,'cpu').train();assert all(torch.equal(old_init[k],v) for k,v in new.state_dict().items());newloss,newinputs,newz,newrng=execute(new)
 assert len(oldinputs)==len(newinputs)==8 and all(torch.equal(v,w) for v,w in zip(oldinputs,newinputs));assert torch.equal(x,saved) and torch.equal(oldrng[0],newrng[0]) and oldrng[1]==newrng[1]
 assert torch.equal(newinputs[0],ordinary) and torch.equal(newinputs[0],newinputs[1]) and all(torch.equal(v,x) for v in newinputs[2:6]) and torch.equal(newinputs[6],newinputs[7])
 assert all(torch.equal(v,w) for v,w in zip(oldz,newz));assert torch.allclose(oldloss-newloss,.25*oldKL,rtol=1e-5,atol=2e-6)
 record=new._joint_details[-1];assert record['extra_pair_loss']==record['extra_pair_CE'] and record['extra_pair_KL']>0 and record['As_KL_loss_weight']==0. and record['As_KL_gradient_terms']==0
 total=.25*(record['ordinary_CE']+record['ordinary_KL'])+.5*(record['raw_CE']+record['raw_KL'])+.25*(record['s_slot_CE']+record['s_slot_KL'])+record['supcon_weight']*record['supcon']+.25*record['extra_pair_CE'];assert math.isclose(float(newloss.detach()),total,abs_tol=1e-6)
 W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=new.get_parameter(W).detach().clone();opt=build(new,list(new.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);newloss.backward();compared=0;worst=0.
 for (name,p),oldg,kg in zip(new.named_parameters(),oldgrads,klgrads):
  if p.grad is None:assert oldg is None;continue
  want=oldg-(torch.zeros_like(oldg) if kg is None else kg);err=float((p.grad-want).abs().max());worst=max(worst,err);assert torch.allclose(p.grad,want,rtol=2e-4,atol=4e-6),(name,err);compared+=1
 assert new._As_pair_calls==1 and new._joint_L_packets==1024 and new._additive_L_packets==256 and new._joint_As_packets==128 and len(record['supcon_details'])==4 and not new._supcon_handles and new._supcon_pending is None and not new._additive_bn_names
 clip_physical(opt,new.parameters(),5.);opt.step();assert not torch.equal(before,new.get_parameter(W))
 new.eval()
 with torch.no_grad(),fresh_parameter_caches(new):
  feat=new.id_backbone.encoder.features(x[:2]);export=api.export(new).eval();assert feat.shape==(2,160) and torch.equal(feat,export(x[:2]));assert new.forward_identity_only(x[:2])['tx_logits'].shape==(2,6)
 command=[sys.executable,'-S',str(root/'experiments/cvs_as_pair_ce20/runner.py'),'--epochs','40'];r=subprocess.run(command,capture_output=True,text=True);assert r.returncode!=0 and 'No budget/data/operator/seed overrides' in r.stderr;negative+=1
 publisher=(root/'experiments/cvs_as_pair_ce20/publish.py').read_text();dispatcher=(root/'experiments/cvs_as_pair_ce20/dispatch.py').read_text();assert 'cvs_supcon_views20/runner.py' not in publisher+dispatcher and publisher.count('experiments/cvs_as_pair_ce20/runner.py')==2 and 'experiments/cvs_as_pair_ce20/runner.py' in dispatcher
 return dict(status='PASS',negative_checks=negative,real_F_synthetic_forward_packets=2054,synthetic_optimizer_updates=1,synthetic_fullAs_generations=256,real_IQ=0,checkpoint_loads=0,target_access=False,equal_initial_state=True,eight_actual_IQ_inputs_exact=True,dropout_global_RNG_exact=True,only_extra_KL_gradient_removed=True,compared_parameter_gradients=compared,max_gradient_equivalence_error=worst,default_head_updated=True,export160_exact=True,objective_KL_detached=True,native_argument_changes=sorted(changed))
if __name__=='__main__':print(json.dumps(run()))
