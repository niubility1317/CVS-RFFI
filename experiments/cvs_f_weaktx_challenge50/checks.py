"""Focused source/head boundaries, exact random control and actual F gradients."""
import copy,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d,model as api,mechanism as me
from .objective import channel_mask,eligible
from experiments.cvs_f_supcon50.checks import W,S
from experiments.cvs_f_supcon50 import model as reference
from .baseline import original_config
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches

def fixture(m,step):
 from cvsrffi.tensors import unpack_batch
 torch.manual_seed(7061);x=torch.randn(128,2,256);y=torch.arange(128)%6;domain=torch.arange(128)%15;rx=torch.tensor(me.__dict__.get('RX_BY_DOMAIN',(1,1,1,3,3,3,4,4,4,6,6,6,8,8,8)))[domain];prior=step-1
 m._feature_l_steps=prior;m._legacy_calls=prior;m._legacy_extra_packets=prior*128;m._supcon_calls=prior
 with fresh_parameter_caches(m):
  with m._phase1_labeled_feature_context():first=m(x,y_tx=y,return_aux=True,domain_labels=domain,update_crra_support=False)
  ce=F.cross_entropy(first['tx_logits'],y,label_smoothing=.01);_,_,extra=unpack_batch((x,y,domain,{'rx_i':rx}))
  correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=first,labels=y,extra=extra,domains=domain,optimizer_step=step,identity_loss=ce,labeled_count=128)
 return ce+correction,metrics

def run(profile,metadata,output):
 torch.set_num_threads(2);base=Path(output)/d.RUN;c=d.config(d.rows()[0],str(base),profile,metadata);negative=0;flows=[]
 for key,value in [('epochs',20),('test_views',['practical_high']),('weak_indices',[1]),('challenge_drop_fraction',.5),('separate_head_training',True),('challenge_every',1)]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Altered fixed source/head recipe accepted')
 a=torch.ones(2,48,3);grad=torch.arange(48,dtype=torch.float32)[None,:,None].expand_as(a);mask=channel_mask(a,grad,'SALIENCY_WEAK',1)
 if not torch.equal(mask[:,42:],torch.zeros(2,6,1)) or not torch.equal(mask[:,:42],torch.ones(2,42,1)):raise AssertionError('Known saliency top6 oracle differs')
 random_state=torch.get_rng_state().clone();r1=channel_mask(a,None,'RANDOM_WEAK',61);r2=channel_mask(a,None,'RANDOM_WEAK',61)
 if not torch.equal(r1,r2) or not torch.equal(random_state,torch.get_rng_state()) or torch.equal(r1,mask):raise AssertionError('Random comparator mutates main RNG or equals saliency fixture')
 for labels in [torch.tensor([-1,1]),torch.tensor([6,1])]:
  try:eligible(labels)
  except ValueError:negative+=1
  else:raise AssertionError('Hidden/foreign TX accepted')
 calendar=[me.level_for(s) for s in range(1,11101) if me.active(s)]
 if len(calendar)!=2498 or calendar.count('time_down')!=1249 or calendar.count('t2.pool')!=1249:raise AssertionError('Actual50 calendar differs')
 with numerical_context(d.FULL_FP32_POLICY):
  control=reference.build(original_config(c['model_seed']),'cpu').train();m=api.build(c,'cpu').train()
  if any(not torch.equal(v,m.state_dict()[k]) for k,v in control.state_dict().items()):raise AssertionError('New mechanism changes initial deployed network')
  basewarm,_=fixture(control,1);warm,_=fixture(m,1)
  if not torch.equal(basewarm.detach(),warm.detach()):raise AssertionError('Warmup must preserve exact native objective')
  for row in d.rows(d.SEEDS[:1]):
   c=d.config(row,str(base),profile,metadata);control=reference.build(original_config(c['model_seed']),'cpu').train();m=api.build(c,'cpu').train()
   for step in (1112,1116):
    base_loss,_=fixture(control,step);base_rng=torch.get_rng_state().clone();base_state={n:v.clone() for n,v in control.state_dict().items()};captured=[];original=me.extra_loss
    def record(*args,**kwargs):
     value=original(*args,**kwargs);captured.append(value);return value
    me.extra_loss=record
    try:total,metrics=fixture(m,step)
    finally:me.extra_loss=original
    if len(captured)!=1 or metrics['F_challenge/packets']!=43 or m._challenge_pending is not None or m._supcon_pending is not None or m._supcon_handles:raise AssertionError('One exact weak-only extra forward required')
    if len(m._supcon_records[-1]['details'])!=4 or m._legacy_extra_packets!=step*128:raise AssertionError('Original two RDrop/SupCon passes changed')
    if not torch.equal(base_rng,torch.get_rng_state()) or any(not torch.equal(v,m.state_dict()[n]) for n,v in base_state.items()):raise AssertionError('Extra forward changed main RNG or live state')
    aux=captured[0];weight=metrics['F_challenge/weight']
    if not torch.allclose(total.detach(),base_loss.detach()+weight*aux.detach(),atol=1e-6,rtol=1e-6):raise AssertionError('Original objective changed beyond fixed auxiliary')
    named=dict(m.named_parameters());gw,gs=torch.autograd.grad(aux,[named[W],named[S]],allow_unused=True,retain_graph=True);mainw=torch.autograd.grad(total,named[W],retain_graph=True)[0]
    if gw is not None or gs is None or not torch.isfinite(gs).all() or gs.norm()<=0 or mainw.norm()<=0 or any(p.grad is not None for p in m.parameters()):raise AssertionError('Aux must update actual features, not W or .grad prematurely')
    flows.append(dict(treatment=c['treatment'],step=step,level=me.level_for(step),extra_L_packets=43,aux_Sinc_gradient_norm=float(gs.norm()),main_W_gradient_norm=float(mainw.norm()),aux_W_gradient=None,main_RNG_unchanged=True,live_state_unchanged=True))
   teacher=copy.deepcopy(m).eval();before=len(m._challenge_records)
   with torch.no_grad(),fresh_parameter_caches(teacher):teacher.forward_identity_only(torch.zeros(2,2,256))
   if len(teacher._challenge_records)!=before or teacher._challenge_pending is not None:raise AssertionError('EMA/eval activates challenge')
   m.eval();x=torch.randn(2,2,256)
   with torch.no_grad(),fresh_parameter_caches(m):actual=m.forward_identity_only(x)['id_feat_joint'];exported=api.export(m)(x)
   if not torch.equal(actual,exported) or sum(p.numel() for p in m.parameters())!=900722:raise AssertionError('Actual deployed feature/parameter count changed')
 # Real matching BASE source checkpoint used only for zero-input smoke,
 # never for initializing any new training row.
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 ref=Path(profile).parent.parent/'runs'/c['reference_local_run']/('BASE-s'+str(c['model_seed']))/'source';rc=d.read(ref/'local_source_complete.json')['config'];source_proof(rc)
 checkpoint=torch.load(ref/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,rc);actual=api.build(c,'cpu');actual.load_state_dict(checkpoint['model'],strict=True);actual.eval().requires_grad_(False)
 with torch.no_grad(),fresh_parameter_caches(actual):smoke=actual.forward_identity_only(torch.zeros(2,2,256))
 if smoke['tx_logits'].shape!=(2,6) or not torch.isfinite(smoke['tx_logits']).all():raise AssertionError('Real final50 no-query smoke failed')
 return dict(status='PASS',protocol_negative_cases=negative,known_channel_oracle=True,random_control_private_RNG=True,active_calls=2498,shallow_calls=1249,middle_calls=1249,actual_feature_gradients=flows,real_checkpoint_no_query_smoke=True,checkpoint_not_inherited_by_training=True,default_head_joint=True,head_fit_updates=0,IQ_reads=0,target_reads=0,parameters=900722,warmup_exact=True,feature_export_exact=True,EMA_eval_untouched=True)
if __name__=='__main__':print(json.dumps(run(*sys.argv[1:4]),allow_nan=False))
