"""Only adapt training batch size; the F extractor and default head are unchanged."""
from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_sigtime_rdrop_legacy import model as legacy
from experiments.cvs_fullsource_concat30_25.f_base.coordinate import CoordinateSinc
from experiments.cvs_fullsource_concat30_25.f_base.x3_features import CompositePool
from experiments.cvs_fullsource_concat30_25.supcon_math import LEVELS,PREFIX,RX_BY_DOMAIN,penalty,schedule
from .route import mask_for_step
def F_configure(m,c):
    ref=m.id_backbone.encoder
    ref.sequence_pool=CompositePool(ref.sequence_pool.channels,'SIG_TIME_X3',c['model_seed'])
    ref.core.id_backbone.sinc=CoordinateSinc(ref.core.id_backbone.sinc,'CENTER_25K')
    m._feature_learning_report.update(arm='SIG_TIME_X3',modes=['SIG_TIME']*3,parameter_matched=False,
        independent_local_projections=True,local_dimension_each=6,descriptor_width_each=28,
        extra_parameters=sum(p.numel() for p in ref.sequence_pool.parameters()),
        combination='Three independent local views jointly feed one actual joint160; no voting',
        sinc_coordinates='CENTER_25K',Sinc_optimizer='CENTER_UNITS',A_F_arm='F')
    return m

source_domain_count,classifier=legacy.source_domain_count,legacy.classifier
def validate_labels(y,n):
 if n!=128 or y.dtype!=torch.long or y.shape!=(128,) or not bool(((y>=0)&(y<6)).all()):raise ValueError('Only the original labeled physical L128 may supervise')
def paired_terms(a,b,y,step):
 validate_labels(y,128)
 if a.shape!=(256,6) or b.shape!=a.shape or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):raise ValueError('Both concatenated default logits must be finite256x6')
 values=[];ces=[];kls=[]
 for s in (slice(0,128),slice(128,256)):
  value,ce,kl,_,_=legacy.terms(a[s],b[s],y,1.,.01);values.append(value);ces.append(ce);kls.append(kl)
 with torch.no_grad():
  mask=mask_for_step(step,y.device);_,satce,satkl,_,_=legacy.terms(a[128:][mask].detach(),b[128:][mask].detach(),y[mask],1.,.01);_,ordce,ordkl,_,_=legacy.terms(a[128:][~mask].detach(),b[128:][~mask].detach(),y[~mask],1.,.01)
 parts=dict(clean_CE=ces[0].detach(),second_CE=ces[1].detach(),clean_KL=kls[0].detach(),second_KL=kls[1].detach(),satellite_CE=satce,secondary_ordinary_CE=ordce,satellite_KL=satkl,secondary_ordinary_KL=ordkl)
 return .5*(values[0]+values[1]),.5*(ces[0]+ces[1]),.5*(kls[0]+kls[1]),parts
def count_features(self,x):
 if self.training:self._full_train_packets+=len(x)
 else:self._full_eval_packets+=len(x)
 return self._full_original_features(x)
def capture_maps(m,destination):
 handles=[]
 for level in LEVELS:
  def capture(module,args,out,level=level):
   if level in destination or out.ndim!=3 or len(out)!=256:raise ValueError('One real concatenated feature map per forward')
   destination[level]=out[:128]
  handles.append(m.get_submodule(PREFIX+level).register_forward_hook(capture))
 return handles
@contextmanager
def labeled_scope(self):
 if not self.training:
  with self._full_base_scope():yield
  return
 if self._full_pending is not None:raise ValueError('Previous labeled graph not consumed')
 step=self._feature_l_steps+1
 pending=dict(step=step,input=None,first={})
 self._full_pending=pending
 def capture(module,args):
  if pending['input'] is not None or args[0].shape!=(256,2,256):raise ValueError('Native concat must provide one real256 IQ batch')
  pending['input']=args[0].detach().clone()
 handles=[self.register_forward_pre_hook(capture)]
 if schedule((step-1)//222+1):handles+=capture_maps(self,pending['first'])
 try:
  with self._full_base_scope():yield
 except BaseException:
  self._full_pending=None;raise
 finally:
  for h in handles:h.remove()
 if pending['input'] is None:raise ValueError('Actual native concatenation not captured')
def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count):
 validate_labels(labels,labeled_count);p=self._full_pending
 if model is not self or not self.training or not torch.is_grad_enabled() or p is None or p['step']!=optimizer_step or optimizer_step!=self._full_calls+1 or self._feature_l_steps!=optimizer_step:raise ValueError('Only ordered native L step permitted')
 if domains is None or domains.shape!=(256,) or not torch.equal(domains[:128],domains[128:]):raise ValueError('Concat domain copies differ')
 rx=torch.as_tensor(extra[1]['rx_i'],device=labels.device,dtype=torch.long)
 if rx.shape!=(128,) or not torch.equal(rx,torch.tensor(RX_BY_DOMAIN,device=domains.device)[domains[:128]]):raise ValueError('Original physical RX/domain mismatch')
 first=labeled_output['tx_logits'];yy=torch.cat((labels,labels));native_ce=F.cross_entropy(first,yy,label_smoothing=.01)
 if not torch.allclose(identity_loss.detach(),native_ce.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Native concat CE normalization changed')
 second_maps={};weight=schedule((optimizer_step-1)//222+1)
 handles=capture_maps(self,second_maps) if weight else []
 try:
  second=self.forward_identity_only(p['input'],domain_labels=domains)['tx_logits']
  value,ce,kl,parts=paired_terms(first,second,labels,optimizer_step);details=[];aux=first.sum()*0
  if weight:
   values=[]
   for phase,maps in [('first',p['first']),('second',second_maps)]:
    vv,stats=penalty(maps,labels,rx,'CROSS_RX');values.append(vv);details+= [dict(phase=phase,**s) for s in stats]
   aux=torch.stack(values).mean();self._full_supcon_applied+=1
  self._full_calls=optimizer_step;self._full_extra_packets+=256
  self._full_records.append(dict(step=optimizer_step,epoch=(optimizer_step-1)//222+1,weight=weight,loss=float(aux.detach()),details=details,physical_L=128,forward_L=512,satellite_L=64,ordinary_weight=.75,satellite_weight=.25,classification_terms={k:float(v) for k,v in parts.items()},head_fit_updates=0,U_truth_read=False))
  metrics={'full/RDrop_CE':ce.detach(),'full/RDrop_KL':kl.detach(),'full/SupCon':aux.detach(),'full/SupCon_weight':weight,**{'full/'+k:v for k,v in parts.items()}}
  return value-identity_loss+weight*aux,metrics
 finally:
  for h in handles:h.remove()
  self._full_pending=None
hook._accepts_labeled_inputs=False
def attach(m,c):
 d.validate(c);m._full_base_scope=m._legacy_original_labeled_scope;m._full_pending=None;m._full_calls=0;m._full_extra_packets=0;m._full_supcon_applied=0;m._full_records=[]
 m._phase1_labeled_feature_context=MethodType(labeled_scope,m);m._source_labeled_feature_hook=MethodType(hook,m)
 e=m.id_backbone.encoder;e._full_original_features=e.features;e._full_train_packets=0;e._full_eval_packets=0;e.features=MethodType(count_features,e)
 m._feature_learning_report['original_SIGTIME_RDrop']=dict(m._feature_learning_report['original_SIGTIME_RDrop'],extra_L_identity_forwards=852480,original_L_identity_forwards=852480,construction_reference='legacy E20 factory config only; no E20 weights, actual training30/20L10U in current design')
 m._feature_learning_report.update(fullsource_native_concat=c['training_mechanism'],SupCon=c['supcon'],deployed_structure_unchanged=True)
 return m
def install(m,c):return attach(F_configure(legacy.install(m,d.parent_config()),c),c)
def build(c,device='cpu'):return attach(F_configure(legacy.build(d.parent_config(),'cpu'),c).to(device),c)
def own_model(c,device='cpu'):
 d.validate(c);out=__import__('pathlib').Path(c['output_root'])
 done=d.read(out/'completion.json');init=d.read(out/'initialization.json');freeze=d.read(out/'source_frozen.json')
 if done['config']!=c or done['epoch']!=30 or done['optimizer_steps']!=6660 or done['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['source_roles']!='EXACT_MATCH' or freeze['config']!=c or freeze['epoch']!=30 or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Own full source FINAL30 provenance differs')
 if d.read(out/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Own checkpoint physical role contract differs')
 ck=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=30 or ck['a1_ema_successful_updates']!=6660 or ck['candidate_id']!=c['row_id'] or ck['run_id']!=d.RUN or not ck['args']['from_scratch'] or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt']:raise ValueError('Fixed scratch model identity differs')
 m=build(c,device);m.load_state_dict(ck['model'],strict=True);return m.eval().requires_grad_(False)
