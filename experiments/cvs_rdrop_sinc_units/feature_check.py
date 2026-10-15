"""Actual saved student/EMA Sinc and deployed-feature boundary; no IQ or head fit."""
import torch
from . import design as d,model as api
from .provenance import source_proof,validate_checkpoint
from .sinc_change import compare
from experiments.cvs_feature_sequence50.features import SequenceReference
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_equivariant_identity.precision import numerical_context
WNAME='id_backbone.encoder.core.id_backbone.cls_head.weight'
def measure(c,initial,final,ema_state):
 torch.set_num_threads(2)
 with numerical_context(d.FULL_FP32_POLICY):
  m=api.build(c,'cpu').eval().requires_grad_(False);names=dict(m.named_parameters());feature=[n for n in names if n.startswith('id_backbone.encoder.') and n!=WNAME]
  count=sum(names[n].numel() for n in feature)
  if count!=(232033 if c['combination']['SIG_TIME']=='SIG_TIME_X3' else 195041) or names[WNAME].shape!=(6,160):raise ValueError('Actual old deployed f_theta/defaultW boundary differs')
  sinc=m.get_submodule('id_backbone.encoder.core.id_backbone.sinc');native_type=sinc._native_type if hasattr(sinc,'_native_type') else type(sinc);student=compare(api.physical_state(initial),api.physical_state(final),native_type);ema=compare(api.physical_state(initial),api.physical_state(ema_state),native_type)
  with torch.random.fork_rng(devices=[]):torch.manual_seed(2026101542);x=torch.randn(8,2,256)
  def export():
   with torch.no_grad(),fresh_parameter_caches(m):return m.forward_identity_only(x)['id_feat_joint'].clone()
  m.load_state_dict(initial,strict=True);before=export();m.load_state_dict(final,strict=True);after=export();saved={n:v.detach().clone() for n,v in m.state_dict().items()};ref=next(v for v in m.modules() if isinstance(v,SequenceReference))
  with torch.no_grad(),fresh_parameter_caches(m):direct=api.export(m)(x).clone()
  with torch.no_grad():names[WNAME].zero_()
  Wzero=export()
  with torch.no_grad():names[WNAME].copy_(final[WNAME])
  prefix='id_backbone.encoder.core.id_backbone.sinc.'
  with torch.no_grad():
   for n in (('low_coordinate','band_coordinate') if c['combination']['Sinc_coordinates']=='CENTER_25K' else ('low_hz_','band_hz_')):names[prefix+n].copy_(initial[prefix+n])
  sinc_initial_only=export()
  with torch.no_grad():
   for n in (('low_coordinate','band_coordinate') if c['combination']['Sinc_coordinates']=='CENTER_25K' else ('low_hz_','band_hz_')):names[prefix+n].copy_(final[prefix+n])
  if not torch.equal(after,direct) or not torch.equal(after,Wzero) or any(not torch.equal(v,m.state_dict()[n]) for n,v in saved.items()):raise ValueError('Feature export/defaultW boundary or diagnostic restore failed')
  if not torch.isfinite(after).all() or (after-before).norm()<=1e-6:raise ValueError('Actual deployed feature unchanged/nonfinite')
  result=dict(status='VERIFIED',row_id=c['row_id'],model_seed=c['model_seed'],feature_parameters=count,defaultW_parameters=960,feature_changed_elements=sum(int((final[n]!=initial[n]).sum()) for n in feature),defaultW_changed_elements=int((final[WNAME]!=initial[WNAME]).sum()),student_Sinc=student,ownEMA_Sinc=ema,actual_feature='joint160/id_feat_joint',feature_export_exact=True,maximum_export_difference=float((after-direct).abs().max()),classW_zeroing_leaves_feature_identical=True,all_feature_parameters_buffers_preserved_by_diagnostic=True,synthetic_initial_final_feature_L2=float((after-before).norm()),synthetic_Sinc_only_counterfactual_L2=float((after-sinc_initial_only).norm()),synthetic_packets=8,synthetic_packet_forwards=40,synthetic_model_calls=5,actual_IQ_reads=0,query_read=False,head_fit_updates=0,optimizer_updates_by_diagnostic=0,limitation='Actual saved weights and synthetic interface/counterfactual check; not a newTX or source/target transfer test')
 return result

def run(c):
 p=source_proof(c);initial=torch.load(p/'initial_smoke.pt',map_location='cpu',weights_only=False);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(ck,c)
 if not initial['scratch_only']:raise ValueError('Own scratch initialization required')
 result=measure(c,initial['model'],ck['model'],ck['ema_model']);d.write(p/'source_feature_witness.json',result);return result
