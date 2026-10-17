"""Actual native64 practical routing and actual F default-head gradients;no real IQ."""
import io,json
import torch
from . import design as d,model as api
from .observer import Observer
from .route import mask_for_step

def run():
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_phase1_stack.fast_execution import configure
 from experiments.cvs_fullsource_concat30_25.supcon_math import RX_BY_DOMAIN
 c=d.config(d.rows()[0]);out=d.BASE/'synthetic';out.mkdir(exist_ok=False);torch.set_num_threads(2)
 m=api.build(c,'cpu').train();before={k:v.clone() for k,v in m.state_dict().items()};x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(3701));ordinary=x+.01;y=torch.arange(128)%6;domain=torch.arange(128)%15
 meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor(RX_BY_DOMAIN)[domain],day_i=domain.remainder(3)+1,sig_i=torch.arange(128),base_index=-torch.arange(128)-1)
 o=Observer(c);o.model=m;o.out=out;o.current=(y.tolist(),meta);o.raw_L=x.clone();o.sequence=io.StringIO();o.physics=io.StringIO();a=configure(d.make_args(c,'cpu'));a.output_dir=str(out/'native_execution')
 with installed(c) as n:
  from cvsrffi import practical_adapter as pa
  previous=pa._context.get();pa.set_training_context(meta,1,'L_s')
  try:
   with o.native_scope(n):
    aug=n.ConcatSatChannelAugment(scenarios=list(d.TRAIN_SCENES),schedule='',p=1.,seed=a.sat_view_seed,apply_fn=n.apply_sat_channel_for_scenario)
    result=aug.expand(ordinary,y,domain,args=a,epoch=1,batch_idx=1);evidence=json.loads(o.physics.getvalue())
  finally:pa._context.set(previous)
 z1=m.forward_identity_only(result.x,domain_labels=result.d_raw)['tx_logits'];z2=m.forward_identity_only(result.x,domain_labels=result.d_raw)['tx_logits'];loss,ce,kl,parts=api.paired_terms(z1,z2,y,1)
 expected=.5*(api.legacy.terms(z1[:128],z2[:128],y,1.,.01)[0]+api.legacy.terms(z1[128:],z2[128:],y,1.,.01)[0])
 if not torch.allclose(loss,expected,atol=1e-7,rtol=1e-6):raise ValueError('Original normalized loss scale changed')
 loss.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head
 if head.weight.grad is None or not torch.isfinite(head.weight.grad).all() or not bool(head.weight.grad.norm()>0) or not any(p.grad is not None and bool(p.grad.norm()>0) for k,p in m.named_parameters() if 'sinc' in k):raise ValueError('Original head and backbone gradients missing')
 if m.id_backbone.encoder._full_train_packets!=512 or o.sat_packets!=64 or len(evidence['records'])!=64 or set(parts)!={'clean_CE','second_CE','clean_KL','second_KL','satellite_CE','secondary_ordinary_CE','satellite_KL','secondary_ordinary_KL'} or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Actual synthetic path/cost/no optimizer mutation differs')
 if __import__('pathlib').Path(c['output_root']).exists():raise ValueError('Synthetic check must never create real training output')
 value=dict(status='PASS',NN=512,synthetic_As=64,real_IQ=0,real_CP=0,optimizer_updates=0,parameters=sum(p.numel() for p in m.parameters()),state_tensors=len(before),head_shape=list(head.weight.shape),head_scale=float(head.scale),head_gradient_finite=True,backbone_gradient_finite=True,all_generated_satellite_supervised=True,old_U_changed=False,ordinary_weight=.75,satellite_weight=.25,classification_parts={k:float(v) for k,v in parts.items()},original_training_context_restored=True)
 d.write(out/'complete.json',value);return value
