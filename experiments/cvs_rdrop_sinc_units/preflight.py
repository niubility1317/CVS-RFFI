"""One real source checkpoint, no query; source metadata and unit policy only."""
import os,torch
from . import design as d,model as api
from .provenance import reference_proof
from .optimizer import specification
from .sinc_change import compare
from experiments.cvs_equivariant_identity.precision import numerical_context

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preflight only')
 torch.set_num_threads(2)
 for row in d.rows():
  if row['arm']==d.BASELINE_ARM:reference_proof(d.config(row))
 row=next(r for r in d.rows() if r['arm']==d.BASELINE_ARM and r['model_seed']==d.SEEDS[0]);c=d.config(row);p=reference_proof(c)
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);initial=torch.load(p/'initial_smoke.pt',map_location='cpu',weights_only=False)
 from experiments.cvs_sigtime_rdrop_legacy.provenance import validate_checkpoint
 validate_checkpoint(cp,d.reference_config(d.SEEDS[0]))
 if not initial['scratch_only']:raise ValueError('Reference actual initial model was not scratch')
 with numerical_context(d.FULL_FP32_POLICY):
  m=api.build(c,'cpu').eval();m.load_state_dict(cp['model'],strict=True)
  with torch.no_grad():out=m.forward_identity_only(torch.zeros(2,2,256))
  if out['tx_logits'].shape!=(2,6) or out['id_feat_joint'].shape!=(2,160) or not torch.isfinite(out['id_feat_joint']).all():raise ValueError('Original default head/deployed feature smoke failed')
  specification(m,[p for p in m.parameters() if p.requires_grad],'NORMALIZED_UNITS',dict(lr=.0002,eps=1e-8,weight_decay=.0001))
  sinc=m.get_submodule('id_backbone.encoder.core.id_backbone.sinc');student=compare(initial['model'],cp['model'],type(sinc));ema=compare(initial['model'],cp['ema_model'],type(sinc))
 value=dict(status='PASS',real_source_checkpoint_no_query_smoke=True,reference_model_code_commit=d.MODEL_CODE_COMMIT,all_three_reference_source_metadata_verified=True,source_only=True,checkpoint_used_to_initialize_training=False,query_read=False,actual_IQ_reads=0,head_fit_updates=0,optimizer_steps=0,reference_student_Sinc=student,reference_ownEMA_Sinc=ema)
 d.write(d.BASE/'preflight.json',value);print('PREFLIGHT PASS realcheckpoint/noquery; continue immediately',flush=True);return value
