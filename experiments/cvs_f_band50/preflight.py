import os,torch
from . import design as d,model as api
from .provenance import reference_proof
from experiments.cvs_equivariant_identity.precision import numerical_context
def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preflight only')
 torch.set_num_threads(2)
 for row in d.rows():
  if row['arm']=='FULL':reference_proof(d.config(row))
 c=d.config(d.rows()[0]);p=reference_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 with numerical_context(d.FULL_FP32_POLICY):
  m=api.build(c,'cpu').eval();m.load_state_dict(ck['model'],strict=True)
  with torch.no_grad():v=m.forward_identity_only(torch.zeros(2,2,256))
  if v['tx_logits'].shape!=(2,6) or v['id_feat_joint'].shape!=(2,160) or not torch.isfinite(v['tx_logits']).all() or len(m.id_backbone.encoder.sequence_pool.views)!=3:raise ValueError('Real F default checkpoint smoke failed')
 d.write(d.BASE/'preflight.json',dict(status='PASS',real_checkpoint_no_query_smoke=True,all_three_reference_provenance='VERIFIED',training_initialization='all new models scratch; no inherited F weight',target_IQ_reads=0,head_fit_updates=0))
 print('PREFLIGHT PASS actual F checkpoint/noquery; continue',flush=True)
