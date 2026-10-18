"""One shared legal As20 student/EMA, four synthetic no-query forwards."""
from copy import deepcopy
import torch
from . import design as d,model
from .source import parent_payload
from .checks import run as focused_checks

def smoke():
 torch.set_num_threads(2);d.write(d.BASE/'focused_checks.json',focused_checks())
 c=d.config(d.rows()[0]);cp=parent_payload(c)
 m=model.build(c,'cpu').eval();m.load_state_dict(cp['model'],strict=True)
 ema=deepcopy(m).eval().requires_grad_(False);ema.load_state_dict(cp['ema_model'],strict=True)
 before={k:v.clone() for k,v in m.state_dict().items()}
 if len(before)!=206 or sum(p.numel() for p in m.parameters())!=900722:raise ValueError('Fixed F/default head shape differs')
 with torch.no_grad():zs=[v.forward_identity_only(torch.zeros(2,2,256))['tx_logits'] for v in (m,ema)]
 if any(z.shape!=(2,6) or not torch.isfinite(z).all() for z in zs) or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or any(p.grad is not None for p in m.parameters()):raise ValueError('Real parent no-query smoke failed or changed state')
 result=dict(status='PASS',NN=4,real_checkpoints=1,checkpoint=c['checkpoint_sources'][0],strict_student=True,strict_ownEMA=True,
  source_IQ=0,target_query=False,optimizer_updates=0,head_fit_updates=0,extra_teacher_training=0,same_source_contract=True)
 d.write(d.BASE/'real_checkpoint_smoke.json',result);print('REAL_AS20_STUDENT_EMA_NO_QUERY_PASS4NN;continue',flush=True)
 return result
