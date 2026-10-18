"""First launcher step: both legal own20 student and EMA states;8NN,no source/query IQ."""
from copy import deepcopy
import torch
from . import design as d,model
from .source import parent_payload
from .checks import run as focused_checks

def smoke():
 torch.set_num_threads(2)
 d.write(d.BASE/'focused_checks.json',focused_checks())
 rows=[]
 for row in d.rows():
  c=d.config(row);cp=parent_payload(c);m=model.build(c,'cpu').eval();m.load_state_dict(cp['model'],strict=True)
  ema=deepcopy(m).eval().requires_grad_(False);ema.load_state_dict(cp['ema_model'],strict=True)
  if len(m.state_dict())!=206 or sum(p.numel() for p in m.parameters())!=900722:raise ValueError('Fixed original F/default head shape')
  before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=[v.forward_identity_only(torch.zeros(2,2,256))['tx_logits'] for v in (m,ema)]
  if any(v.shape!=(2,6) or not torch.isfinite(v).all() for v in z) or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Own student/EMA/default head smoke failed')
  rows.append(dict(arm=c['arm'],checkpoint=c['parent_source']+'/final_ssdg.pth',student_NN=2,EMA_NN=2,strict_load=True))
  del m,ema,cp,before
 r=dict(status='PASS',NN=8,real_checkpoints=2,rows=rows,source_IQ=0,target_query=False,optimizer_updates=0,head_fit_updates=0,extra_teacher_training=0,same_source_contract=True,training_initializers=True)
 d.write(d.BASE/'real_checkpoint_smoke.json',r)
 print('OWN20_STUDENT_AND_EMA_NO_QUERY_PASS8NN;continue training',flush=True)
 return r
