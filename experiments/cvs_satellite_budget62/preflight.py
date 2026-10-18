"""One compound no-query smoke of both actual FINAL30 parents; then train."""
import torch
from . import design as d,model
from .checks import run as focused_checks
from .source import parent_payload

def smoke():
 d.write(d.BASE/'focused_checks.json',focused_checks())
 records=[]
 for row in d.rows():
  c=d.config(row);parent=parent_payload(c);m=model.build(c,'cpu').eval();m.load_state_dict(parent['model'],strict=True);before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
  if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or m._sat_engine.active:raise ValueError('Own actual FINAL30 default-head no-query smoke failed')
  records.append(dict(row_id=row['row_id'],mode=c['satellite_mode'],NN=2,source_IQ=0,target_query=False,checkpoint=c['parent_source']+'/final_ssdg.pth',training_initializer=True,same_source_contract=True,optimizer_updates=0))
  del m,parent,before
 result=dict(status='PASS',records=records,NN=4,real_checkpoints=2,source_IQ=0,target_query=False,synthetic_remote_NN=0,optimizer_updates=0)
 d.write(d.BASE/'real_checkpoint_smoke.json',result)
 print('COMPOUND_OWN_FINAL30_REAL_CP_NO_QUERY PASS4NN;continue training',flush=True)
 return result
