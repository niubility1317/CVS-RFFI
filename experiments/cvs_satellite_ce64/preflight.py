"""First launcher step: one real As20 parent, no query or repeated F synthetic."""
import torch
from . import design as d,model
from .source import parent_payload
from .checks import run as focused_checks

def smoke():
    torch.set_num_threads(2)
    d.write(d.BASE/'focused_checks.json',focused_checks())
    c=d.config(d.rows()[0]);cp=parent_payload(c);m=model.build(c,'cpu').eval();m.load_state_dict(cp['model'],strict=True);before={k:v.clone() for k,v in m.state_dict().items()}
    with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
    if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Actual own As20 default-head smoke failed')
    result=dict(status='PASS',NN=2,real_checkpoints=1,checkpoint=c['parent_source']+'/final_ssdg.pth',training_initializer=True,same_source_contract=True,source_IQ=0,target_query=False,optimizer_updates=0,synthetic_remote_NN=0)
    d.write(d.BASE/'real_checkpoint_smoke.json',result)
    print('ACTUAL_AS20_PARENT_DEFAULT_HEAD_NO_QUERY PASS2NN;continue training',flush=True)
    return result
