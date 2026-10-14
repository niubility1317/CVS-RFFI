import json
import torch
from . import design as d
from .checks import checks
from .audit import source_model
from experiments.cvs_signature_stats50.predict import source_proof


def main():
    torch.set_num_threads(2);local=checks()
    source_proof(d.source_config(),design=d.parent)
    model=source_model().eval().requires_grad_(False);before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256))
    if out['id_feat_joint'].shape!=(2,160) or out['tx_logits'].shape!=(2,6) or not torch.isfinite(out['tx_logits']).all():
        raise ValueError('Real source checkpoint no-query smoke failed')
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):
        raise ValueError('No-query smoke changed checkpoint state')
    p=d.ROOT/'preflight.json'
    if p.exists():raise FileExistsError('Immutable preflight exists')
    value=dict(status='PASS',checks=local,real_checkpoint_no_query_smoke=True,synthetic_IQ_packets=2,
        checkpoint_parent_run=d.parent.RUN,checkpoint_epoch=50,training_inherits_checkpoint=False,
        source_only=True,query_read=False,head_updates=0,optimizer_steps=0)
    d.write(p,value);print(json.dumps(value))


if __name__=='__main__':main()
