import json
import torch
from . import design as d
from .checks import checks
from experiments.cvs_backbone_fishr50 import predict as api


def main():
    from . import audit
    torch.set_num_threads(2)
    local = checks()
    c = d.source_config(); api.source_proof(c)
    model = api.own_model(c, 'cpu').eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        result = model.forward_identity_only(torch.zeros(2, 2, 256))
    if result['id_feat_joint'].shape != (2, 160) or not torch.isfinite(result['id_feat_joint']).all():
        raise ValueError('Actual checkpoint joint160 smoke failed')
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()):
        raise ValueError('No-query smoke changed checkpoint state')
    p = d.ROOT / 'preflight.json'
    if p.exists():
        raise FileExistsError('Immutable preflight exists')
    v = dict(status='PASS', checks=local, actual_audit_import=True,
             real_checkpoint_no_query_smoke=True, synthetic_IQ_packets=2,
             training_inherits_checkpoint=False, source_only=True, query_read=False,
             head_updates=0, optimizer_steps=0)
    d.write(p, v); print(json.dumps(v))


if __name__ == '__main__':
    main()
