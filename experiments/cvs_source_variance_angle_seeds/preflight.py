import json
import torch
from . import design as d
from .checks import checks
from experiments.cvs_backbone_fishr_seeds import predict as api


def main():
    from . import audit
    torch.set_num_threads(2)
    local = checks()
    previous = d.read(d.first.BASE / 'analysis.json')
    d.first.validate(previous['config'])
    if previous['status'] != 'ANALYZED' or previous['qualification']['status'] != 'PASS':
        raise ValueError('First source-only qualification missing or failed')
    c = d.source_config(d.rows()[0]); api.source_proof(c)
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
             source_qualification='PASS', source_decision_commit=d.SOURCE_DECISION_COMMIT,
             real_checkpoint_no_query_smoke=True, synthetic_IQ_packets=2,
             training_inherits_checkpoint=False, source_only=True, query_read=False,
             head_updates=0, optimizer_steps=0)
    d.write(p, v); print(json.dumps(v))


if __name__ == '__main__':
    main()
