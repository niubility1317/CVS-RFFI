import json
import torch
from . import design as d
from .checks import checks
from experiments.cvs_receiver_dro50 import predict as first_api


def main():
    torch.set_num_threads(2)
    d.require_source(); local = checks()
    c = d.first.config(next(r for r in d.first.training_rows() if r['arm'] == 'SIG_TIME'))
    m = first_api.own_model(c, 'cpu').eval().requires_grad_(False)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        z = m.forward_identity_only(torch.zeros(2, 2, 256))['id_feat_joint']
    if z.shape != (2, 160) or not torch.isfinite(z).all():
        raise ValueError('Remaining context real checkpoint no-query smoke failed')
    if any(not torch.equal(v, m.state_dict()[k]) for k, v in before.items()):
        raise ValueError('Read-only no-query smoke changed checkpoint state')
    path = d.ROOT / 'preflight.json'
    if path.exists():
        raise FileExistsError('Immutable preflight exists')
    value = dict(status='PASS', checks=local, source_qualification='PASS',
        source_qualification_run=d.first.RUN, real_checkpoint_no_query_smoke=True,
        synthetic_IQ_packets=2, source_checkpoint='Predetermined first SIG_TIME final50, same source contract',
        training_inherits_checkpoint=False, query_read=False, head_fit_updates=0, optimizer_updates=0)
    d.write(path, value); print(json.dumps(value))


if __name__ == '__main__':
    main()
