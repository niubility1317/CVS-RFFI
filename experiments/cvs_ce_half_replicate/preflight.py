"""One real-checkpoint no-query smoke per new seed, then immediate training."""
import torch
from . import design as d, model
from .source import parent_payload


def smoke():
    torch.set_num_threads(2)
    records = []
    for row in d.rows():
        c = d.config(row)
        cp = parent_payload(c)
        m = model.build(c, 'cpu').eval().requires_grad_(False)
        m.load_state_dict(cp['model'], strict=True)
        before = {k: v.clone() for k, v in m.state_dict().items()}
        with torch.no_grad():
            z = m.forward_identity_only(torch.zeros(2, 2, 256))['tx_logits']
        if z.shape != (2, 6) or not torch.isfinite(z).all():
            raise ValueError('Real parent default-head smoke failed')
        if any(not torch.equal(v, m.state_dict()[k]) for k, v in before.items()):
            raise ValueError('No-query smoke mutated parent state')
        if m._replication_factory_seed != row['model_seed']:
            raise ValueError('Actual F factory seed differs')
        records.append(dict(row_id=row['row_id'], seed=row['model_seed'], NN=2,
                            checkpoint=c['checkpoint_sources'][0], exact_source_roles=True,
                            own_ema_present=True, target_access=False, updates=0))
        del m, cp, before
    result = dict(status='PASS', records=records, NN=4, source_IQ=0,
                  target_query=False, optimizer_updates=0, continue_immediately=True)
    d.write(d.BASE / 'real_checkpoint_smoke.json', result)
    print('REAL_PARENT_DEFAULT_HEAD_SMOKE PASS; immediately continue', flush=True)
    return result
