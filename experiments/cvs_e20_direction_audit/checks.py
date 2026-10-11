"""Bounded CPU/CUDA checks using real convolution/pool/dropout autograd paths."""
import argparse
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_e20_cross_style.multilevel import MultiLevel, calibrate
from .diagnostic import audit, style_hooks, cosine, _groups, _flat_grad, LEVELS


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.bn = nn.BatchNorm1d(channels)
        self.pool = nn.AvgPool1d(2)
        self.drop = nn.Dropout(.2)

    def forward(self, x):
        return self.drop(self.pool(F.silu(self.bn(self.pw(self.dw(x))))))


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_fuse = nn.Conv1d(2, 4, 3, padding=1)
        self.time_down = nn.AvgPool1d(2)
        self.t1 = Block(4)
        self.t2 = Block(4)
        self.projection = nn.Linear(4, 7)
        self.cls_head = nn.Linear(7, 6)
        self.low_hz_ = nn.Parameter(torch.tensor(.05))

    def forward(self, x):
        z = self.time_down(F.silu(self.time_fuse(x)*(1+self.low_hz_)))
        z = self.t2(self.t1(z)).mean(-1)
        return self.cls_head(torch.tanh(self.projection(z)))


class Base(nn.Module):
    def __init__(self):
        super().__init__(); self.id_backbone = Identity()

    def forward(self, x):
        return self.id_backbone(x)


class Model(nn.Module):
    def __init__(self, seed):
        super().__init__(); self.base = Base(); self.multilevel = MultiLevel(self.base, seed)

    def forward(self, x):
        return self.base(x)


def _config(seed):
    return dict(model_seed=seed, rounds=36, strengths=[.3, 1.],
        level_sets=[['time_down'], ['t2.pool'], list(LEVELS)],
        multilevel=dict(enabled=True, fishr_levels=list(LEVELS), fishr_estimator='cross', batch_size=120,
            pair_batch_size=60, pair_quota=2, fishr_fixed_denominator=2, fishr_calibration_pairs=12,
            fishr_ratio=.2, fishr_eps=1e-12, fishr_cap=10000., interval=4))


def _fixture():
    g = torch.Generator().manual_seed(77109)
    return dict(role='L_s', x=torch.randn(360, 2, 64, generator=g),
        y=torch.arange(6).repeat_interleave(20).repeat(3),
        rx=torch.tensor([1, 3, 4, 6, 8]).repeat_interleave(4).repeat(18),
        day=torch.arange(1, 4).repeat_interleave(120), ids=list(range(360)))


def run(device='cpu'):
    torch.set_num_threads(2); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.manual_seed(12011); seed=2026092701; model=Model(seed).to(device); L=_fixture(); c=_config(seed)
    assert cosine(torch.tensor([1.]), torch.tensor([2.])) == 1.
    assert cosine(torch.tensor([1.]), torch.tensor([-2.])) == -1.
    assert cosine(torch.tensor([0.]), torch.tensor([2.])) is None
    assert cosine(torch.tensor([1., 0.]), torch.tensor([0., 2.])) == 0.
    groups=_groups(model)
    assert {id(p) for p in groups['full_identity']} == {id(p) for p in model.base.id_backbone.parameters() if p.requires_grad}
    assert any(id(p) == id(model.base.id_backbone.low_hz_) for p in groups['full_identity'])
    model.eval(); a,b,_,_=model.multilevel.pair_indices(L, 4000004); ix=torch.cat((a,b))
    x=L['x'][ix].to(device); y=L['y'][ix].to(device); rx=L['rx'][ix]; days=L['day'][ix]; ids=[L['ids'][i] for i in ix.tolist()]
    clean=model(x); clean_grad=_flat_grad(F.cross_entropy(clean,y), groups['full_identity'])
    hooks_before=sum(len(m._forward_hooks) for m in model.modules())
    for p,eta in ((1.,0.),(0.,1.)):
        with style_hooks(model,y,rx,days,ids,121,LEVELS,eta,p=p) as info:
            unchanged=model(x)
        assert torch.equal(clean,unchanged)
        assert torch.equal(clean_grad,_flat_grad(F.cross_entropy(unchanged,y), groups['full_identity']))
        assert info['hook_order']==list(LEVELS)
    with style_hooks(model,y,rx,days,ids,121,LEVELS,1.,p=1.) as info:
        changed=model(x)
    assert not torch.equal(clean,changed)
    changed_ce=F.cross_entropy(changed,y); clean_ce=F.cross_entropy(clean,y)
    torch.testing.assert_close(_flat_grad(changed_ce-clean_ce,groups['full_identity']),
        _flat_grad(changed_ce,groups['full_identity'])-clean_grad,rtol=1e-5,atol=1e-8)
    assert info['hook_order']==list(LEVELS) and all(v['changed_packets']>0 for v in info['levels'].values())
    assert all(v['legal_donor_assignments']==v['donor_assignments'] for v in info['levels'].values())
    try:
        with style_hooks(model,y,rx,days,ids,121,LEVELS,1.):
            raise RuntimeError('fixture exception')
    except RuntimeError as exc:
        assert str(exc)=='fixture exception'
    assert sum(len(m._forward_hooks) for m in model.modules())==hooks_before
    with TemporaryDirectory() as tmp:
        calibration=calibrate(c,model,L,device,tmp)
    assert calibration['state_unchanged']
    for role in ('U_s','U','V','target'):
        try:audit(model,dict(L,role=role),c,device)
        except ValueError:pass
        else:raise AssertionError('Forbidden role accepted')
    # Include a preexisting gradient and deliberately mixed module mode flags.
    model.train(); model.base.id_backbone.t2.bn.eval()
    model.base.id_backbone.cls_head.bias.grad=torch.ones_like(model.base.id_backbone.cls_head.bias)
    before={k:v.clone() for k,v in model.state_dict().items()}; modes=[m.training for m in model.modules()]
    grads=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
    trng=torch.random.get_rng_state().clone(); crng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    nrng=np.random.get_state(); prng=random.getstate()
    result=audit(model,L,c,device)
    assert result['status']=='COMPLETE' and len(result['records'])==36 and len(result['dropout_controls'])==3
    assert set(result['day_counts'].values())=={12}
    assert result['resources']==dict(model_forwards=369,source_packet_readouts=39960)
    assert all(len(r['styles'])==6 for r in result['records'])
    assert all(r['split_combined_clean_close'] for r in result['records'])
    assert all(r['same_mask_replay_exact'] for r in result['dropout_controls'])
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert [m.training for m in model.modules()]==modes
    assert torch.equal(trng,torch.random.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(crng,torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    assert random.getstate()==prng
    np_after=np.random.get_state(); assert nrng[0]==np_after[0] and np.array_equal(nrng[1],np_after[1]) and nrng[2:]==np_after[2:]
    assert all((p.grad is None and g is None) or (g is not None and torch.equal(p.grad,g)) for p,g in zip(model.parameters(),grads))
    assert sum(len(m._forward_hooks) for m in model.modules())==hooks_before
    for record in result['records']:
        assert record['baseline_directions']['full_identity']['norms']['Fishr_both']>0
        for row in record['styles']:
            v=row['directions']['full_identity']
            assert v['norms']['clean_CE']>0
            assert 'clean_CE|style_increment' in v['pairwise_cosine']
    json.dumps(result,allow_nan=False)
    return dict(status='PASS',device=device,finite_json=True,role_rejection=True,positive_negative_zero_cos=True,
        no_style_logits_gradients_exact=True,legal_duplicate_physical_donors=True,hook_order_and_exception_cleanup=True,
        full_identity_including_Hz=True,all_216_style_rows=True,three_day_dropout_replay=True,
        state_modes_RNG_and_existing_gradients_unchanged=True,resources=result['resources'],
        max_split_combined_difference=max(r['split_combined_clean_max_abs_difference'] for r in result['records']),
        calibration=calibration['fixed_coefficients'],summary=result['summary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--device',default='cpu'); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=run(args.device); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='summary'},allow_nan=False))
