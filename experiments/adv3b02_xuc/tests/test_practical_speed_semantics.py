"""Execution-only optimizations: channel identity, RNG, and disabled work."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import cvsrffi.practical_adapter as adapter

VARIANTS = [('full', False, 'zf'), ('full', True, 'zf'),
            ('full', True, 'mmse'), ('residual', False, 'zf')]


def arguments(tmp_path, variant):
    route, eq, method = variant
    return SimpleNamespace(practical_route=route, practical_equalization=eq,
                           practical_equalizer_method=method, practical_fs_hz=25e6,
                           practical_fc_hz=2.462e9, practical_receiver_seed=2027,
                           practical_eval_cache_dir=str(tmp_path), output_dir=str(tmp_path / 'logs'))


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('scene', adapter.PRACTICAL[1:])
def test_all_scenes_exact_cache_hit_and_rng(tmp_path, variant, scene, monkeypatch):
    args = arguments(tmp_path, variant)
    x = torch.randn(3, 2, 256, generator=torch.Generator().manual_seed(4))
    adapter.set_evaluation_context(['a', 'b', 'c'])
    gen = torch.Generator().manual_seed(17)
    y, meta = adapter.apply_practical(x, scene, args, gen=gen, return_meta=True)
    assert adapter._last_meta['cache_event'] == 'miss'
    saved = gen.get_state().clone()
    def forbidden(*a, **kw):
        raise AssertionError('Cache hit must not simulate the channel')
    monkeypatch.setattr(adapter, 'apply_leo_practical_channel_batch', forbidden)
    z, other = adapter.apply_practical(x, scene, args, gen=gen, return_meta=True)
    assert adapter._last_meta['cache_event'] == 'hit'
    assert torch.equal(y, z) and torch.equal(saved, gen.get_state())
    assert all(torch.equal(meta[k], other[k]) for k in meta)
    z, other = adapter.apply_practical(x, scene, args, gen=gen, return_meta=False)
    assert torch.equal(y, z) and other is None


def test_cache_identity_invalidation_and_dynamic_bypass(tmp_path):
    args = arguments(tmp_path, VARIANTS[0])
    x = torch.randn(2, 2, 256)
    ids = ['a', 'b']
    def run(seed=7):
        return adapter.apply_practical(x, 'practical_high', args,
            gen=torch.Generator().manual_seed(seed), return_meta=True)
    adapter.set_evaluation_context(ids);run()
    x[0, 0, 0] += .1;run()
    assert adapter._last_meta['cache_event'] == 'miss'
    adapter.set_evaluation_context(list(reversed(ids)));run()
    assert adapter._last_meta['cache_event'] == 'miss'
    args.practical_equalization = True;run()
    assert adapter._last_meta['cache_event'] == 'miss'
    meta = dict(base_index=[1, 2], rx_i=[1, 1], day_i=[0, 0])
    adapter.set_source_evaluation_context(meta)
    a, _ = run(9);run(9)
    assert adapter._last_meta['cache_event'] == 'hit'
    b, _ = run(10)
    assert adapter._last_meta['cache_event'] == 'miss' and not torch.equal(a, b)
    before = len(list(tmp_path.rglob('*.npz')))
    # Training never caches even when the same seed is explicitly requested.
    for epoch in (1, 2):
        adapter.set_training_context(meta, epoch, 'U');run(9)
        assert adapter._last_meta['cache_event'] == 'bypass'
    assert len(list(tmp_path.rglob('*.npz'))) == before


def test_source_cache_preserves_generator_draws(tmp_path):
    args = arguments(tmp_path, VARIANTS[0]);x = torch.randn(2, 2, 256)
    meta = dict(base_index=[1, 2], rx_i=[1, 1], day_i=[0, 0])
    adapter.set_source_evaluation_context(meta)
    expected = None
    for i in range(2):
        gen = torch.Generator().manual_seed(23)
        adapter.apply_practical(x, 'practical_mid', args, gen=gen)
        if expected is None: expected = gen.get_state().clone()
        else: assert torch.equal(expected, gen.get_state())
    assert adapter._last_meta['cache_event'] == 'hit'


@pytest.mark.parametrize('damage', ['bytes', 'null', 'missing_fields'])
def test_invalid_cache_recomputed(tmp_path, damage):
    args = arguments(tmp_path, VARIANTS[0]);x = torch.randn(2, 2, 256)
    adapter.set_evaluation_context(['a', 'b'])
    gen = torch.Generator().manual_seed(3)
    y, _ = adapter.apply_practical(x, 'practical_high', args, gen=gen)
    path = next(tmp_path.rglob('*.npz'))
    if damage == 'bytes':
        path.write_bytes(b'broken cache')
    else:
        with np.load(path, allow_pickle=False) as entry:
            content = {key: entry[key] for key in entry.files}
        content['metadata'] = np.frombuffer(('null' if damage == 'null' else '[{},{}]').encode(), dtype=np.uint8)
        np.savez(path, **content)
    z, _ = adapter.apply_practical(x, 'practical_high', args, gen=gen)
    assert torch.equal(y, z) and adapter._last_meta['cache_event'] == 'invalid_recomputed'


def test_unused_source_satellite_pass_removed_without_changing_geometry(tmp_path, monkeypatch):
    from scripts.train_rc4_practical import native
    cfg = ROOT / 'configs/rc4_practical_full_noeq_20260918.json'
    from scripts.train_rc4_practical import build_args
    args = build_args(cfg, 'unused', str(tmp_path), 'unused', 'unused', 'unused', 'test')
    args.direct_metric_multiview_separate = False
    args.sat_train_protocol_scenario_list = list(adapter.PRACTICAL[1:])
    args.eval_max_batches = 0
    # Compare the previous implementation of the exact function, not a second
    # handwritten copy of the expected geometry calculation.
    old = (ROOT / 'tests/reference/source_val_geometry_064aae40.py').read_text(encoding='utf-8')
    fn = next(n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef) and n.name == '_evaluate_source_val_tail_geometry')
    env = dict(vars(native));exec(compile(ast.Module(body=[fn], type_ignores=[]), '<reference>', 'exec'), env)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.calls = 0
        def forward(self, x, **kw):self.calls += 1;return {'z_id':x.flatten(1)[:, :4]}
    batches = [(torch.randn(2,2,256),torch.tensor([0,1]),[torch.tensor([0,0]),{'base_index':[1,2],'rx_i':[1,1],'day_i':[0,0]}])]
    def move(batch, device):return batch
    def domains(*a):return torch.tensor([0,0])
    def loss(z,y,d,**kw):return z.sum()*0,{'active':1.,'checksum':float(z.sum())}
    def apply(x,*a,**kw):return x+.1,None
    for name, value in [('move_batch',move),('domain_from_extra',domains),('direct_metric_acceptance_loss',loss),('apply_sat_channel_for_scenario',apply)]:
        env[name] = value;monkeypatch.setattr(native,name,value)
    ctx={'val_loader':batches,'domain_label_map':{0:0}}
    model = Model();state=torch.get_rng_state().clone()
    reference=env['_evaluate_source_val_tail_geometry'](model,ctx,torch.device('cpu'),args)
    assert model.calls == 2
    model.calls=0
    optimized=native._evaluate_source_val_tail_geometry(model,ctx,torch.device('cpu'),args)
    assert model.calls == 1 and model.training and torch.equal(state,torch.get_rng_state())
    assert optimized['checksum'] == reference['checksum']
    assert optimized['multiview_sample_count'] == 0 and not optimized['satellite_geometry_requested']


def test_real_model_geometry_state_and_rng_parity(tmp_path):
    from scripts.train_rc4_practical import build_args, native
    from torch.utils.data import Dataset, DataLoader
    args = build_args(ROOT/'configs/rc4_practical_full_noeq_20260918.json',
                      'unused',str(tmp_path),'unused','unused','unused','test')
    args.direct_metric_multiview_separate = False
    args.sat_train_protocol_scenario_list = list(adapter.PRACTICAL[1:])
    args.eval_max_batches = 0
    ma=native.merge_checkpoint_args({'model':None,'args':{},'stats':{},'split_info':None},args,input_len=256,num_domains=15)
    model=native.build_baseline_model(native._apply_model_cli_args(ma,args),torch.device('cpu'))
    class Source(Dataset):
        def __init__(self):self.x=torch.randn(12,2,256)
        def __len__(self):return 12
        def __getitem__(self,i):return self.x[i],i%6,0,{'base_index':i,'rx_i':1,'day_i':1}
    ctx={'val_loader':DataLoader(Source(),batch_size=4),'domain_label_map':{0:0}}
    old=(ROOT/'tests/reference/source_val_geometry_064aae40.py').read_text(encoding='utf-8')
    env=dict(vars(native));exec(compile(old,'<reference>','exec'),env)
    before={k:v.clone() for k,v in model.state_dict().items()}
    # DataLoader itself consumes one global base_seed per iterator in both paths.
    initial=torch.get_rng_state().clone()
    ref=env['_evaluate_source_val_tail_geometry'](model,ctx,torch.device('cpu'),args)
    after_reference=torch.get_rng_state().clone()
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    torch.set_rng_state(initial)
    result=native._evaluate_source_val_tail_geometry(model,ctx,torch.device('cpu'),args)
    assert torch.equal(after_reference,torch.get_rng_state()) and model.training
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    for key in ref:
        if key!='multiview_sample_count':assert json.dumps(ref[key],sort_keys=True)==json.dumps(result[key],sort_keys=True),key
