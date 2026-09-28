"""Synthetic frozen cache compatibility, provenance and read-only boundaries."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
import d92_orbit_feature_cache as mod
from cvsrffi.stage2_d92_orbit_shared import FROZEN_CONFIG
from test_export_d92_bnna_features import fixture, dump
import export_d92_bnna_features as exporter


def inputs(tmp_path, monkeypatch, k=1):
    args, _, _ = fixture(tmp_path, monkeypatch, k)
    with threadpool_limits(limits=1):
        exporter.export(**args)
    return {**{key: args[key] for key in ('row_root', 'capsule', 'expected_capsule_id', 'expected_checkpoint_sha256')},
        'orbit_features': args['output'], 'algorithm': copy.deepcopy(FROZEN_CONFIG)}


def test_real_producer_cache_readonly_without_iq_model_or_adaptation_access(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch)
    original = Path.open; opened = []
    allowed = {args['orbit_features'] / name for name in ('startup.json', 'features_complete.json', 'checkpoint_provenance.json', mod.CACHE_NAME)}
    allowed |= {args['row_root'] / 'd92_startup.json', args['row_root'] / 'received_features/checkpoint_provenance.json',
        args['capsule'] / 'received.npz', mod.PRODUCER_CONFIG}
    def guarded(path, *a, **kw):
        assert path in allowed
        opened.append(path); return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', guarded)
    getitem = np.lib.npyio.NpzFile.__getitem__
    def member(data, key):
        assert key != 'iq'
        return getitem(data, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', member)
    z, f, ids, marker, previous = mod.load_features(**args)
    assert z.shape == (7, 4, 160) and f.shape == (7, 96) and ids.tolist() == [f'physical-{i}' for i in range(7)]
    assert marker['model_seed'] == previous['seed'] == 123 and opened
    for value in (z, f):
        with pytest.raises(ValueError): value.flat[0] = 0
        with pytest.raises(ValueError): value.setflags(write=True)


@pytest.mark.parametrize('fault', ['phase_order', 'fft_floor', 'full_input', 'native_scope', 'schema', 'encoder',
    'truth', 'startup', 'inheritance', 'epoch', 'seed', 'classes', 'checkpoint', 'capsule', 'ids', 'extra', 'nonfinite', 'dtype'])
def test_mismatched_cache_rejected(tmp_path, monkeypatch, fault):
    args = inputs(tmp_path, monkeypatch)
    root = args['orbit_features']; marker = mod.read(root / 'features_complete.json')
    provenance = mod.read(root / 'checkpoint_provenance.json'); startup = mod.read(root / 'startup.json')
    if fault in ('phase_order', 'fft_floor', 'full_input'):
        key, value = {'phase_order': ('phases_quarter_turns', [0, 2, 1, 3]), 'fft_floor': ('fft_norm_floor', 1e-12),
            'full_input': ('input_shape', [2, 192])}[fault]
        # Consistent shape alone cannot authorize a different feature formula.
        marker['algorithm'][key] = value; startup['config']['algorithm'][key] = value
    elif fault == 'native_scope': marker['native_forward_scope'] = 'batch dependent normalization'
    elif fault == 'schema': marker['schema'] = 'other'
    elif fault == 'encoder': marker['encoder_updated'] = True
    elif fault == 'truth': marker['truth_read'] = True
    elif fault == 'startup': startup['query_fit_access'] = True
    elif fault == 'inheritance': provenance['checkpoint_inheritance'] = ['old']
    elif fault == 'epoch': provenance['checkpoint_epoch'] = 199
    elif fault == 'seed': marker['model_seed'] = 124
    elif fault == 'classes': marker['classes'] = marker['classes'][::-1]
    elif fault == 'checkpoint': args['expected_checkpoint_sha256'] = 'b' * 64
    elif fault == 'capsule': args['expected_capsule_id'] = 'other'
    else:
        path = root / mod.CACHE_NAME
        with np.load(path) as data: values = {key: data[key] for key in data.files}
        if fault == 'ids': values['ids'] = values['ids'][::-1]
        elif fault == 'extra': values['adapted_identity'] = values['identity_views']
        elif fault == 'nonfinite': values['fft'][0, 0] = np.nan
        elif fault == 'dtype': values['identity_views'] = values['identity_views'].astype(np.float64)
        np.savez(path, **values); marker['feature_file_bytes'] = path.stat().st_size
    dump(root / 'features_complete.json', marker); dump(root / 'checkpoint_provenance.json', provenance)
    dump(root / 'startup.json', startup)
    with pytest.raises(ValueError): mod.load_features(**args)


def test_consumer_formula_mismatch_is_rejected(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch); args['algorithm']['fft_norm_floor'] = 1e-12
    with pytest.raises(ValueError, match='feature formula'): mod.load_features(**args)


def test_runtime_import_has_no_previous_adaptation_or_torch_dependency():
    code = "import sys; sys.path[:0]=['tools','code']; import predict_d92_orbit_shared; assert 'torch' not in sys.modules; assert not any('stage2_d92_bnna' in k or 'predict_d92_bnna' in k for k in sys.modules)"
    subprocess.run([sys.executable, '-X', 'utf8', '-c', code], cwd=ROOT, check=True, capture_output=True, text=True)
