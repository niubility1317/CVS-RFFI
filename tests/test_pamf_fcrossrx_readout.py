import copy
from pathlib import Path
import sys
import json

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'code')]
from experiments.pamf_fcrossrx_readout import frozen as f
from experiments.pamf_fcrossrx_readout import source_diagnostic as diagnostic
from experiments.standard_ce_baselines.sigtime import SIGTimeCVS


def metadata():
    seed = f.SEEDS[0]
    config = dict(run_id=f.REFERENCE_RUN, row_id=f'CROSS_RX-s{seed}',
                  model_seed=seed, arm='CROSS_RX', epochs=50, label_epochs=33,
                  pseudo_epochs=17, updates=11100, checkpoint_selection='final_only',
                  separate_head_training=False)
    initial = dict(scratch_only=True, checkpoint_sources=[], ancestors=[],
                   target_contact=False, source_roles='EXACT_MATCH', seed=seed)
    done = dict(config=config, status='SOURCE_TRAINED', epoch=50, label_epochs=33,
                pseudo_epochs=17, logged_steps=11100, optimizer_steps=11100,
                head_fit_updates=0, checkpoint_sources=[], ancestors=[],
                initialization='scratch', target_access=False, target_evaluated=False,
                checkpoint='original/source/final_ssdg.pth')
    freeze = dict(status='SOURCE_FROZEN', config=config, epoch=50, selection='final_only',
                  target_access=False, checkpoint=done['checkpoint'])
    args = dict(seed=seed, epochs=50, label_epochs=33, pseudo_epochs=17,
                label_smoothing=.01, use_aug=True, use_ema_teacher=True,
                use_concat_sat_channel_aug=False, baseline_ckpt='', teacher_ckpt='',
                batch_size=128, max_grad_norm=5., pseudo_temporal_mode='batch_neighbor',
                phase1_lr_schedule='cosine', labeled_ratio=.07, unlabeled_ratio=.63,
                source_val_ratio=.3)
    aux = dict(status='VERIFIED', mode='CROSS_RX', calls=11100, applied_calls=9990,
               head_fit_updates=0, U_truth_read=False, default_head_jointly_trained=True)
    return [initial, done, freeze, args, aux, seed]


@pytest.mark.parametrize('part,key,value', [
    (0, 'ancestors', ['unknown_parent']), (0, 'target_contact', True),
    (1, 'head_fit_updates', 1), (1, 'target_access', True), (1, 'epoch', 20),
    (2, 'selection', 'best_target'), (3, 'use_concat_sat_channel_aug', True),
    (3, 'use_ema_teacher', False), (3, 'baseline_ckpt', 'unknown.pt'),
    (4, 'U_truth_read', True), (4, 'mode', 'ALL_TX'),
])
def test_reject_wrong_lineage_recipe_or_head(part, key, value):
    evidence = metadata()
    f.validate_metadata(*evidence)
    evidence[part][key] = value
    with pytest.raises(ValueError, match='CHECKPOINT_PROVENANCE_UNVERIFIED'):
        f.validate_metadata(*evidence)


def test_original_coordinates_and_default_head_survive_strict_load():
    torch.set_num_threads(2)
    native = SIGTimeCVS(f.SEEDS[0])
    native.sequence_pool = f.TripleSignaturePool(96)
    original_sinc = native.core.id_backbone.sinc
    with torch.no_grad():
        original_sinc.low_hz_.add_(2.5)
        original_sinc.band_hz_.add_(1.25)
    native.eval().requires_grad_(False)
    converted = copy.deepcopy(native)
    converted.core.id_backbone.sinc = f.FrozenCenteredSinc(original_sinc)
    with torch.no_grad():
        converted.core.id_backbone.sinc.low_coordinate.add_(.01)
        original_sinc.low_hz_.add_(250.)
    state = {'id_backbone.encoder.'+k: v for k, v in converted.state_dict().items()}
    loaded = f.from_state(state, f.SEEDS[0])
    x = torch.randn(3, 2, 256)
    before = {k: v.clone() for k, v in loaded.state_dict().items()}
    with torch.inference_mode():
        assert torch.equal(native(x), loaded(x))
        singles = torch.cat([loaded(v[None]) for v in x])
        torch.testing.assert_close(loaded(x), singles, atol=1e-4, rtol=1e-4)
        assert torch.equal(loaded(x).argmax(1), singles.argmax(1))
    assert all(not p.requires_grad for p in loaded.parameters())
    assert all(torch.equal(v, before[k]) for k, v in loaded.state_dict().items())
    damaged = dict(state)
    del damaged['id_backbone.encoder.core.id_backbone.cls_head.weight']
    with pytest.raises(RuntimeError):
        f.from_state(damaged, f.SEEDS[0])


def test_payload_rejects_different_run_and_recipe_but_allows_runtime_representation():
    evidence = metadata()
    config, args = f.validate_metadata(*evidence)
    payload = dict(run_id=f.REFERENCE_RUN, candidate_id=config['row_id'], epoch=50,
                   checkpoint_selection='final_only', a1_ema_successful_updates=11100,
                   args=dict(args, unused_runtime_field=float('nan')))
    f.validate_checkpoint(payload, config, args)
    payload['args']['epochs'] = 20
    with pytest.raises(ValueError):
        f.validate_checkpoint(payload, config, args)
    payload['args']['epochs'] = 50
    payload['candidate_id'] = 'ALL_TX-s2026092701'
    with pytest.raises(ValueError):
        f.validate_checkpoint(payload, config, args)


def test_source_score_rejects_incomplete_views_before_label_open(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    ids = np.array(['source0', 'source1'])
    monkeypatch.setattr(diagnostic, 'source_reference', lambda *_: (ids, None, np.array([0, 1])))
    (tmp_path/'complete.json').write_text(json.dumps(dict(
        status='SOURCE_PREDICTIONS_COMPLETE', role='source_V', views=['clean'],
        readouts=list(diagnostic.READOUTS), count=2,
        target_reads=0, TX_label_reads=0, state_updates=0)), encoding='utf-8')
    opened = []
    monkeypatch.setattr(np, 'load', lambda *a, **kw: opened.append(a[0]))
    with pytest.raises(ValueError, match='All fixed source predictions'):
        diagnostic.score(tmp_path, tmp_path/'forbidden', tmp_path, tmp_path/'metrics.json')
    assert opened == []


def test_source_prediction_rejects_negative_decisions(tmp_path):
    ids = np.array(['source0', 'source1'])
    (tmp_path/'complete.json').write_text(json.dumps(dict(
        status='SOURCE_PREDICTIONS_COMPLETE', role='source_V', views=list(diagnostic.VIEWS),
        readouts=list(diagnostic.READOUTS), count=2,
        target_reads=0, TX_label_reads=0, state_updates=0)), encoding='utf-8')
    for view in diagnostic.VIEWS:
        values = {k: np.array([0, 1], np.int64) for k in diagnostic.READOUTS}
        if view == 'clean':
            values['residual'][0] = -1
        np.savez(tmp_path/(view+'.npz'), ids=ids, **values)
    with pytest.raises(ValueError, match='Malformed default-head'):
        diagnostic.prediction_arrays(tmp_path, ids)
