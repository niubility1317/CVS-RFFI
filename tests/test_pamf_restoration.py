"""Focused protocol, numerical, diffusion and original-PAMF interface tests."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'code')]
from experiments.pamf_restoration.models import build, NAMES, ConditionalDiffusion, ConditionalFlow
from experiments.pamf_restoration.views import PAMFViews, RestoredPAMF, objective, metrics, scalar_aligned
from experiments.pamf_restoration.data import IQSource, fixed_noise

torch.set_num_threads(2)


@pytest.mark.parametrize('name', NAMES)
def test_candidate_real_objective_and_finite_gradient(name):
    torch.manual_seed(7)
    model = build(name)
    x = torch.randn(3, 2, 256)*.6
    y = x + .1*torch.randn_like(x)
    extractor = PAMFViews()
    if isinstance(model, (ConditionalDiffusion, ConditionalFlow)):
        estimate, auxiliary = model.training_prediction(x, y)
    else:
        estimate, auxiliary = model(y), 0
    loss, parts = objective(estimate, x, extractor)
    (loss+auxiliary).backward()
    assert estimate.shape == x.shape and torch.isfinite(estimate).all()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    assert all(p.grad is None for p in extractor.parameters())
    assert set(parts) == {'raw_iq', 'aligned_iq', 'log_fft', 'frequency', 'pa', 'sinc', 'highfreq', 'reference'}


def test_views_are_original_native_pamf_inputs_and_adapter_preserves_head():
    extractor = PAMFViews().eval()
    x = torch.randn(2, 2, 256)
    views = extractor(x)
    backbone = extractor.probe.core.id_backbone
    assert torch.equal(views['frequency'], backbone._mirror_compressed_features(x)[0])
    assert torch.equal(views['pa'], backbone.pa_lift(x))
    assert torch.equal(views['sinc'], backbone._sinc_on_iq(x))
    assert torch.equal(views['reference'], extractor.probe.response(x))
    model = build('rescnn').eval()
    adapter = RestoredPAMF(model, extractor.probe).eval()
    original_head = extractor.probe.core.id_backbone.cls_head
    before = {k: v.clone() for k, v in extractor.probe.state_dict().items()}
    with torch.no_grad():
        expected = extractor.probe(model(x))
        actual = adapter(x)
    assert torch.equal(actual, expected) and actual.shape == (2, 6)
    assert adapter.pamf.core.id_backbone.cls_head is original_head
    assert all(torch.equal(v, before[k]) for k, v in extractor.probe.state_dict().items())


def test_diffusion_conditioning_noise_is_per_sample_and_no_state_update():
    model = build('diffusion').eval()
    # A trained conditional denoiser must actually depend on received IQ.
    torch.nn.init.normal_(model.denoiser.out.weight, std=.02)
    y = torch.randn(2, 2, 256)
    noise = fixed_noise(['a', 'b'], 123, y.shape, 'cpu')
    assert torch.equal(noise[1:], fixed_noise(['b'], 123, (1, 2, 256), 'cpu'))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        output = model(y, noise)
        partitioned = torch.cat([model(y[i:i+1], noise[i:i+1]) for i in range(2)])
        other_condition = model(torch.zeros_like(y), noise)
    assert torch.allclose(output, partitioned, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(output, other_condition)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match='fixed noise'):
        model(y, None)


def test_absolute_and_scalar_aligned_metrics_cannot_be_conflated():
    x = torch.randn(2, 2, 256)
    y = torch.stack((-2*x[:, 1], 2*x[:, 0]), 1)
    extractor = PAMFViews()
    m = metrics(y, x, extractor)
    assert m['iq_nmse'].min() > 1
    assert m['aligned_iq_nmse'].max() < 1e-8
    assert torch.isfinite(scalar_aligned(torch.zeros_like(x), x)).all()


def test_source_loader_never_opens_TX_labels_and_rejects_target(tmp_path, monkeypatch):
    manifest = dict(status='READY', native_role_ids='EXACT_MATCH',
                    source_receivers=[1, 3, 4, 6, 8], source_days=[1, 2, 3])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    for role in ('L_s', 'U_s', 'V'):
        folder = tmp_path/role
        folder.mkdir()
        for name, array in dict(iq=np.ones((2, 2, 256), np.float32), physical_ids=np.array(['a', 'b']),
                                rx_i=np.array([1, 3]), day_i=np.array([1, 2])).items():
            np.save(folder/(name+'.npy'), array, allow_pickle=False)
        # Even available labeled-role files must never be opened.
        (folder/'labels.npy').write_bytes(b'forbidden')
    load = np.load
    opened = []
    def guarded(path, *args, **kwargs):
        assert 'label' not in str(path) and 'truth' not in str(path) and 'target' not in str(path)
        opened.append(Path(path).name)
        return load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    for role in ('L_s', 'U_s', 'V'):
        source = IQSource(tmp_path, role)
        assert source.iq.shape == (2, 2, 256)
    assert set(opened) == {'iq.npy', 'physical_ids.npy', 'rx_i.npy', 'day_i.npy'}
    with pytest.raises(ValueError, match='Source roles only'):
        IQSource(tmp_path, 'query')


def test_score_refuses_partial_predictions(tmp_path):
    from experiments.pamf_restoration.runner import score_row
    prediction = tmp_path/'prediction'
    prediction.mkdir()
    (prediction/'complete.json').write_text(json.dumps(dict(status='PARTIAL', views=['clean'])), encoding='utf-8')
    with pytest.raises(ValueError, match='All registered predictions'):
        score_row(tmp_path/'not_read', tmp_path)


def test_independent_full_scorer_joins_source_labels_after_fixed_predictions(tmp_path):
    from experiments.pamf_restoration.runner import score_row
    from experiments.pamf_restoration.data import SCENES
    reference = tmp_path/'reference'
    folder = tmp_path/'row'
    prediction = folder/'prediction'
    source = tmp_path/'source'
    reference.mkdir()
    prediction.mkdir(parents=True)
    (source/'V').mkdir(parents=True)
    rng = np.random.default_rng(6)
    clean = rng.normal(size=(6, 2, 256)).astype(np.float32)*.5
    ids = np.array(['id'+str(i) for i in range(6)])
    labels = np.arange(6)
    np.save(reference/'physical_ids.npy', ids)
    np.save(reference/'rx.npy', np.ones(6, np.int64))
    np.save(reference/'day.npy', np.ones(6, np.int64))
    np.save(prediction/'physical_ids.npy', ids)
    np.save(source/'V/physical_ids.npy', ids)
    np.save(source/'V/labels.npy', labels)
    features = rng.normal(size=(6, 160)).astype(np.float32)
    for view in ('clean', *SCENES):
        np.save(reference/(view+'.npy'), clean)
        np.save(prediction/(view+'.npy'), clean)
        for mode, guesses in (('restored', labels), ('identity', (labels+1)%6)):
            np.save(prediction/f'{view}.cleanF.{mode}.logits.npy', np.eye(6, dtype=np.float32)[guesses])
            np.save(prediction/f'{view}.cleanF.{mode}.features.npy', features)
    (prediction/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE', count=6, candidate='fixture',
        views=['clean', *SCENES], frozen_readers=['cleanF'])), encoding='utf-8')
    score_row(reference, folder, source_cache=source)
    result = json.loads((folder/'metrics.json').read_text())
    assert result['prediction_complete_before_truth'] and result['training_TX_labels_read'] == 0
    assert result['scorer_TX_labels_read'] == 6
    assert result['views']['practical_low_urban']['classification']['cleanF']['net_repaired'] == 6
    assert result['views']['clean']['classification']['cleanF']['accuracy'] == 100


def test_gpu_capacity_counts_other_owners_and_run_reservations():
    from experiments.pamf_restoration.dispatcher import available_cards
    state = {0: dict(free_mb=20000, pids={1, 2}),
             1: dict(free_mb=11000, pids=set()),
             5: dict(free_mb=20000, pids={3})}
    assert available_cards(state, [0, 1, 5], {}) == [5]
    assert available_cards(state, [0, 1, 5], {'ours': dict(gpu=5)}) == []


def test_full_complex_native_channel_is_reproducible_without_metadata_inputs():
    from experiments.pamf_restoration.data import corrupt_one
    raw = np.random.default_rng(7).normal(size=(2, 256)).astype(np.float32)
    item = (0, raw, 'practical_low_urban', 'public-id', 'public-session', 'train', 19)
    a, b = corrupt_one(item), corrupt_one(item)
    assert a[0] == b[0] == 0 and a[1].shape == (2, 256)
    assert np.array_equal(a[1], b[1]) and np.isfinite(a[1]).all()
