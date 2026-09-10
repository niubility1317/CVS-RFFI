"""Real train-loop smoke evidence over synthetic physical WiSig records.

CUDA tests are opt-in to keep the ordinary unit suite bounded. Run with
CORE90_SYNTHETIC_INTEGRATION=1 after selecting a free local device.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_core90_cross_response import make_fixture, run_synthetic_variant, run_resume_verification, run_u0_off_verification
from scripts.core90_cross_response_matrix import VARIANTS


def test_synthetic_fixture_has_real_distinct_physical_metadata(tmp_path):
    import torch
    from dataset_wisig import load_wisig_compact_pkl, WiSigCompactDataset
    path = tmp_path/'fixture.pkl'
    manifest = make_fixture(path, records_per_cell=32)
    dataset = WiSigCompactDataset(load_wisig_compact_pkl(str(path)),
                                 rx_keep=[0,1,2,3], day_keep=[0], equalized=1)
    assert manifest['synthetic'] and len(dataset) == 4*4*32
    x0, tx0, d0, meta0 = dataset[0]
    x1, tx1, d1, meta1 = dataset[1]
    assert x0.shape == (2,256) and torch.isfinite(x0).all()
    assert tx0 == tx1 and d0 == d1
    assert meta0['sig_i'] != meta1['sig_i']
    assert not torch.equal(x0, x1)
    ids = {(r.tx_i, r.rx_i, r.day_i, r.eq_i, r.sig_i) for r in dataset.index}
    assert len(ids) == len(dataset)
    assert {r.rx_i for r in dataset.index} == {0,1,2,3}
    assert {r.day_i for r in dataset.index} == {0}


@pytest.mark.skipif(os.environ.get('CORE90_SYNTHETIC_INTEGRATION') != '1',
                    reason='opt-in actual CORE90 CUDA train loop')
@pytest.mark.parametrize('variant', VARIANTS)
def test_real_training_entrypoint_all_variants(variant, tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('real-model integration uses available local CUDA')
    result = run_synthetic_variant(variant, tmp_path, device='cuda:0', epochs=2)
    assert result['status'] == 'TRAIN_EXECUTED'
    assert result['formal_experiment'] is False
    assert Path(result['checkpoint_path']).is_file()
    assert result['activation']


@pytest.mark.skipif(os.environ.get('CORE90_SYNTHETIC_INTEGRATION') != '1',
                    reason='opt-in actual CORE90 CUDA resume replay')
def test_real_u5_resume_exact_model_optimizer_auxiliary_and_rng(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('real-model integration uses available local CUDA')
    report = run_resume_verification(tmp_path, variant='U5', device='cuda:0')
    assert report['status'] == 'EXACT_REPLAY_VERIFIED'
    assert report['tensor_tolerance'] == dict(rtol=0,atol=0)
    assert set(report['negative_cases_rejected']) == {
        'variant','source_contract','checkpoint_selection','checkpoint_role'}


@pytest.mark.skipif(os.environ.get('CORE90_SYNTHETIC_INTEGRATION') != '1',
                    reason='opt-in actual CORE90 package-off baseline replay')
def test_real_u0_package_off_preserves_baseline_losses_gradients_and_state(tmp_path):
    if not __import__('torch').cuda.is_available():
        pytest.skip('real-model integration uses available local CUDA')
    report = run_u0_off_verification(tmp_path,device='cuda:0')
    assert report['status'] == 'U0_PACKAGE_OFF_EXACT_PARITY'
    assert report['baseline_loss_gradient_metrics'] > 0
