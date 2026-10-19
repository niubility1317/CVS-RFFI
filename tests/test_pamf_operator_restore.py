from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from leo_practical.channel import Config,receiver_for_session
from leo_practical.equalization import linear_filter
from experiments.pamf_operator_restore.operators import (
    filter_matrix,real_operator,real_pair,complex_pair,oracle_snapshot,GaussianSystem,estimate_action)
from experiments.pamf_operator_restore import runner


def test_finite_reflect_operator_and_widely_linear_coordinates_match_native():
    rng=np.random.default_rng(4);z=rng.normal(size=17)+1j*rng.normal(size=17)
    h=rng.normal(size=7)+1j*rng.normal(size=7);g=rng.normal(size=5)+1j*rng.normal(size=5)
    main=filter_matrix(h,17);image=filter_matrix(g,17)
    expected=linear_filter(z,h)[0]+linear_filter(z.conj(),g)[0]
    np.testing.assert_allclose(complex_pair(real_operator(main,image)@real_pair(z)),expected,atol=1e-12)
    np.testing.assert_allclose(main@z,linear_filter(z,h)[0],atol=1e-12)
    with pytest.raises(ValueError):real_operator(np.zeros((2,3)),np.zeros((3,2)))


@pytest.mark.parametrize('scene',['practical_high','practical_low_suburban','practical_mid_urban','practical_low_urban'])
def test_true_action_reconstructs_native_signal_and_colored_noise_without_engine_patch(scene):
    cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False)
    rng=np.random.default_rng(9);z=rng.normal(size=64)+1j*rng.normal(size=64);z/=np.sqrt(np.mean(abs(z)**2))
    receiver=receiver_for_session(cfg,2026101910,'source_RX1_day1')
    observed,a,c,meta=oracle_snapshot(z,cfg,99,receiver)
    assert observed.shape==(64,) and a.shape==(128,128) and c.shape==(64,2,2)
    assert np.linalg.eigvalsh(c).min()>0 and meta['oracle_signal_relative_error']<1e-9
    assert meta['observation_decomposition_relative_error']<2e-7
    assert meta['oracle_uses_true_AGC_lock_channel'] and not meta['estimated_methods_receive_oracle_fields']
    with pytest.raises(ValueError):oracle_snapshot(z,Config(fs_hz=25e6,scenario=scene,processing_route='full'),99,receiver)


def test_soft_covariance_update_matches_full_dense_gaussian_system_and_adjoint():
    rng=np.random.default_rng(2);n=6;a=rng.normal(size=(2*n,2*n));r=rng.normal(size=2*n);prior=rng.normal(size=2*n)
    b=rng.normal(size=(n,2,2));c=b@b.transpose(0,2,1)+np.eye(2)[None]*.3
    dense=np.zeros((2*n,2*n))
    for i in range(n):dense[np.ix_([i,n+i],[i,n+i])]=c[i]
    inverse=np.linalg.inv(dense);precision=3.7
    expected=np.linalg.solve(a.T@inverse@a+precision*np.eye(2*n),a.T@inverse@r+precision*prior)
    system=GaussianSystem(a,r,c)
    np.testing.assert_allclose(system.solve(prior,precision),expected,rtol=1e-11,atol=1e-11)
    u=rng.normal(size=2*n);v=rng.normal(size=2*n)
    assert np.dot(a@u,v)==pytest.approx(np.dot(u,a.T@v))
    assert 0<=system.geometry(precision)['effective_observed_dimensions']<=2*n
    assert np.linalg.norm(a@system.solve(prior,precision)-r)>1e-8
    with pytest.raises(ValueError):GaussianSystem(a,r,0.)
    with pytest.raises(ValueError):system.solve(prior,0.)


def test_received_only_estimation_includes_identity_and_preserves_exact_clean_without_scene_input():
    rng=np.random.default_rng(5);z=rng.normal(size=64)+1j*rng.normal(size=64)
    a,variance,details=estimate_action(z,z,(0,24))
    np.testing.assert_allclose(complex_pair(a@real_pair(z)),z,atol=1e-11)
    assert variance>0 and details['persistent_state_updates']==0 and not details['true_channel_clean_TX_inputs']
    system=GaussianSystem(a,real_pair(z),variance)
    np.testing.assert_allclose(complex_pair(system.solve(real_pair(z),1/runner.SCALE**2)),z,atol=1e-11)
    with pytest.raises(ValueError):estimate_action(z,z,(24,))
    with pytest.raises(ValueError):estimate_action(z,np.ones(63),(0,24))
    assert 0 in runner.LOCAL_TAPS and 24 in runner.LOCAL_TAPS and len(runner.LOCAL_TAPS)==10


def test_registered_panel_is_source_only_no_target_cli_or_identity_model_inputs(monkeypatch):
    monkeypatch.setattr(sys,'argv',['operator','--target-truth','forbidden'])
    with pytest.raises(SystemExit):runner.main()
    cfg=runner.configuration()
    assert cfg['TX_labels_read']==cfg['F_outputs_read']==cfg['target_reads']==cfg['training_updates']==0
    assert cfg['count']==60 and cfg['views']==['clean',*runner.SCENES]
    assert len(cfg['methods'])==7 and cfg['source_per_RX_day']==4
    assert runner.CAPACITY_LOCK=='receiver_residual_capacity4.lock'


def test_real_module_entrypoint_does_not_require_test_pythonpath():
    result=subprocess.run([sys.executable,'-m','experiments.pamf_operator_restore.runner','--help'],
        cwd=ROOT,env=dict(os.environ,PYTHONPATH=''),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert '--source-cache' in result.stdout and '--target-truth' not in result.stdout
