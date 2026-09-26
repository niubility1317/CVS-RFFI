import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from tools.comparison_final_eval import build_data, score, metrics
from tools import comparison


def test_final_builder_is_disjoint_truth_separated_and_real_residual(tmp_path):
    raw=dict(tx_list=['a','b'],rx_list=['source','target'],capture_date_list=['d'],equalized_list=[1],
        data=[[[[np.random.default_rng(tx+rx).normal(size=(12,256,2)).astype('float32')]] for rx in range(2)] for tx in range(2)])
    contract=tmp_path/'contract.json'
    contract.write_text(json.dumps(dict(source_rxs=[0],role_ids={'L_s':['unrelated'],'U_s':[],'V':[]})))
    spec=dict(execution={'remote_run_root':str(tmp_path)},data={'dataset':'unused','contract_ref':str(contract),
        'target_receivers':[1],'target_days':[0]},evaluation_seed=3,augmentation_seed=4)
    with patch('dataset_wisig.load_wisig_compact_pkl',return_value=raw):
        build_data(spec)
    capsule=tmp_path/'data/capsule'
    index=np.load(capsule/'index.npz')
    assert len(index['ids'])==24 and len(set(index['ids']))==24
    assert set(index.files)=={'ids','scenes'}
    assert not (capsule/'truth.json').exists()
    clean=np.load(capsule/'clean.npy')
    sat=np.load(capsule/'satellite.npy')
    assert sat.shape==clean.shape==(24,2,256)
    assert np.isfinite(sat).all() and not np.array_equal(clean,sat)
    assert json.loads((capsule/'manifest.json').read_text())['one_satellite_observation_per_id']
    other=tmp_path/'other'
    other.mkdir()
    spec['execution']['remote_run_root']=str(other)
    spec['data']['target_receivers']=[0]
    with patch('dataset_wisig.load_wisig_compact_pkl',return_value=raw),pytest.raises(ValueError,match='overlap'):
        build_data(spec)


def test_score_gate_precedes_truth_and_metrics_are_correct(tmp_path):
    with pytest.raises(ValueError,match='All predictions'):
        score(dict(execution={'remote_run_root':str(tmp_path)},rows=[{'output_root':str(tmp_path/'missing')}]))
    result=metrics(np.array([0,0,1,1]),np.array([0,1,1,1]),2)
    assert result['accuracy']==.75
    assert result['macro_f1']==pytest.approx((2/3+4/5)/2)


def test_prepare_defaults_and_output_collision(tmp_path,monkeypatch):
    # Keep genuine presets but relocate generated files to an isolated fixture.
    import shutil
    for rel in [f'automation_reports/CV-SincNet/{comparison.SOURCE}/experiment.json',
                'configs/baselines_practical_20260927/poster-ce-s392005.json']:
        dest=tmp_path/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,dest)
    monkeypatch.setattr(comparison,'ROOT',tmp_path)
    monkeypatch.setattr(comparison,'prepare_evaluation',lambda *a:None)
    monkeypatch.setattr(comparison.subprocess,'run',lambda *a,**kw:None)
    args=argparse.Namespace(run_id='test-run',methods='poster-ce',seeds='392005,123',gpus='0',epochs=200,batch_size=128,name=None,owner='test')
    comparison.prepare(args)
    spec=json.loads((tmp_path/'configs/comparisons/test-run/registration.json').read_text(encoding='utf-8'))
    assert spec['final_evaluation']['enabled'] and spec['final_evaluation']['checkpoint']=='last.pt'
    assert spec['final_evaluation']['views']==['clean','practical_high','practical_mid','practical_low_urban']
    assert len(spec['rows'])==2
    assert all(row['epochs']==200 for row in spec['rows'])
    with pytest.raises(FileExistsError):
        comparison.prepare(args)
