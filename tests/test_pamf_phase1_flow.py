import numpy as np
import pytest
import torch

from experiments.pamf_phase1_flow import experiment as e


def test_checkpoint_preserves_inference_and_budget(tmp_path):
    assert e.STEPS == 14790
    model=e.make_model();path=tmp_path/'model.pt';commit='a'*40
    e.save(path,model,commit,0)
    restored=e.load(path,'cpu',commit,epoch=0)
    x=torch.randn(2,2,256)
    with torch.no_grad():torch.testing.assert_close(restored(x),x,rtol=1e-6,atol=1e-6)
    with pytest.raises(ValueError):e.load(path,'cpu','b'*40,epoch=0)


def test_score_rejects_incomplete_prediction_before_truth(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda root:(root,np.array([1,2])))
    folder=tmp_path/'flow/test_prediction';folder.mkdir(parents=True)
    e.write(folder/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=2,views=list(e.VIEWS),
            F_run=e.F_RUN,F_row=e.F_ROW,commit='x',TX_truth_reads=0,model_updates=0))
    np.save(folder/'physical_ids.npy',np.array([1,2]))
    with pytest.raises(FileNotFoundError) as err:e.score(tmp_path,tmp_path,'x')
    assert '.raw.npy' in str(err.value)
