import pytest
from experiments.pamf_phase1_flow import dura_trisig_baseline as d


def test_baseline_rejects_different_reader_before_iq_access(tmp_path,monkeypatch):
    monkeypatch.setattr(d.core,'RUN',d.core.RUN)
    prior=tmp_path/'runs'/d.PRIOR/'flow/test_prediction';prior.mkdir(parents=True)
    d.write(prior/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=168000,views=list(d.core.VIEWS),
            F_run='wrong',F_row=d.core.F_ROW,TX_truth_reads=0,model_updates=0))
    with pytest.raises(ValueError,match='same-reader raw'):d.predict(tmp_path,tmp_path,'x')
