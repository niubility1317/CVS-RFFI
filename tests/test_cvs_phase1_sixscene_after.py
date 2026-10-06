from types import SimpleNamespace
import json
import numpy as np
import pytest
from experiments.cvs_phase1_stack import sixscene_after as s


def test_query_cannot_start_before_parent_completion(monkeypatch):
    monkeypatch.setattr(s,'original',lambda:object())
    monkeypatch.setattr(s,'parent_ready',lambda d:False)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Query touched before parent'))
    with pytest.raises(RuntimeError,match='query remains closed'):s.predict('anything')


def test_waiting_parent_and_failure_are_distinct(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'BASE',tmp_path/'seven_views')
    (tmp_path/'mixed').mkdir()
    d=SimpleNamespace(BASE=str(tmp_path))
    assert s.parent_ready(d) is False
    (tmp_path/'mixed/failure.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Parent technical failure'):s.parent_ready(d)


def test_full_sixscene_manifest_rejects_three_scene_subset(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'VIEWS_ROOT',tmp_path)
    d=SimpleNamespace(CLASSES=['a','b','c','d','e','f'],CAPSULE='/capsule',read=lambda p:json.loads(p.read_text()))
    m=dict(status='VALIDATED_ONCE',count=168000,scenes=list(s.SCENES),classes=d.CLASSES,truth_read=False,channel='residual/post_sync/noeq',source_capsule=d.CAPSULE,clean_ref='/capsule/clean.npy',rows_share_observations=True)
    s.write(tmp_path/'manifest.json',m);assert s.manifest(d)==m
    m['scenes']=m['scenes'][:3];s.write(tmp_path/'manifest.json',m)
    with pytest.raises(ValueError,match='sixscene capsule'):s.manifest(d)


def test_all_seven_predictions_required_before_truth(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'BASE',tmp_path);monkeypatch.setattr(s,'VIEWS_ROOT',tmp_path)
    monkeypatch.setattr(s,'manifest',lambda d:None)
    ids=np.asarray(['q'+str(i) for i in range(168000)])
    np.savez(tmp_path/'index.npz',ids=ids)
    out=tmp_path/'row/prediction';out.mkdir(parents=True)
    done=dict(status='PREDICTIONS_COMPLETE',count=168000,views=list(s.VIEWS),truth_read=False,query_fit=False)
    s.write(out/'complete.json',done)
    d=SimpleNamespace(rows=lambda:[dict(row_id='row')],read=lambda p:json.loads(p.read_text()))
    values={v:np.zeros(168000,dtype=np.int8) for v in s.VIEWS}
    np.savez(out/'predictions.npz',ids=ids,**values)
    assert np.array_equal(s.prediction_preflight(d),ids)
    values.pop(s.VIEWS[-1]);np.savez(out/'predictions.npz',ids=ids,**values)
    with pytest.raises(ValueError,match='identity mismatch'):s.prediction_preflight(d)
