import pytest
from experiments.cvs_readout_attribution import publish

@pytest.mark.parametrize('output',[b'{"existing": []}\n','{"existing": []}\n'])
def test_exclusive_path_preflight_accepts_transport_bytes_and_text(monkeypatch,output):
    monkeypatch.setattr(publish,'ssh',lambda script:output)
    assert publish.verify_exclusive_paths(['/p/run'])=={'existing':[]}

@pytest.mark.parametrize('output',[b'{"existing": ["/p/run"]}',b'{}'])
def test_exclusive_path_preflight_rejects_present_or_unverified_paths(monkeypatch,output):
    monkeypatch.setattr(publish,'ssh',lambda script:output)
    with pytest.raises(FileExistsError):publish.verify_exclusive_paths(['/p/run'])
