from pathlib import Path
import numpy as np
import pytest
from tools import analyze_pamf_r17_target_saved_predictions as m


def setup(monkeypatch,*,broken=None):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');runs=Path('/runs');truth=Path('/truth.json');ids=np.arange(168000);opened=[];arrays=[]
    class Truth:
        def __getitem__(self,key):return dict(label=int(key)%6,receiver=int(key)%7,day=int(key)%4)
    def read(path):
        if path==truth:
            assert len(arrays)==3 and opened==['formal_ids']
            opened.append('truth');return Truth()
        if path.name=='launch.json':return dict(commit=('wrong' if broken=='method' else m.METHOD) if path.parent.name==m.d.RUN else m.d.PARENT_COMMIT)
        if path.name=='complete.json':return dict(status='ANALYZED',target_rows=list(m.d.ARMS)[:1] if broken=='matrix' else list(m.d.ARMS),new_training_updates=0,target_feedback=False)
        if path.name=='comparison_plan.json':return m.d.configuration()
        if path.name=='manifest.json':return dict(status='FORMAL_REFERENCE_READY',count=168000,views=list(m.d.VIEWS))
        pytest.fail('Unexpected file '+str(path))
    def saved(path,role,**kwargs):
        assert role=='target'
        arrays.append(path)
        if broken=='last_prediction' and len(arrays)==3:raise ValueError('Missing final view')
        return ids,{},{}
    def load(path,**kwargs):
        assert path.name=='physical_ids.npy' and len(arrays)==3
        opened.append('formal_ids');return ids+1 if broken=='ids' else ids
    monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'saved_arrays',saved);monkeypatch.setattr(m.np,'load',load)
    monkeypatch.setattr(m,'paired_views',lambda a,b,labels,rx,day,*s:dict(count=len(labels),TX=len(np.unique(labels)),RX=len(np.unique(rx)),days=len(np.unique(day))))
    return runs,truth,opened,arrays


def test_all_fixed_rows_are_validated_before_truth_and_both_diagnostics_retained(monkeypatch):
    runs,truth,opened,arrays=setup(monkeypatch)
    result=m.analyze_saved_target(runs,truth)
    assert opened==['formal_ids','truth'] and len(arrays)==3
    assert set(result['rows'])==set(m.d.ARMS) and result['rows'][m.d.ARMS[0]]==dict(count=168000,TX=6,RX=7,days=4)
    assert result['model_forwards']==result['raw_IQ_reads']==result['training_updates']==0


@pytest.mark.parametrize('broken',['method','matrix','last_prediction','ids'])
def test_incomplete_or_mismatched_predictions_keep_truth_closed(monkeypatch,broken):
    runs,truth,opened,arrays=setup(monkeypatch,broken=broken)
    with pytest.raises(ValueError):m.analyze_saved_target(runs,truth)
    assert 'truth' not in opened
