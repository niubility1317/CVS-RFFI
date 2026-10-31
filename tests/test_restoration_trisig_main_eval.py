from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.pamf_phase3_restoration import main_split_eval as m


def identity():
    return dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,
                source_run=m.F_RUN,source_row=m.F_ROW,source_code_commit=m.F_COMMIT)


@pytest.mark.parametrize('field,value',[('parameters',334721),('source_code_commit','4df7-old-F50'),('source_run','Phase3_PCD')])
def test_wrong_classifier_base_rejected_even_when_frozen(field,value):
    i=identity();i[field]=value
    with pytest.raises(ValueError,match='Tri-SIG-TIME SSL'):
        m.valid_identity(i)


def test_one_finished_population_cannot_publish_all_complete_or_open_truth(monkeypatch):
    a=SimpleNamespace(root=Path('root'),commit='a'*40)
    calls=[]
    def read(path):
        calls.append(path)
        if 'cross_both' in str(path):raise FileNotFoundError('Other population incomplete')
        return dict(status='PREDICTIONS_COMPLETE',run_id=m.RUN,task='cross_rx',count=126000,
            code_commit=a.commit,G_checkpoint_commit=m.G_COMMIT,F0_identity=identity(),
            readouts=list(m.STREAMS),views=list(m.VIEWS),truth_read=False,query_fit=False,model_state_updates=0)
    monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(m,'write',lambda *a:pytest.fail('Partial matrix must not publish ALL_COMPLETE'))
    with pytest.raises(FileNotFoundError):m.finalize(a)
    assert len(calls)==2 and all(p.name=='complete.json' for p in calls)


def test_scene_fail_and_clean_damage_remain_fail_despite_good_mean():
    rows=[]
    for task in m.COUNTS:
        for stream in m.STREAMS:
            for index,view in enumerate(m.VIEWS):
                correct=80
                if stream=='TF_IQ' and index>0:correct=70 if index==6 else 95
                if stream=='TF_F0' and index==0:correct=79
                cm=np.zeros((6,6),np.int64);cm[0,0]=correct;cm[0,1]=100-correct
                rows.append(dict(test=task,readout=stream,view=view,dimension='overall',stratum='ALL',confusion=cm.tolist()))
    result=m.summary({'results':rows})
    for task in result:
        iq=result[task]['acceptance']['TF_IQ'];guided=result[task]['acceptance']['TF_F0']
        assert iq['scenes_passing_3pp']==5 and not iq['performance_passed']
        assert guided['all_six_passed'] and not guided['clean_preserved'] and not guided['performance_passed']
