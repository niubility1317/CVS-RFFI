import json
import numpy as np
import pytest
from tools import analyze_pamf_r17_task_saved_predictions as m


def values(decisions):
    logits=np.zeros((len(decisions),6));logits[np.arange(len(decisions)),decisions]=4
    return dict(logits=logits,features=np.ones((len(decisions),160)),iq=np.ones((len(decisions),2,256)))


def summary(decisions,labels):
    matrix=np.zeros((6,6),np.int64);np.add.at(matrix,(labels,decisions),1)
    return dict(count=len(labels),records=[dict(view=v,readout='restored',dimension='overall',value='all',
        confusion=matrix.tolist(),accuracy=100*matrix.trace()/len(labels)) for v in m.d.VIEWS])


def test_equal_score_keeps_lost_and_gained_samples_and_partitions_visible():
    labels=np.arange(4);a=values([0,1,1,0]);b=values([1,1,2,4])
    output=m.paired_views({v:a for v in m.d.VIEWS},{v:b for v in m.d.VIEWS},labels,np.array([0,0,1,1]),
        np.array([1,2,1,2]),summary(a['logits'].argmax(1),labels),summary(b['logits'].argmax(1),labels))
    for row in output.values():
        assert row['lost']==row['gained']==1 and row['net_correct_delta']==0 and row['decision_changed']==3
        for groups in row['groups'].values():assert sum(v['lost'] for v in groups.values())==1


def test_saved_predictions_must_match_existing_independent_score():
    labels=np.arange(4);a=values([0,1,2,3]);wrong=summary([1,1,2,3],labels)
    with pytest.raises(ValueError,match='independent score'):
        m.paired_views({v:a for v in m.d.VIEWS},{v:a for v in m.d.VIEWS},labels,np.zeros(4),np.zeros(4),wrong,wrong)


@pytest.mark.parametrize('bad', ['Faccess','missing_view','wrong_role'])
def test_frozen_metadata_rejected_before_any_saved_arrays(tmp_path,monkeypatch,bad):
    folder=tmp_path/'row';(folder/'prediction').mkdir(parents=True)
    done=dict(status='PREDICTIONS_COMPLETE',kind=m.d.KIND,role='source',count=1800,views=list(m.d.VIEWS),
        readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=m.d.F_RUN,F_seed=m.d.F_SEED)
    score=dict(status='ANALYZED',role='source',count=1800,F_run=m.d.F_RUN,F_seed=m.d.F_SEED,original_default_head=True,
        all_predictions_before_truth=True,target_access=False,head_fit_updates=0,training_F_access=True,training_TX_labels_read=0)
    if bad=='Faccess':score['training_F_access']=False
    elif bad=='missing_view':done['views'].pop()
    else:done['role']='target'
    (folder/'prediction/complete.json').write_text(json.dumps(done));(folder/'summary.json').write_text(json.dumps(score))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('No array open after invalid header'))
    with pytest.raises(ValueError):m.saved_arrays(folder,'source',task_knowledge=True)


def test_an_incomplete_candidate_keeps_all_truth_closed(monkeypatch,tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    run=tmp_path/m.d.RUN;run.mkdir();original=tmp_path/m.d.PARENT_RUN;original.mkdir()
    (run/'launch.json').write_text(json.dumps(dict(commit=m.METHOD)))
    (original/'launch.json').write_text(json.dumps(dict(commit=m.d.PARENT_COMMIT)))
    (run/'complete.json').write_text(json.dumps(dict(status='ANALYZED',source_rows=list(m.d.ARMS),target_rows=[])))
    (run/'source_decision.json').write_text(json.dumps(dict(status='SOURCE_DECISION_FROZEN',selected=None,target_rows=[],target_feedback=False,target_reads_before_decision=0)))
    calls=[]
    def arrays(path,*a,**k):
        calls.append(path)
        if m.d.ARMS[-1] in path.parts:raise ValueError('Last candidate incomplete')
        return np.arange(1800),{},{}
    monkeypatch.setattr(m,'saved_arrays',arrays)
    monkeypatch.setattr(m,'source_truth',lambda *a,**k:pytest.fail('Source truth must remain closed'))
    with pytest.raises(ValueError,match='Last candidate incomplete'):m.analyze_saved(tmp_path,tmp_path/'cache',tmp_path/'truth.json')
    assert len(calls)==3


def test_unselected_target_is_rejected_without_truth(monkeypatch,tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    run=tmp_path/m.d.RUN;run.mkdir();original=tmp_path/m.d.PARENT_RUN;original.mkdir()
    (run/'launch.json').write_text(json.dumps(dict(commit=m.METHOD)))
    (original/'launch.json').write_text(json.dumps(dict(commit=m.d.PARENT_COMMIT)))
    (run/'complete.json').write_text(json.dumps(dict(status='ANALYZED',source_rows=list(m.d.ARMS),target_rows=[])))
    (run/'source_decision.json').write_text(json.dumps(dict(status='SOURCE_DECISION_FROZEN',selected=None,target_rows=[],target_feedback=False,target_reads_before_decision=0)))
    (run/m.d.ARMS[0]/'target_confirmation').mkdir(parents=True)
    monkeypatch.setattr(m,'saved_arrays',lambda *a,**k:(np.arange(1800),{},{}))
    monkeypatch.setattr(m,'source_truth',lambda *a,**k:pytest.fail('No truth before target selection check'))
    with pytest.raises(ValueError,match='Unselected'):m.analyze_saved(tmp_path,tmp_path/'cache',tmp_path/'truth.json')
