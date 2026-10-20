import copy,json
import numpy as np
import pytest
from tools.analyze_pamf_r17_saved_fidelity import fidelity,source_arrays,VIEWS,READER


def values(decisions):
    logits=np.zeros((len(decisions),6));logits[np.arange(len(decisions)),decisions]=4
    return dict(logits=logits,features=np.ones((len(decisions),160)),iq=np.ones((len(decisions),2,256)))


def test_paired_identity_losses_are_not_hidden_by_unchanged_accuracy():
    teacher=values([0,1,1,0]);student=values([1,1,2,4]);student['iq'][[0,2,3]]+=.1
    result=fidelity(teacher,student,np.arange(4))
    assert result['teacher_correct']==result['student_correct']==2
    assert result['decision_changed']==3 and result['lost']==result['gained']==1 and result['net_correct_delta']==0
    assert result['both_wrong_changed']==1
    assert result['cases']['decision_unchanged']['saved_IQ_NMSE_to_R17_mean']==0
    assert np.isclose(result['cases']['decision_changed']['saved_IQ_NMSE_to_R17_mean'],.01)
    assert sum(v['count'] for v in result['confidence_bins'])==4
    assert sum(v['lost'] for v in result['confidence_bins'])==1


def test_identical_predictions_have_no_flips_and_empty_case_statistics_are_explicit():
    teacher=values([0,1,2]);result=fidelity(teacher,copy.deepcopy(teacher),np.arange(3))
    assert result['decision_changed']==result['lost']==result['gained']==0
    assert result['cases']['all']['feature_cosine_distance_to_R17_mean']<1e-12
    assert result['cases']['decision_changed']['teacher_confidence_mean'] is None
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('field,invalid',[('logits',np.full((3,6),np.nan)),('features',np.zeros((3,159))),('iq',np.zeros((3,2,255)))])
def test_incomplete_or_nonfinite_saved_predictions_are_rejected(field,invalid):
    teacher=values([0,1,2]);student=copy.deepcopy(teacher);student[field]=invalid
    with pytest.raises(ValueError):fidelity(teacher,student,np.arange(3))
    with pytest.raises(ValueError):fidelity(teacher,teacher,np.array([0,1,6]))


@pytest.mark.parametrize('bad', ['target','missing_view'])
def test_target_or_incomplete_source_metadata_rejected_before_prediction_or_label_arrays(tmp_path,monkeypatch,bad):
    folder=tmp_path/'row';(folder/'prediction').mkdir(parents=True)
    complete=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='source',count=1800,views=list(VIEWS),
        readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=READER,F_seed=2026092701)
    summary=dict(status='ANALYZED',role='source',count=1800,F_run=READER,F_seed=2026092701,original_default_head=True,
        all_predictions_before_truth=True,target_access=False,head_fit_updates=0)
    if bad=='target':complete['role']='target'
    else:complete['views'].pop()
    (folder/'prediction/complete.json').write_text(json.dumps(complete));(folder/'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('No prediction or label array open'))
    with pytest.raises(ValueError):source_arrays(folder)
