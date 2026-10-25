import pytest
import torch
from experiments.pamf_channel_pl.consensus import policies,summarize


def logits(classes=6):
    z=torch.zeros(3,4,classes)
    z[:,0,0]=11.;z[0,1,1]=11.;z[1:,1,0]=11.;z[:,2,2]=11.;z[:2,3,0]=11.
    return z


@pytest.mark.parametrize('classes',[6,10])
def test_consensus_is_not_proof_of_correctness_and_peer_gate_excludes_own_model(classes):
    p=policies(logits(classes))
    assert p['unanimous3']['accepted'].tolist()==[True,False,True,False]
    assert p['mean_probability3']['accepted'].tolist()==[True,False,True,False]
    assert p['peer_pair_for_0']['pseudo'][1]==0 and p['peer_pair_for_0']['accepted'][1]
    assert p['single_0']['pseudo'][1]==1
    assert p['peer_pair_for_2']['accepted'][3]
    y=torch.tensor([0,0,3,0]);s=summarize(p['unanimous3'],y)
    assert s['accepted']==2 and s['correct']==1 and s['precision_percent']==50.


def test_predicted_class_precision_is_distinct_from_true_class_grouping():
    p=policies(logits())['single_0'];y=torch.tensor([0,0,3,0])
    predicted=summarize(p,y,group=p['pseudo']==2)
    truth=summarize(p,y,group=y==3)
    assert predicted['accepted']==truth['accepted']==1 and predicted['correct']==0
    assert summarize(p,y,group=p['pseudo']==3)['accepted']==0


@pytest.mark.parametrize('bad',['parents','classes','empty','nonfinite'])
def test_rejects_incomplete_or_invalid_teacher_outputs(bad):
    z=logits()
    if bad=='parents':z=z[:2]
    elif bad=='classes':z=z[:,:,:3]
    elif bad=='empty':z=z[:,:0]
    else:z[0,0,0]=float('nan')
    with pytest.raises(ValueError):policies(z)


@pytest.mark.parametrize('bad',['visible_GPU','GPU_policy','run'])
def test_CPU_diagnostic_rejects_GPU_or_foreign_run_before_opening_inputs(monkeypatch,bad):
    from experiments.pamf_channel_pl.consensus_source import RUN,dispatch
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    rt=dict(run_id=RUN,allowed_gpus=[])
    if bad=='visible_GPU':monkeypatch.setenv('CUDA_VISIBLE_DEVICES','5')
    elif bad=='GPU_policy':rt['allowed_gpus']=[5]
    else:rt['run_id']='foreign'
    with pytest.raises(ValueError):dispatch(rt)
