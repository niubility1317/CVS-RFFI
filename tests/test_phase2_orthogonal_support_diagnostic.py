import torch

from experiments.pamf_phase2_practical.clean_orthogonal_support_diagnostic import summarize_scores


def test_support_diagnostic_distinguishes_old_restricted_from_open_class_decision():
    scores=torch.tensor([[.2,.1,.9],[.1,.2,.8]])
    labels=torch.tensor([0,1])
    result=summarize_scores(scores,labels,old_count=2)
    assert result['old_only_correct']==2 and result['all_correct']==0
    assert result['predicted_new']==2 and result['count']==2


def test_affine_mopc_anchor_is_rejected_as_original_orthogonal_bridge():
    import pytest
    from experiments.pamf_phase2_practical.clean_orthogonal_support_diagnostic import run
    job=dict(methods=['mopc_hr'],save_B_for_mopc=True,points=[[5,20]],mopc_bridge='proto_affine_v1')
    with pytest.raises(ValueError,match='original bridge'):
        run({},job,5,20,'cpu')
