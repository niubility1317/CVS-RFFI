import pytest
import torch
from experiments.pamf_ssl_research.source_view_reliability import summarize


def probabilities(classes,confidence):
    q=torch.full((len(classes),6),(1-confidence)/5)
    for i,label in enumerate(classes):q[i,label]=confidence
    return q


def test_view_difficulty_does_not_relabel_raw_pseudo_correctness():
    raw=probabilities([0,1,2],.99);view=probabilities([0,3,2],.97);truth=torch.tensor([0,1,3])
    r=summarize(raw,view,'V',truth)
    assert r['raw_accepted']['accepted']==3 and r['raw_accepted']['raw_pseudo_correct']==2
    assert r['raw_accepted']['view_prediction_correct']==1 and r['raw_accepted']['same_input_view_agreement']==2
    gate=next(v for v in r['gates'] if v['augmented_confidence_threshold']==.95 and v['require_raw_label_agreement'])
    assert gate['accepted']==2 and gate['raw_pseudo_correct']==1


def test_U_truth_is_rejected_and_true_accuracy_stays_unknown():
    q=probabilities([0,1],.99);r=summarize(q,q,'U_s')
    assert r['raw_accepted']['raw_pseudo_correct'] is None and r['all_view_prediction_correct'] is None and not r['U_truth_read']
    with pytest.raises(ValueError):summarize(q,q,'U_s',torch.tensor([0,1]))
    with pytest.raises(ValueError):summarize(q,q,'target')


def test_empty_acceptance_reports_null_losses_and_precision():
    r=summarize(probabilities([0,1],.8),probabilities([0,1],.99),'V',torch.tensor([0,1]))
    assert r['raw_accepted']['accepted']==0 and r['raw_accepted']['hard_CE_to_raw_pseudo_mean'] is None
    assert r['raw_accepted']['raw_pseudo_precision_percent'] is None


@pytest.mark.parametrize('bad',[torch.zeros(2,6),torch.full((2,6),float('nan')),torch.full((2,5),.2)])
def test_malformed_probabilities_cannot_yield_a_source_claim(bad):
    with pytest.raises(ValueError):summarize(probabilities([0,1],.99),bad,'U_s')
