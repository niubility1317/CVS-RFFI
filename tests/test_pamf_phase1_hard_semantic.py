import torch
from experiments.pamf_phase1_flow.hard_semantic import source_weights,VARIANCE_CUTOFF


def test_source_difficulty_weights_exclude_clean_replacement():
    variance=torch.tensor([1.,0.,VARIANCE_CUTOFF*.9,VARIANCE_CUTOFF*1.1])
    assert source_weights(variance,clean_prefix=1).tolist()==[1.,1.,1.,3.]
