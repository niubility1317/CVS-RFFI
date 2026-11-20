import torch
from experiments.pamf_repl_source_adapt.early_label_anchor import anchor_revisions


def test_early_revisions_require_current_acceptance_disagreement_and_label_coverage_threshold():
    student=torch.tensor([1,1,1,3,1])
    teacher=torch.tensor([3,3,3,3,3])
    conf=torch.tensor([.9,.9,.96,.99,.9])
    sparse=torch.tensor([7,35,7,7,7])
    accepted=torch.tensor([True,True,True,True,False])
    assert anchor_revisions(student,teacher,conf,sparse,accepted,.8).tolist()==[True,False,True,False,False]
    assert anchor_revisions(student,teacher,conf,sparse,accepted,.9).tolist()==[False,False,True,False,False]
