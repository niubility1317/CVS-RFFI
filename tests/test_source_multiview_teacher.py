import pytest
import torch
from experiments.phase1_dura_orthogonal.source_multiview_teacher import build_teacher


def test_teacher_uses_matching_records_and_repeats_in_view_order():
    count=2
    clean=torch.stack([torch.full((2,256),float(i)) for i in (1,2)]*3)
    anchor=torch.stack([torch.full((2,256),float(i)) for i in (1,10,4,40,7,70)])
    teacher=build_teacher(anchor,clean,count)
    assert torch.equal(teacher[:,0,0],torch.tensor([4.,40.,4.,40.,4.,40.]))


def test_teacher_rejects_cross_record_view_misalignment():
    clean=torch.stack([torch.full((2,256),float(i)) for i in (1,2)]*3)
    clean[2],clean[3]=clean[3].clone(),clean[2].clone()
    with pytest.raises(ValueError):
        build_teacher(torch.randn_like(clean),clean,2)
