import torch
from student_scale import scale_with_bound

def test_scaled_correction_preserves_global_l2_bound():
    torch.manual_seed(13);parent=torch.randn(4,2,256);student=parent+.01*torch.randn_like(parent)
    torch.testing.assert_close(scale_with_bound(parent,student,0),parent)
    for scale in (1,4,8):
        y=scale_with_bound(parent,student,scale);assert torch.all((y-parent).flatten(1).norm(dim=1)<=.020001*parent.flatten(1).norm(dim=1))
