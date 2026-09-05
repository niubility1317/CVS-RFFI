"""The EMA teacher must classify with the weights used by safety geometry."""
import copy
import torch
import torch.nn.functional as F
from model import CosFaceHead
from SSDG.train_ssdg import _update_ema_model
from model_dual_cvsincnet import DualCVSincNetDisentangle
from cvsrffi.pair_reform_runtime import _fixed_head_weights


def test_ema_update_invalidates_warmed_cosface_cache():
    torch.manual_seed(7)
    student=CosFaceHead(16,3)
    teacher=copy.deepcopy(student).eval()
    x=torch.randn(8,16)
    with torch.no_grad():
        teacher(x)  # Populate the normalized-weight cache before EMA updates.
        for step in range(4):
            student.weight.add_(torch.randn_like(student.weight)*.01)
            _update_ema_model(teacher,student,.5)
            expected=F.linear(F.normalize(x,dim=-1,eps=1e-4),F.normalize(teacher.weight,dim=-1,eps=1e-4))*teacher.s
            torch.testing.assert_close(teacher(x),expected)


def test_safe_teacher_after_real_dual_ema_updates():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    student=DualCVSincNetDisentangle(num_classes=3,num_domains=2,input_len=256)
    teacher=copy.deepcopy(student).eval().requires_grad_(False)
    x=torch.randn(2,2,256); domains=torch.zeros(2,dtype=torch.long)
    with torch.no_grad():
        teacher.forward_identity_only(x,domain_labels=domains)
        student.id_backbone.cls_head.head.weight.add_(.01*torch.randn_like(student.id_backbone.cls_head.head.weight))
        _update_ema_model(teacher,student,.5)
        outputs=teacher.forward_identity_only(x,domain_labels=domains)
        _fixed_head_weights(teacher,outputs)
