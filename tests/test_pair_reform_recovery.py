"""Regressions for the two observed E11 failures, using real parser and CUDA AMP."""
import copy
import pytest
import torch
from model_dual_cvsincnet import DualCVSincNetDisentangle
from cvsrffi.pair_reform_runtime import compute_pair_batch, _fixed_head_weights
from test_pair_reform_training import args_for


def test_fixed_head_check_inside_autocast():
    torch.set_num_threads(2)
    model=DualCVSincNetDisentangle(num_classes=3,num_domains=2,input_len=256).eval()
    with torch.no_grad():
        outputs=model.forward_identity_only(torch.randn(2,2,256),domain_labels=torch.zeros(2,dtype=torch.long))
    # The old matmul was recast even though its inputs explicitly called float().
    with torch.autocast('cpu',dtype=torch.bfloat16):
        assert _fixed_head_weights(model,outputs).dtype==torch.float32


@pytest.mark.parametrize('mode,tangent,route',[('safe',0.,0.),('point',.035,0.),('point',0.,.05),('point',.035,.05)])
def test_real_parser_cuda_amp_active_pair_backward(mode,tangent,route):
    if not torch.cuda.is_available():
        pytest.skip('CUDA needed for actual fp16 regression')
    torch.set_num_threads(2)
    torch.manual_seed(392005)
    model=DualCVSincNetDisentangle(num_classes=3,num_domains=2,input_len=256).cuda()
    teacher=copy.deepcopy(model).eval().requires_grad_(False)
    x=torch.randn(2,2,256,device='cuda'); domains=torch.zeros(2,dtype=torch.long,device='cuda')
    args=args_for(mode); args.pair_tangent_weight=tangent; args.pair_route_weight=route
    args.pair_direction_ratio=1.
    assert not hasattr(args,'sat_fs_hz')
    with torch.autocast('cuda',dtype=torch.float16):
        out=model(torch.cat([x,x+.02]),return_aux=True,domain_labels=domains.repeat(2))
        split=[{k:v[s:s+2] for k,v in out.items() if torch.is_tensor(v) and v.ndim and len(v)==4} for s in (0,2)]
        result=compute_pair_batch(model=model,teacher=teacher,clean=x,channel=x+.02,
            student_clean=split[0],student_channel=split[1],domains=domains,
            labels=split[0]['tx_logits'].argmax(-1),metadata={'snr_db':torch.full((2,),25.,device='cuda')},
            args=args,epoch=11,physical_ids=[1,2])
        loss=result['loss']+out['tx_logits'].float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert not any(p.grad is not None for p in teacher.parameters())
    assert ('tangent' in result['weighted_components'])==bool(tangent)
    assert ('route' in result['weighted_components'])==bool(route)
