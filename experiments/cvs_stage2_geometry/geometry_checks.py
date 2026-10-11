"""Focused algebra, exact classifier-interface and source-readout checks."""
import json
import torch
from .features import identity
from .geometry_audit import scatter,spectrum,ridge_readout


def run():
    from experiments.cvs_stage2_decouple import design as old
    from experiments.cvs_stage2_decouple.common import parameter_groups
    from experiments.cvs_stage2_decouple.deterministic_forward import deterministic_identity
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context
    torch.set_num_threads(2);torch.manual_seed(9173)
    y=torch.arange(6).repeat_interleave(12);rx=torch.arange(2).repeat(36);day=torch.arange(3).repeat_interleave(2).repeat(12)
    z=torch.nn.functional.one_hot(y,6).double()+.01*torch.randn(len(y),6,dtype=torch.double)
    sx,sr=scatter(z,y,rx,day)
    stats,v,u=spectrum(sx,sr+1e-3*torch.eye(6,dtype=torch.double))
    assert stats['rank']==5 and stats['max_eigen_residual']<1e-8
    got=ridge_readout(z,y,z,y);assert got['correct']==len(y)
    # Duplicating an entire RX/day/TX group cannot change the equal-group means.
    idx=(y==0)&(rx==0)&(day==0)
    doubled=scatter(torch.cat((z,z[idx])),torch.cat((y,y[idx])),torch.cat((rx,rx[idx])),torch.cat((day,day[idx])))[0]
    assert torch.allclose(sx,doubled,atol=1e-12)
    c=old.config(next(r for r in old.rows() if r['arm']=='C0'))
    args=old.make_args(c,'cpu');args.num_domains=5;args.input_len=256
    with numerical_context(old.FULL_FP32_POLICY),installed(c) as native:
        model=native.build_baseline_model(args,torch.device('cpu'))
        parameter_groups(model);model.train();x=torch.randn(4,2,256)
        with deterministic_identity(model):
            logits,z160=identity(model,x,verify=True)
            assert z160.shape==(4,160) and torch.allclose(z160.norm(dim=1),torch.ones(4),atol=2e-6)
            loss=z160[:,0].sum();loss.backward()
            head=model.id_backbone.encoder.core.id_backbone.cls_head
            assert head.weight.grad is None  # feature constraint reaches E, not label head
            grads=[p.grad for p in model.id_backbone.encoder.core.parameters() if p.grad is not None]
            assert grads and sum(float(g.square().sum()) for g in grads)>0
    return dict(status='PASS',equal_RX_day_TX_group_weighting=True,TX_rank=stats['rank'],
        generalized_eigen_residual=stats['max_eigen_residual'],actual_joint160_reproduces_logits=True,
        joint_feature_gradient_to_encoder=True,target_access=False,checkpoint_reads=0)


if __name__=='__main__':print(json.dumps(run()))
