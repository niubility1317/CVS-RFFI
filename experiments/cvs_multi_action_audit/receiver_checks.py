"""Focused mathematical and implementation counterexamples, no dataset reads."""
import json
import torch
from .receiver import ContributionStatistics, packet_time_statistics, anchor_loss, relation_loss


def run():
    torch.manual_seed(713)
    n=torch.arange(256)
    z=torch.exp(1j*.03*n)
    shifted=z*torch.exp(1j*.012*n)
    def iq(v):return torch.stack((v.real,v.imag))
    x=torch.stack((iq(z),iq(-z),iq(shifted),iq(-shifted)))
    time=packet_time_statistics(x)
    assert torch.equal(x[:2].mean(0),x[2:].mean(0))
    assert torch.allclose(time[0],time[1],atol=1e-6)
    assert torch.allclose(time[2],time[3],atol=1e-6)
    assert float((time[:2].mean(0)-time[2:].mean(0)).norm())>.1
    pool=ContributionStatistics(max_age_steps=888,min_count=1)
    views=torch.zeros(32,49);h=torch.ones(32,349)
    pool.update(torch.zeros(32,dtype=torch.long),torch.zeros(32,dtype=torch.long),views,h,0,'v0')
    for step in (800,1600,2400,3200):
        pool.update(torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long),views[:1],2*h[:1],step,'v0')
    groups=pool.groups(3200,'v0')
    assert next(iter(groups.values()))['count']==2
    assert torch.equal(next(iter(groups.values()))['h'],2*h[0])
    pool.update(torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long),views[:1],3*h[:1],3201,'v1')
    assert torch.equal(next(iter(pool.groups(3201,'v1').values()))['h'],3*h[0])
    assert torch.equal(next(iter(pool.groups(3201,'v0').values()))['h'],2*h[0])
    ages=pool.age_report(3201)
    balanced=ContributionStatistics(min_count=1)
    labels=torch.arange(12).repeat(2);rx=torch.arange(2).repeat_interleave(12)
    balanced.update(labels,rx,torch.zeros(24,49),torch.randn(24,349),0,'v')
    gen=torch.Generator().manual_seed(891)
    for _ in range(12):balanced.sample_pairs(balanced.groups(0,'v'),gen,8)
    coverage=balanced.coverage()
    assert coverage['unique_tx']==list(range(12))
    assert coverage['unique_group_relations']==24
    counts=list(coverage['uses_by_group_relation'].values())
    assert max(counts)-min(counts)<=1
    # Common rotation preserves pair geometry but changes fixed class anchors.
    anchors=torch.eye(3,requires_grad=True)
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    moved=(anchors.detach()@rotation).requires_grad_(True)
    old=relation_loss(anchors,moved);new=anchor_loss(moved,anchors,anchors)
    assert float(old.detach())==0 and float(new.detach())>0
    new.backward()
    assert float(moved.grad.norm())>0 and anchors.grad is None
    return dict(status='VERIFIED',phase_cancellation_preserves_CFO=True,
        expiration_not_refreshed=ages,reference_version_no_mixing=True,
        balanced_sampling=coverage,common_rotation_relation_loss=float(old.detach()),
        common_rotation_anchor_loss=float(new.detach()),frozen_anchors_gradient=True)


def runtime_check():
    from .receiver import receiver_audit, ReceiverAction, _fixed
    from experiments.cvs_multi_disentangle.checks import identity,source_fixture
    from experiments.cvs_multi_disentangle import design
    from experiments.cvs_multi_disentangle.model import intermediate,identity_from_intermediate
    torch.set_num_threads(1)
    config=design.config(design.rows()[0])
    model=identity(config,'cpu').id_backbone.eval()
    x,y,rx=source_fixture()
    g=torch.Generator().manual_seed(991)
    audit_x=x+.0001*torch.randn(x.shape,generator=g)
    state={k:v.clone() for k,v in model.state_dict().items()}
    report=receiver_audit(model,x,y,rx,audit_x,y,rx,g,steps=2,
        fit_ids=['fit-'+str(i) for i in range(len(x))],
        audit_ids=['audit-'+str(i) for i in range(len(x))])
    assert report['report']['status']=='VERIFIED',report['report']
    assert len(report['step_logs'])==2
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert all(p.grad is None for p in model.parameters())
    assert report['report']['relation_vs_anchor']
    try:
        receiver_audit(model,x,y,rx,audit_x,y,rx,g,steps=1,
            fit_ids=list(range(len(x))),audit_ids=list(range(len(x))))
    except ValueError as error:
        assert 'overlap' in str(error)
    else:raise AssertionError('Overlapping physical IDs accepted')
    with _fixed(model):
        with torch.no_grad():h=intermediate(model,x[:4])
        branch=ReceiverAction()
        delta=branch(torch.randn(4,49),torch.randn(4,49),h)
        loss=(identity_from_intermediate(model,h+delta)-identity_from_intermediate(model,h).detach()).square().mean()
        loss.backward()
        mass=sum(float(p.grad.norm()) for p in branch.parameters() if p.grad is not None)
        assert mass>0
        assert all(p.grad is None for p in model.parameters())
    return dict(status='VERIFIED',physical_overlap_rejected=True,identity_unchanged=True,
        differentiable_frozen_G_gradient_mass=mass,fit_steps=2,
        audit_relations=report['report']['audit_group_relations'])


def jensen_check():
    from unittest.mock import patch
    from .receiver import _error_report
    a=torch.tensor([[1.,0.],[3.,0.]])
    b=torch.tensor([[1.,0.],[5.,0.]])
    h0,h1=a.mean(0,keepdim=True),b.mean(0,keepdim=True)
    # Nonlinear G(h)=h^2 has different mean G and G(mean h), including
    # different group margin changes despite perfect mean-h prediction.
    pool=ContributionStatistics(min_count=2)
    packets=torch.cat((a,b));z=packets.square()
    pool.update(torch.zeros(4,dtype=torch.long),torch.tensor([0,0,1,1]),
        torch.zeros(4,49),packets,0,'frozen',z=z,margin=z[:,0]-z[:,1])
    pairs=pool.all_pairs(pool.groups(0,'frozen'))
    p=next(p for p in pairs if p['source'][1]==0)
    with patch('experiments.cvs_multi_action_audit.receiver.identity_from_intermediate',lambda _,h:h.square()), \
         patch('experiments.cvs_multi_action_audit.receiver.classify_intermediate',lambda _,h:h.square()):
        m=_error_report(None,h0,h1,h1-h0,torch.zeros(1,dtype=torch.long),[p])
    assert m['centroid_G_mse']==0
    assert m['centroid_margin_delta_mae']==0
    assert m['predicted_centroid_vs_destination_mean_G_mse']==8
    assert m['predicted_centroid_delta_vs_group_mean_G_delta_mse']==4.5
    assert m['predicted_centroid_margin_delta_vs_mean_packet_margin_delta_mae']==3
    assert m['source_centroid_vs_mean_G_gap_mse']==.5
    assert m['destination_centroid_vs_mean_G_gap_mse']==8
    assert m['group_mean_packet_margin_actual_delta_abs_mean']==8
    assert m['centroid_margin_actual_delta_abs_mean']==5
    return dict(status='VERIFIED',mean_packet_statistics_preserved=True,
                centroid_and_group_metrics_separate=True,metrics=m)


if __name__=='__main__':
    print(json.dumps(dict(mathematical=run(),runtime=runtime_check(),jensen=jensen_check()),indent=2))
