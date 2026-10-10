"""Focused numerical/data-separation regressions for receiver distribution risk."""
import json
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from . import receiver as r


class TinyIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(349,349)
        self.backend=nn.Linear(349,6)


class TinyAction(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(349,349,bias=False)
        nn.init.eye_(self.linear.weight)
    def forward(self,h,v0,v1):
        return self.linear(h)*(v1-v0).mean()


def fixture(online):
    """Full crossed physical groups, no shared IDs between roles or partitions."""
    online.reset('v1')
    for tx in range(6):
        for rx in range(2):
            for day in range(2):
                counts={(partition,role):0 for partition in range(2) for role in range(3)}
                candidate=0
                while min(counts.values())<4:
                    pid=f'physical-{tx}-{rx}-{day}-{candidate}';candidate+=1
                    digest=r._digest(pid);role=digest%3;partition=(digest//3)%2
                    if counts[(partition,role)]>=4:continue
                    counts[(partition,role)]+=1
                    h=torch.randn(349)*.2+rx*.1
                    online.entries[(pid,'clean')]=dict(pid=pid,y=tx,rx=rx,day=day,condition='clean',
                        role=role,partition=partition,step=0,h=h,v=torch.randn(49)+rx,
                        logits=torch.randn(6)+F.one_hot(torch.tensor(tx),6)*2)
                    if role==1 and partition==1:
                        online.entries[(pid,'clean')]['audit_x']=F.pad(h,(0,163)).reshape(2,256)
                    online.seen_ids.add(pid)


def checks(device='cpu'):
    torch.manual_seed(1901)
    from .actions import SharedActionCore
    hh=torch.randn(4,349,device=device);v0=torch.randn(1,49,device=device);v1=torch.randn(1,49,device=device)
    for width in (48,64):
        core=SharedActionCore(hidden=width).to(device)
        actual=r.ReceiverDistributionAction(shared_core=core).to(device)
        independent=r.ReceiverDistributionAction(shared_core=SharedActionCore(hidden=width)).to(device)
        independent.load_state_dict(actual.state_dict())
        assert actual.core is core and independent.core is not core
        assert torch.equal(actual(hh,v0,v0),torch.zeros_like(hh))
        assert torch.allclose(actual(hh,v0,v1),independent(hh,v0,v1))
        assert sum(p.numel() for p in actual.parameters())==sum(p.numel() for p in independent.parameters())
        actual(hh,v0,v1).square().mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in core.parameters())
    x=torch.randn(8,2,256,device=device)
    x[:4,:,16:]=x[:4,:,:240]
    x=x/x.square().sum(1).mean(1)[:,None,None].sqrt()
    views=r.descriptors(x)
    assert views.shape==(8,49) and views[:,29:32].std(0).max()>.001
    # Report counterexample: equal normalized means and all mean projections,
    # but different directional covariance, margin lower tail and accuracy.
    a=torch.tensor([[.6,.8,0],[.6,-.8,0]],device=device)
    b=torch.tensor([[.6,0,.8],[.6,0,-.8]],device=device)
    directions=torch.tensor([[1.,0,0],[0,1,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]],device=device)
    la=a@directions.T;lb=b@directions.T
    sa=r.margin_statistics(r.margin_vector(la,0));sb=r.margin_statistics(r.margin_vector(lb,0))
    mean,_=r.fit_objective(sa,sb,'mean');distribution,errors=r.fit_objective(sa,sb)
    assert torch.allclose(a.mean(0),b.mean(0)) and float(mean)<1e-12
    assert float(distribution)>0 and float(errors['covariance'])>0 and float(errors['lower_tail'])>0
    assert float((la.argmax(1)==0).float().mean())==.5
    assert float((lb.argmax(1)==0).float().mean())==1
    assert sa['covariance_split_error'] is None and sa['tail_split_error'] is None
    singleton=r.margin_statistics(torch.ones(1,5,device=device))
    assert singleton['mean_se'] is None and singleton['evidence_weight']<.12
    assert torch.isfinite(singleton['covariance']).all()
    many=r.margin_statistics(torch.randn(16,5,device=device))
    assert many['covariance_split_error'] is not None and many['tail_split_error'] is not None
    shrunken=r.margin_statistics(torch.zeros(2,5,device=device),prior=many)
    assert shrunken['coarse_shrink'] and torch.allclose(shrunken['mean'],.8*many['mean'])
    # Fractional empirical tail exactly equals variational CVaR objective.
    risk=torch.tensor([1.,2.,4.,8.,9.],device=device,requires_grad=True)
    tail=r.cvar(risk,.3);assert abs(float(tail.detach())-8.6666667)<1e-5
    tail.backward();assert risk.grad[0]==0 and risk.grad[-1]>0

    identity=TinyIdentity().to(device);online=r.OnlineReceiver(action=TinyAction(),max_age_steps=50).to(device)
    fixture(online)
    for entry in online.entries.values():
        for key in ('h','v','logits','audit_x'):
            if key in entry:entry[key]=entry[key].to(device)
    with patch.object(r,'classify_intermediate',lambda model,h:model.backend(h)), \
         patch.object(r,'intermediate',lambda model,x:x.flatten(1)[:,:349]):
        relations=online._relations(online._groups(0,'v1'),fit=True)
        assert relations and all(k[0]!=0 and 0 not in c['donor_tx'] for k,source,c in relations)
        key,source,c=relations[0]
        role_sets=[{e['pid'] for e in bag} for bag in (c['donors'],source,c['dest']+c['coarse'])]
        assert not any(role_sets[a]&role_sets[b] for a,b in ((0,1),(0,2),(1,2)))
        assert c['key'][2:]==key[2:]
        target=online._target(c,key[0]);assert target['coarse_shrink']
        fit,stats=online.fit_loss(identity,0,'v1');fit.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in online.parameters())
        assert all(p.grad is None for p in identity.parameters())
        assert stats['fit_used']==1 and stats['fit_used_this_call']==1 and stats['available']>1
        assert stats['unique_physical_ids']==len(online.entries)
        assert stats['unique_used_physical_ids']<stats['unique_physical_ids']
        online.zero_grad(set_to_none=True)
        h=torch.stack([e['h'] for e in source]).detach().requires_grad_(True)
        # Use output of E to prove no auxiliary gradient goes to the backbone.
        live=identity.encoder(h)
        n=len(h);ids=[e['pid'] for e in source]
        args=(identity,live,torch.full((n,),key[0]),torch.full((n,),key[1]),
              torch.full((n,),key[2]),ids,'clean',0,'v1')
        loss,used=online.identity_loss(*args,reliability=.5);assert loss is not None
        loss.backward()
        assert h.grad is None and all(p.grad is None for p in identity.encoder.parameters())
        assert all(p.grad is None for p in online.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in identity.backend.parameters())
        before=online.usage['identity_used'];online.identity_loss(*args,reliability=0)
        assert online.usage['identity_used']==before
        delta=online._delta(h,c)
        assert bool((delta.norm(dim=1)<=.100001*h.norm(dim=1)).all())
        audit=online.calibrate(identity,0,'v1',max_relations=100)
        assert audit['tx_holdout_records'] and audit['audit_used_this_call']>0
        assert all(0<=w<=1 for w in audit['weights'].values())
        assert all('directed_other_tx_mean_lower_tail' in row for row in audit['records'])
        held=next((k,s,c) for k,s,c in online._relations(online._groups(0,'v1',1)) if k[0]==0)
        k,s,c=held;hl=torch.stack([e['h'] for e in s]);n=len(s)
        live_args=(identity,hl,[k[0]]*n,[k[1]]*n,[k[2]]*n,[e['pid'] for e in s],'clean',0,'v1')
        cal1=online.calibrate_live(*live_args,min_distinct_ids=16)
        cal2=online.calibrate_live(*live_args,min_distinct_ids=16)
        assert cal1['metrics']['clean']['distinct_recipient_ids']==cal2['metrics']['clean']['distinct_recipient_ids']==32
        assert cal2['reencoded_audit_packets']==32 and cal2['audit_packet_budget']==32
        assert all(row['recipient_distinct_ids']>=4 for row in cal2['records']['clean'])
        singleton=online._audit_row(identity,hl[:1],k,c)
        assert singleton['learned_covariance'] is None and singleton['learned_lower_tail'] is None
        unsupported_weight,unsupported=online._reliability([singleton])
        assert unsupported_weight==0 and unsupported['skills']['covariance'] is None
        # Drift invalidates data/reliability without rejuvenating a contribution.
        online.expire(51,'v1');assert not online.entries
        online.reset('v2');assert not online.live_calibration
    # Same fit/identity loss across modes except statistic objective switch.
    assert r.OnlineReceiver(action=TinyAction(),target_mode='mean').tail_weight==online.tail_weight
    oracle=singleton_oracle_check(device)
    return dict(status='PASS',device=device,checks=['equal_mean_different_accuracy_counterexample',
        'actual_shared_core_same_form_zero_action','variable_quality_proxy_after_RMS_normalization',
        'five_margin_covariance_and_lower_tail','small_bag_uncertainty_and_coarse_shrink',
        'exact_fractional_CVaR','fit_reference_frozen','identity_GC_only_no_E_or_action_gradient',
        'physical_three_bags','TX_holdout_excludes_fit_descriptors','day_condition_match',
        'actual_use_vs_available_counts','repeats_not_unique_samples','zero_admission_no_identity_use',
        'bounded_action','true_destination_heldout_calibration','version_age_expiry',
        'repeated_singleton_IQ_accumulates_supported_live_distribution',
        'oracle_live_reliability_positive_without_relaxing_gates',
        'unsupported_covariance_and_tail_are_null','fresh_current_student_reencoding'],
        singleton_oracle=oracle)


def singleton_oracle_check(device):
    class OracleAction(nn.Module):
        def forward(self,h,v0,v1):return .1*h
    identity=TinyIdentity().to(device)
    online=r.OnlineReceiver(action=OracleAction(),max_age_steps=50).to(device)
    online.reset('oracle-v1')
    def ids(prefix,role,n):
        result=[];i=0
        while len(result)<n:
            pid=f'{prefix}-{i}';i+=1
            if r._digest(pid)%3==role and (r._digest(pid)//3)%2==1:result.append(pid)
        return result
    recipient_ids=ids('oracle-recipient',1,16);destination_ids=ids('oracle-destination',2,16)
    samples=torch.randn(16,349,device=device)+.5
    with patch.object(r,'classify_intermediate',lambda model,h:model.backend(h)), \
         patch.object(r,'intermediate',lambda model,x:x.flatten(1)[:,:349]), \
         patch.object(r,'descriptors',lambda x:torch.zeros(len(x),49,device=x.device)):
        for i,pid in enumerate(destination_ids):
            moved=1.05*samples[i]
            online.entries[(pid,'clean')]=dict(pid=pid,y=0,rx=1,day=0,condition='clean',
                role=2,partition=1,step=0,h=moved,v=torch.zeros(49,device=device),logits=identity.backend(moved).detach())
        for rx in (0,1):
            for pid in ids('oracle-donor-'+str(rx),0,4):
                online.entries[(pid,'clean')]=dict(pid=pid,y=1,rx=rx,day=0,condition='clean',
                    role=0,partition=1,step=0,h=torch.zeros(349,device=device),v=torch.zeros(49,device=device),
                    logits=torch.zeros(6,device=device))
        snapshots=[]
        for i,pid in enumerate(recipient_ids):
            h=samples[i:i+1];x=F.pad(h,(0,163)).reshape(1,2,256)
            online.update(x,h,[0],[0],[0],[pid],'clean',i,'oracle-v1',identity)
            result=online.calibrate_live(identity,h,[0],[0],[0],[pid],'clean',i,'oracle-v1')
            snapshots.append(result.get('weights',{}).get('clean',0.))
        assert snapshots[-1]>.6 and all(value==0 for value in snapshots[:15])
        assert result['records']['clean'][0]['recipient_distinct_ids']==16
        assert result['metrics']['clean']['distinct_recipient_ids']==16
        assert result['reencoded_audit_packets']==16
        assert all(p.grad is None for p in identity.parameters())
        # Change CURRENT backend; unchanged cache must yield changed errors now.
        previous=result['metrics']['clean']['learned_mean']
        with torch.no_grad():identity.backend.weight.mul_(1.3)
        changed=online.calibrate_live(identity,h,[0],[0],[0],[pid],'clean',16,'oracle-v1')
        assert changed['metrics']['clean']['learned_mean']>previous+1e-8
        assert all(e['role']==1 and e['partition']==1 for e in online.entries.values() if 'audit_x' in e)
        online.expire(70,'oracle-v1');assert not online.entries and not online.live_calibration
    return dict(singleton_calls=16,final_distinct_recipient_ids=16,oracle_live_weight=snapshots[-1],
                premature_admissions=sum(value>0 for value in snapshots[:15]))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    print(json.dumps(checks(args.device),indent=2))
