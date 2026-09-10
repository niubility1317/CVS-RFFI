import torch

from cvsrffi.evidence_head import EvidenceHead


def test_one_class_episode_is_not_counted_as_effective_support_training():
    head=EvidenceHead(2,2,{'variant':'H4','covariance_rank':1})
    head.response.fit_state_domain(torch.tensor([[-1.,-1.],[1.,1.]]),source_training=True)
    state=torch.zeros(4,2)
    result={'z':torch.tensor([[-1.,0.],[-.8,.1],[1.,0.],[.8,.1]]),
            'scores':torch.zeros(4,2,requires_grad=True),
            'means':head.response(state),
            'covariance':torch.eye(2).expand(4,2,2,2),
            'state':state,'observed':torch.ones(4,2,dtype=torch.bool),
            'observation_variance':torch.ones(4,2),
            'state_covariance':torch.zeros(4,2,2,2),
            'pair_residual':torch.zeros(4,2,2),
            'state_in_domain':torch.ones(4,dtype=torch.bool)}
    _,single=head.extra_loss(result,torch.zeros(4,dtype=torch.long),['a','b','c','d'])
    assert single['support_competing_classes']==1
    assert single['support_effective_queries']==single['support_queries']==0
    assert single['support_loss']==0
    _,multiple=head.extra_loss(result,torch.tensor([0,0,1,1]),['a','b','c','d'])
    assert multiple['support_competing_classes']==2
    assert multiple['support_effective_queries']==multiple['support_queries']==2
    assert multiple['support_loss']>0
