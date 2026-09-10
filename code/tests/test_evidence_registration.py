import pytest
import torch

from cvsrffi.evidence_head import EvidenceHead
from cvsrffi.evidence_registration import fit_registered_evidence, FrozenRegisteredEvidence


def setup(variant='H4'):
    torch.manual_seed(9)
    head=EvidenceHead(2,2,{'variant':variant,'covariance_rank':1}).double()
    head.response.fit_state_domain(torch.tensor([[-1.,-1.],[1.,1.]],dtype=torch.double),source_training=True)
    with torch.no_grad():
        head.response.mean.copy_(torch.tensor([[-1.,0.],[1.,0.]],dtype=torch.double))
        head.response.shared_slopes.fill_(.1)
    result={'z':torch.tensor([[-1.,.1],[2.,.2]],dtype=torch.double),
            'observed':torch.ones(2,2,dtype=torch.bool),
            'state':torch.tensor([[.1,.2],[.2,.1]],dtype=torch.double),
            'observation_variance':torch.ones(2,2,dtype=torch.double)*.2}
    return head,result


@pytest.mark.parametrize('variant',['H4','H5'])
def test_new_class_k1_permutation_batch_invariance_and_snapshot(variant):
    head,result=setup(variant)
    kwargs={'source_class_labels':['old',70]}
    predictor=fit_registered_evidence(head,result,['old','new'],['p1','p2'],['old','new'],**kwargs)
    reversed_predictor=fit_registered_evidence(head,result,['old','new'],['p1','p2'],['new','old'],**kwargs)
    before=predictor.to_dict()
    scored=predictor.predict(result)
    assert scored['scores'].shape==(2,2)
    assert all(c['physical_shots']==1 and c['design_rank']==1 for c in scored['coverage'])
    torch.testing.assert_close(scored['scores'],reversed_predictor.predict(result)['scores'].flip(-1))
    single={k:v[:1] for k,v in result.items()}
    torch.testing.assert_close(scored['scores'][:1],predictor.predict(single)['scores'])
    assert scored['support_covariance'].diagonal(dim1=-2,dim2=-1).min()>0
    assert not torch.allclose(scored['scores'],predictor.predict(result,include_support_uncertainty=False)['scores'])
    # Supplied source scores/means cannot influence registration or inference.
    poisoned={**result,'scores':torch.full((2,2),float('nan')),'means':torch.full((2,2,2),float('nan'))}
    torch.testing.assert_close(scored['scores'],predictor.predict(poisoned)['scores'])
    for k,v in predictor.to_dict().items():
        torch.testing.assert_close(before[k],v)
    restored=FrozenRegisteredEvidence.from_dict(before)
    torch.testing.assert_close(scored['scores'],restored.predict(result)['scores'])
    assert all(isinstance(v,torch.Tensor) for v in before.values())
    with torch.no_grad():
        head.response.mean.add_(100.)
    torch.testing.assert_close(scored['scores'],predictor.predict(result)['scores'])


def test_duplicate_rejection_and_explicit_mapping():
    head,result=setup()
    with pytest.raises(ValueError,match='distinct'):
        fit_registered_evidence(head,result,['old','new'],['p','p'],['old','new'],source_class_labels=['old',70])
    with pytest.raises(ValueError,match='at least one'):
        fit_registered_evidence(head,result,['old','old'],['p','q'],['old','new'],source_class_labels=['old',70])


def test_shared_new_prior_label_name_invariance_and_missing_evidence():
    head,result=setup('H5')
    first=fit_registered_evidence(head,result,[101,303],['p','q'],[101,303],source_class_labels=['old',70])
    renamed=fit_registered_evidence(head,result,['foo','bar'],['p','q'],['foo','bar'],source_class_labels=['old',70])
    torch.testing.assert_close(first.predict(result)['scores'],renamed.predict(result)['scores'])
    empty={**result,'observed':torch.zeros_like(result['observed'])}
    assert (first.predict(empty)['scores']==0).all()
