import torch
from experiments.phase1_dura_orthogonal.source_band_retrieval import features, ranks


def test_aligned_retrieval_and_explicit_tie_count():
    torch.manual_seed(81)
    iq=torch.randn(8,2,256)
    for balanced in (False,True):
        f=features(iq,balanced)
        rank,ties=ranks(f,f,batch=3)
        assert (rank==1).all() and (ties==0).all()
        torch.testing.assert_close(f.norm(dim=1),torch.ones(8))
    identical=features(torch.ones(3,2,256),True)
    rank,ties=ranks(identical,identical,batch=2)
    assert (rank==1).all() and (ties==2).all()
