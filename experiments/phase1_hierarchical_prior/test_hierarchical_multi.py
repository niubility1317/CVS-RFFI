import numpy as np
from hierarchical_multi import fit_multi,kmeans_farthest

def test_multimodal_prior_is_reproducible_and_positive():
 rng=np.random.default_rng(71);centers=np.array([[a,b,0,0] for a in (-5,5) for b in (-3,-1,1,3)]);x=np.concatenate([c+rng.normal(scale=.12,size=(20,4)) for c in centers]);labels=np.repeat([0,0,0,0,1,1,1,1],20);a=fit_multi(x,labels,nclasses=2,nclusters=4,nproj=3,min_part=5);b=fit_multi(x,labels,nclasses=2,nclusters=4,nproj=3,min_part=5);means,covs,priors,counts,classes,ridge=a;assert means.shape==(8,4) and covs.shape==(8,4,4);assert counts.sum()==len(x) and np.allclose(priors.sum(),1) and np.array_equal(classes,[0,0,0,0,1,1,1,1]);assert np.linalg.eigvalsh(covs).min()>0 and all(np.array_equal(p,q) for p,q in zip(a,b));assert np.allclose((means*priors[:,None]).sum(0),x.mean(0))

def test_identical_features_rejected():
 import pytest
 with pytest.raises(ValueError):kmeans_farthest(np.ones((8,3)),4,42)
