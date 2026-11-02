import numpy as np
from hierarchical_prior import fit_hierarchical,two_clusters

def test_hierarchical_prior_is_deterministic_and_positive_definite():
 rng=np.random.default_rng(41);centers=np.array([[-4,0,0,0],[-2,0,0,0],[2,0,0,0],[4,0,0,0]]);x=np.concatenate([c+rng.normal(scale=.2,size=(20,4)) for c in centers]);labels=np.repeat([0,0,1,1],20);a=fit_hierarchical(x,labels,nclasses=2,nproj=3,min_part=5);b=fit_hierarchical(x,labels,nclasses=2,nproj=3,min_part=5);means,covs,priors,counts,classes,ridge=a;assert means.shape==(4,4) and covs.shape==(4,4,4);assert np.allclose(priors.sum(),1) and counts.sum()==len(x) and np.array_equal(classes,[0,0,1,1]);assert np.linalg.eigvalsh(covs).min()>0;assert all(np.array_equal(p,q) for p,q in zip(a,b));assert np.allclose((means*priors[:,None]).sum(0),x.mean(0))

def test_identical_cluster_features_are_rejected():
 import pytest
 with pytest.raises(ValueError):two_clusters(np.ones((5,3)),42)
