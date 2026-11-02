import numpy as np
from hierarchical_balanced import fit_balanced

def test_balanced_subdivision_preserves_samples_and_covariance():
 rng=np.random.default_rng(88);centers=np.array([[-4,0,0,0],[-2,0,0,0],[2,0,0,0],[4,0,0,0]]);x=np.concatenate([c+rng.normal(scale=.2,size=(40,4)) for c in centers]);labels=np.repeat([0,0,1,1],40);a=fit_balanced(x,labels,nclasses=2,nproj=3,min_part=5);b=fit_balanced(x,labels,nclasses=2,nproj=3,min_part=5);means,covs,priors,counts,classes,ridge=a;assert means.shape==(8,4) and covs.shape==(8,4,4);assert counts.sum()==len(x) and np.all(counts>=5) and np.allclose(priors.sum(),1);assert all(abs(counts[j]-counts[j+1])<=1 for j in (0,2,4,6));assert np.linalg.eigvalsh(covs).min()>0 and all(np.array_equal(p,q) for p,q in zip(a,b));assert np.allclose((means*priors[:,None]).sum(0),x.mean(0))
