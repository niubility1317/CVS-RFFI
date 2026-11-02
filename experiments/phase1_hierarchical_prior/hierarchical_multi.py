"""Source-only per-class multi-component Gaussian prior."""
import numpy as np

def kmeans_farthest(x,nclusters,seed,max_iter=50):
 if x.ndim!=2 or not np.isfinite(x).all() or nclusters<2 or len(x)<nclusters:raise ValueError('Finite source features and valid cluster count required')
 rng=np.random.default_rng(seed);first=int(rng.integers(len(x)));centers=[x[first]];nearest=((x-centers[0])**2).sum(1)
 for _ in range(1,nclusters):
  idx=int(np.argmax(nearest))
  if not nearest[idx]>0:raise ValueError('Distinct source features required')
  centers.append(x[idx]);nearest=np.minimum(nearest,((x-x[idx])**2).sum(1))
 centers=np.stack(centers);labels=np.full(len(x),-1,dtype=np.int64)
 for _ in range(max_iter):
  d=((x[:,None,:]-centers[None,:,:])**2).sum(2);updated=d.argmin(1)
  if np.array_equal(updated,labels):break
  labels=updated
  if not all(np.any(labels==j) for j in range(nclusters)):raise ValueError('Every source subcluster must be nonempty')
  centers=np.stack([x[labels==j].mean(0) for j in range(nclusters)])
 return labels

def fit_multi(values,labels,nclasses=6,nclusters=4,nproj=16,min_part=64,seed=2026121754):
 values=np.asarray(values,dtype=np.float64);labels=np.asarray(labels,dtype=np.int64)
 if values.ndim!=2 or labels.shape!=(len(values),) or not np.isfinite(values).all() or set(np.unique(labels))!=set(range(nclasses)):raise ValueError('Finite class-labeled source features required')
 dim=values.shape[1];mean=values.mean(0);v=values-mean;global_cov=v.T@v/len(v);global_cov=(global_cov+global_cov.T)*.5;ridge=max(1e-12,1e-4*np.trace(global_cov)/dim);means=[];covs=[];counts=[];classes=[]
 for cls in range(nclasses):
  data=values[labels==cls]
  if len(data)<nclusters*min_part:raise ValueError('Insufficient source samples per class')
  center=data.mean(0);dv=data-center;class_cov=dv.T@dv/len(data);class_cov=(class_cov+class_cov.T)*.5;_,eigvec=np.linalg.eigh(class_cov);sub=kmeans_farthest(dv@eigvec[:,-min(nproj,dim):],nclusters,seed+cls)
  for k in range(nclusters):
   part=data[sub==k]
   if len(part)<min_part:raise ValueError('Insufficient samples in a source subcluster')
   m=part.mean(0);d=part-m;cov=d.T@d/len(part);cov=.4*((cov+cov.T)*.5)+.4*class_cov+.2*global_cov+ridge*np.eye(dim);means.append(m);covs.append(cov);counts.append(len(part));classes.append(cls)
 return np.stack(means),np.stack(covs),np.asarray(counts,dtype=np.float64)/sum(counts),np.asarray(counts),np.asarray(classes),ridge
