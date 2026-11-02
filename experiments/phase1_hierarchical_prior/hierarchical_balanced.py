"""Balanced source-only four-component subdivision of each pseudo class."""
import numpy as np
from hierarchical_prior import two_clusters

def fit_balanced(values,labels,nclasses=6,nproj=16,min_part=64,seed=2026121654):
 values=np.asarray(values,dtype=np.float64);labels=np.asarray(labels,dtype=np.int64)
 if values.ndim!=2 or labels.shape!=(len(values),) or not np.isfinite(values).all() or set(np.unique(labels))!=set(range(nclasses)):raise ValueError('Finite class-labeled source features required')
 dim=values.shape[1];mean=values.mean(0);v=values-mean;global_cov=v.T@v/len(v);global_cov=(global_cov+global_cov.T)*.5;ridge=max(1e-12,1e-4*np.trace(global_cov)/dim);means=[];covs=[];counts=[];classes=[]
 for cls in range(nclasses):
  data=values[labels==cls]
  if len(data)<4*min_part:raise ValueError('Insufficient source samples per class')
  center=data.mean(0);dv=data-center;class_cov=dv.T@dv/len(data);class_cov=(class_cov+class_cov.T)*.5;_,eigvec=np.linalg.eigh(class_cov);coarse=two_clusters(dv@eigvec[:,-min(nproj,dim):],seed+cls)
  for group in (0,1):
   parent=data[coarse==group]
   if len(parent)<2*min_part:raise ValueError('Insufficient source samples for balanced subgroup split')
   centered=parent-parent.mean(0);local_cov=centered.T@centered/len(parent);_,local_vec=np.linalg.eigh((local_cov+local_cov.T)*.5);order=np.argsort(centered@local_vec[:,-1],kind='stable');halves=np.array_split(order,2)
   for inds in halves:
    part=parent[inds];m=part.mean(0);d=part-m;cov=d.T@d/len(part);cov=.4*((cov+cov.T)*.5)+.4*class_cov+.2*global_cov+ridge*np.eye(dim);means.append(m);covs.append(cov);counts.append(len(part));classes.append(cls)
 return np.stack(means),np.stack(covs),np.asarray(counts,dtype=np.float64)/sum(counts),np.asarray(counts),np.asarray(classes),ridge
