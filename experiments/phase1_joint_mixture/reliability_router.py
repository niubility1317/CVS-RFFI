"""Source-trained candidate correctness scores and fixed-margin selection."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def choose_by_score(scores,predictions,margin):
 scores=np.asarray(scores);predictions=np.asarray(predictions)
 if scores.ndim!=2 or scores.shape[1]!=5 or predictions.shape!=scores.shape or margin<0 or not np.isfinite(scores).all():raise ValueError('Aligned finite five-candidate scores and predictions required')
 best=np.argmax(scores,axis=1);rows=np.arange(len(scores));switch=(predictions[rows,best]!=predictions[:,0])&(scores[rows,best]>scores[:,0]+margin)
 return np.where(switch,best,0).astype(np.int64)

class ReliabilityRouter:
 def __init__(self):self.models=[]
 def fit(self,features,predictions,labels):
  features=np.asarray(features);predictions=np.asarray(predictions);labels=np.asarray(labels)
  if features.ndim!=3 or features.shape[1:]!=(5,19) or predictions.shape!=(len(features),5) or labels.shape!=(len(features),) or not np.isfinite(features).all():raise ValueError('Finite aligned source candidate features and labels required')
  self.models=[]
  for j in range(5):
   target=(predictions[:,j]==labels).astype(np.int64)
   if np.unique(target).size!=2:raise ValueError('Both source correctness outcomes required')
   scaler=StandardScaler().fit(features[:,j,:]);model=LogisticRegression(C=.1,solver='lbfgs',max_iter=1000).fit(scaler.transform(features[:,j,:]),target);self.models.append((scaler,model))
  return self
 def score(self,features):
  features=np.asarray(features)
  if len(self.models)!=5 or features.ndim!=3 or features.shape[1:]!=(5,19) or not np.isfinite(features).all():raise ValueError('Fitted router and finite candidate features required')
  return np.column_stack([model.predict_proba(scaler.transform(features[:,j,:]))[:,1] for j,(scaler,model) in enumerate(self.models)])
