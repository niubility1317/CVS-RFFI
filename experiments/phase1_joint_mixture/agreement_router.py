"""Choose among fixed waveforms from equality patterns, without class-specific rules."""
import numpy as np

class AgreementRouter:
 def __init__(self,prior_strength=16.,min_count=8,margin=0.):
  if prior_strength<=0 or min_count<1 or margin<0:raise ValueError('Invalid fixed router parameters')
  self.prior_strength=float(prior_strength);self.min_count=int(min_count);self.margin=float(margin);self.patterns=None;self.prior=None
 @staticmethod
 def signature(row):
  if len(row)!=5:raise ValueError('Five fixed candidates required')
  return tuple(bool(row[i]==row[j]) for i in range(5) for j in range(i+1,5))
 def fit(self,predictions,labels):
  predictions=np.asarray(predictions);labels=np.asarray(labels)
  if predictions.ndim!=2 or predictions.shape[1]!=5 or labels.shape!=(len(predictions),) or len(labels)==0:raise ValueError('Aligned nonempty training predictions and labels required')
  if np.any((predictions<0)|(predictions>=6)|(labels[:,None]<0)|(labels[:,None]>=6)):raise ValueError('Valid six-class source labels required')
  correct=predictions==labels[:,None];self.prior=correct.mean(axis=0);self.patterns={}
  for row,truth in zip(predictions,correct):
   key=self.signature(row)
   if key not in self.patterns:self.patterns[key]=[0,np.zeros(5,dtype=np.float64)]
   self.patterns[key][0]+=1;self.patterns[key][1]+=truth
  return self
 def choose(self,predictions):
  predictions=np.asarray(predictions)
  if self.patterns is None or predictions.ndim!=2 or predictions.shape[1]!=5:raise ValueError('Fitted router and aligned candidates required')
  chosen=np.zeros(len(predictions),dtype=np.int64)
  for i,row in enumerate(predictions):
   item=self.patterns.get(self.signature(row))
   if item is None or item[0]<self.min_count:continue
   n,success=item;scores=(success+self.prior_strength*self.prior)/(n+self.prior_strength);best=int(np.argmax(scores))
   if row[best]!=row[0] and scores[best]>scores[0]+self.margin:chosen[i]=best
  return chosen
