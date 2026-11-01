"""Paired source-truth counts after predictions have been fixed."""
import numpy as np

def summarize(predictions,truth,mask):
 if predictions.ndim!=2 or predictions.shape[1]!=4 or truth.shape!=(len(predictions),) or mask.shape!=(len(predictions),):raise ValueError('Expected matching four-column predictions, truth and mask')
 if not np.count_nonzero(mask):raise ValueError('At least one labeled source record required')
 base=predictions[:,2]==truth;out=dict(count=int(np.count_nonzero(mask)))
 for index,name in enumerate(('clean','raw','dura','blind')):
  correct=predictions[:,index]==truth;out[name]=dict(correct=int(np.count_nonzero(mask&correct)),repairs=int(np.count_nonzero(mask&~base&correct)),damages=int(np.count_nonzero(mask&base&~correct)))
 return out
