"""Current-support RF likelihood for the frozen old-head class prior."""
from dataclasses import dataclass
import numpy as np

RULE='support_rf_old_class_product_v1'


@dataclass(frozen=True)
class Calibrator:
    labels:np.ndarray
    old_counts:np.ndarray

    @classmethod
    def fit(cls,labels,old_count):
        y=np.array(labels,dtype=np.int64,copy=True)
        if y.ndim!=1 or not len(y) or (y<0).any() or old_count<2:raise ValueError('Invalid RF prior class labels')
        counts=np.bincount(y,minlength=old_count)[:old_count]
        if (counts<=0).any():raise ValueError('Every registered old class needs train support')
        y.flags.writeable=False;counts.flags.writeable=False;return cls(y,counts)

    def probabilities(self,original,affinity):
        p=np.asarray(original,np.float64);a=np.asarray(affinity,np.float64);m=len(self.old_counts)
        if p.ndim!=2 or p.shape[1]!=m or a.shape!=(len(p),len(self.labels)) or not np.isfinite(a).all() or (a<0).any():raise ValueError('RF prior class alignment differs')
        likelihood=np.column_stack([a[:,self.labels==c].mean(1) for c in range(m)])
        value=p*likelihood;normalizer=value.sum(1,keepdims=True)
        if not np.isfinite(normalizer).all() or (normalizer<=0).any():raise FloatingPointError('Nonpositive RF class prior likelihood')
        return value/normalizer

    def audit(self):
        return dict(rule=RULE,old_count=len(self.old_counts),old_rows=int(self.old_counts.sum()),physical_count=len(self.labels),
            state_bytes=self.labels.nbytes+self.old_counts.nbytes,scope='current_physical_train_support',
            likelihood='per_old_class_mean_equal_C4_C8_RBF',prior='normalize(original_head_probability_times_RF_likelihood)',
            neutral_RF_keeps_original=True,query_fit=False)
