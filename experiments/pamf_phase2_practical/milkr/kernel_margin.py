"""Maximum-margin diagnostic on the unchanged R14 support-selected kernel."""
from dataclasses import dataclass
import numpy as np
from sklearn.svm import SVC
from . import nested_fisher,support_calibrated as base

CS=(.1,1.,10.,100.)


def gram(state,features):
    query=state.normalizer(features);geometry,_,mix,scales=state.config
    return base.make_kernel(state.similarity_fn(query,state.features,geometry),state.tau,mix,scales)


@dataclass(frozen=True)
class State:
    kernel_state:object
    classifier:object
    selection:dict

    def __getattr__(self,name):return getattr(self.kernel_state,name)

    @property
    def alpha(self):return self.classifier.dual_coef_

    def score(self,features):
        values=self.classifier.decision_function(gram(self.kernel_state,features))
        if values.ndim!=2 or values.shape[1]!=len(self.classifier.classes_) or not np.isfinite(values).all():raise FloatingPointError('Invalid all-class margin scores')
        return values


def fit_grid(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    state=nested_fisher.fit(features,labels,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo)
    K=gram(state,features);y=np.asarray(labels,np.int64);out={'ridge':state}
    for C in CS:out[C]=fit_classifier(state,K,y,C)
    return out


def fit_classifier(state,K,y,C):
    classifier=SVC(C=C,kernel='precomputed',decision_function_shape='ovr',tol=1e-4,max_iter=100000,cache_size=256)
    classifier.fit(K,y)
    if classifier.fit_status_!=0:raise RuntimeError('Margin optimizer did not converge')
    if not np.array_equal(classifier.classes_,np.arange(int(y.max())+1)):raise ValueError('Margin registered class order differs')
    selection=dict(state.selection,margin_classifier=dict(C=C,kernel='precomputed_unchanged_R14',objective='C_SVC_one_vs_one_OVR_decision_scores',
        support_vectors=int(classifier.n_support_.sum()),R14_kernel_selection_ridge=state.ridge,tol=1e-4,max_iter=100000,converged=True,query_fit=False))
    return State(state,classifier,selection)


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('margin_C') not in CS:raise ValueError('Source-frozen margin C required')
    state=nested_fisher.fit(features,labels,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo)
    return fit_classifier(state,gram(state,features),np.asarray(labels,np.int64),fixed['margin_C'])
