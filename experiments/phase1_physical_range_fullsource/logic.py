"""Matching and pair counts for fixed source predictions."""
import numpy as np

def match_ids(reference, query):
    reference=np.asarray(reference);query=np.asarray(query)
    if reference.ndim!=1 or query.ndim!=1 or len(reference)==0:
        raise ValueError('One-dimensional nonempty reference IDs are required')
    if len(np.unique(reference))!=len(reference) or len(np.unique(query))!=len(query):
        raise ValueError('IDs must be unique')
    order=np.argsort(reference);pos=np.searchsorted(reference[order],query)
    if np.any(pos>=len(reference)) or not np.array_equal(reference[order[np.minimum(pos,len(reference)-1)]],query):
        raise ValueError('All query IDs must be in the reference')
    return order[pos]

def pair_counts(target, baseline, candidate, eligible):
    target=np.asarray(target);baseline=np.asarray(baseline);candidate=np.asarray(candidate);eligible=np.asarray(eligible)
    if target.ndim!=1 or any(x.shape!=target.shape for x in (baseline,candidate,eligible)) or eligible.dtype!=bool:
        raise ValueError('Matching one-dimensional targets and eligibility are required')
    a=baseline==target;b=candidate==target
    return dict(count=int(eligible.sum()),baseline_correct=int((eligible&a).sum()),
                candidate_correct=int((eligible&b).sum()),repairs=int((eligible&~a&b).sum()),
                damages=int((eligible&a&~b).sum()))
