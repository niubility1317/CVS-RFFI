"""Fixed nested source fit sets excluding physical-prior training IDs."""
import numpy as np

def select_fit(training,prior_ids,size):
 if size not in (1024,8192):raise ValueError('Supported source fit sizes are 1024 and 8192')
 pool=np.setdiff1d(training,prior_ids,assume_unique=True);small=np.sort(np.random.default_rng(2026135055).choice(pool,size=1024,replace=False))
 if size==1024:return small
 remaining=np.setdiff1d(pool,small,assume_unique=True);extra=np.random.default_rng(2026135058).choice(remaining,size=7168,replace=False);return np.sort(np.r_[small,extra])
