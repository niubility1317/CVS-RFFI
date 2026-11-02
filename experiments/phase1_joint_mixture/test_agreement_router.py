import numpy as np
from agreement_router import AgreementRouter

def test_class_permutation_does_not_change_selected_expert():
 p=np.array([[0,1,1,2,2],[0,1,1,2,2],[3,4,4,5,5],[0,0,0,0,0]]*12);y=np.array([1,1,4,0]*12);perm=np.array([4,2,5,0,3,1]);a=AgreementRouter(min_count=2).fit(p,y);b=AgreementRouter(min_count=2).fit(perm[p],perm[y]);assert np.array_equal(a.choose(p),b.choose(perm[p]))

def test_unknown_pattern_and_full_agreement_abstain():
 p=np.tile(np.array([[0,1,1,2,2]]),(12,1));y=np.ones(12,dtype=int);r=AgreementRouter(min_count=8).fit(p,y);assert r.choose(np.array([[0,0,0,0,0],[0,1,2,3,4]])).tolist()==[0,0]
