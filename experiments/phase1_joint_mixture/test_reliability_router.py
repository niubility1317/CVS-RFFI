import numpy as np
from reliability_router import choose_by_score,ReliabilityRouter

def test_fixed_margin_selects_only_different_candidate():
 scores=np.array([[.7,.8,.1,.2,.3],[.7,.8,.1,.2,.3],[.7,.72,.1,.2,.3]]);pred=np.array([[0,1,2,3,4],[0,0,2,3,4],[0,1,2,3,4]])
 assert choose_by_score(scores,pred,.05).tolist()==[1,0,0]
 permutation=np.array([4,2,5,0,3,1]);assert np.array_equal(choose_by_score(scores,pred,.05),choose_by_score(scores,permutation[pred],.05))

def test_train_predict_correctness_scores_are_finite():
 rng=np.random.default_rng(3);features=rng.normal(size=(40,5,19));pred=rng.integers(0,6,size=(40,5));labels=rng.integers(0,6,size=40);net=ReliabilityRouter().fit(features,pred,labels);scores=net.score(features[:3]);assert scores.shape==(3,5) and np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1))
