import json
import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr import kernel_margin as m,nested_fisher as n


def test_margin_uses_current_support_kernel_and_all_class_queries_independently():
    rng=np.random.default_rng(417);count=8;k=3;x={};q={}
    for name,d in [('z_id',160),('fft',96),('t_emb',160),('f_emb',160),('pa_local',160),('sig_views',336)]:
        centers=rng.normal(size=(count,d));x[name]=np.repeat(centers,k,axis=0)+.03*rng.normal(size=(count*k,d));q[name]=np.repeat(centers,2,axis=0)+.03*rng.normal(size=(count*2,d))
    y=np.repeat(np.arange(count),k);ids=[str(i) for i in range(len(y))]
    rule=dict(config=['signature_identity','all',.5,'single'],ridge=.1,outer_folds=3,outer_strengths=[0.,.25,.5,.75])
    states=m.fit_grid(x,y,ids,fixed=rule);reference=n.fit(x,y,ids,fixed=rule)
    assert np.array_equal(reference.alpha,states['ridge'].alpha)
    state=states[10.];before=state.classifier.dual_coef_.copy();scores=state.score(q)
    assert np.array_equal(scores.argmax(1),np.repeat(np.arange(count),2))
    individual=np.concatenate([state.score({name:value[i:i+1] for name,value in q.items()}) for i in range(len(scores))])
    assert np.allclose(scores,individual) and np.array_equal(before,state.classifier.dual_coef_)
    json.dumps(state.selection,allow_nan=False)
    with pytest.raises(ValueError,match='Source-frozen'):m.fit(x,y,ids,fixed=rule)
