import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr import support_calibrated as base


def features(seed,n):
    rng=np.random.default_rng(seed)
    return {key:rng.normal(size=(n,96 if key=='fft' else 160)) for key in base.NAMES}


@pytest.mark.parametrize('band',['nearest','all'])
def test_registered_bandwidth_uses_current_new_support_and_matches_formula(band):
    x=features(23,8);y=np.arange(8);ids=list(map(str,y));rule=dict(config=['identity',band,0.,'single'],ridge=.1)
    old=base.fit(x,y,ids,fixed=rule);state=base.fit(x,y,ids,fixed=dict(rule,bandwidth_scope='registered'))
    unit=base.unit_features(x);d=np.maximum(2-2*(unit['z_id']@unit['z_id'].T),0)
    expected=np.median(np.min(np.where(y[:,None]!=y[None,:],d,np.inf),axis=1)) if band=='nearest' else np.median(d[y[:,None]!=y[None,:]])
    assert state.tau==pytest.approx(expected) and state.selection['bandwidth_physical_count']==8
    changed={key:value.copy() for key,value in x.items()}
    changed['z_id'][6]=x['z_id'][0]+.001;changed['z_id'][7]=x['z_id'][1]+.001
    assert base.fit(changed,y,ids,fixed=rule).tau==old.tau
    assert base.fit(changed,y,ids,fixed=dict(rule,bandwidth_scope='registered')).tau!=state.tau
    query=features(24,5);scores=state.score(query)
    assert np.allclose(scores,np.concatenate([state.score({key:value[i:i+1] for key,value in query.items()}) for i in range(5)]))
    assert state.tau==pytest.approx(expected)


def test_B_is_identical_and_invalid_scope_rejected():
    x=features(25,6);y=np.arange(6);ids=list(map(str,y));rule=dict(config=['historical','nearest',.5,'multi'],ridge=.1)
    old=base.fit(x,y,ids,fixed=rule);new=base.fit(x,y,ids,fixed=dict(rule,bandwidth_scope='registered'))
    assert old.tau==new.tau and np.array_equal(old.alpha,new.alpha) and np.array_equal(old.score(x),new.score(x))
    with pytest.raises(ValueError,match='bandwidth'):base.fit(x,y,ids,fixed=dict(rule,bandwidth_scope='query'))


def test_nested_outer_folds_use_only_train_support_for_registered_scale(monkeypatch):
    from experiments.pamf_phase2_practical.milkr import nested_fisher as nested,fisher_signature as fisher
    rng=np.random.default_rng(26);x=features(26,24);x['sig_views']=rng.normal(size=(24,336));y=np.repeat(np.arange(8),3);ids=list(map(str,range(24)))
    original=fisher.fit;observed=[]
    def capture(*args,**kwargs):
        state=original(*args,**kwargs);observed.append((len(args[1]),state.selection['bandwidth_physical_count'],state.selection['bandwidth_scope']));return state
    monkeypatch.setattr(fisher,'fit',capture)
    rule=dict(config=['signature_identity','all',0.,'single'],ridge=.1,outer_folds=3,outer_strengths=[0.,.25,.5,.75],bandwidth_scope='registered')
    nested.fit(x,y,ids,fixed=rule)
    assert len(observed)==13 and all(size==used and scope=='registered' for size,used,scope in observed)
    assert {size for size,_,_ in observed}=={16,24}
