import numpy as np
import pytest
from experiments.pamf_phase2_practical.teacher_features import probabilities,TeacherBridge
from experiments.pamf_phase2_practical.milkr.orbit_cca import OrbitCCA
from experiments.pamf_phase2_practical.milkr.source_bridge import DIMS
from experiments.pamf_phase2_practical.milkr import teacher_kernel as model


def examples(seed,n):
    rng=np.random.default_rng(seed)
    x={key:rng.normal(size=(n,d)) for key,d in [('z_id',160),('fft',96),('t_emb',160),('f_emb',160),('pa_local',160),('sig_views',336)]}
    x['teacher_probs']=probabilities(rng.normal(size=(n,4,6)))
    return x


def rule():return dict(config=['signature_identity','all',.5,'single'],ridge=.1,ridge_reference_classes=6,teacher_weight=.25)


def test_source_old_columns_are_removed_before_softmax_and_phase_mean():
    rng=np.random.default_rng(25);logits=rng.normal(size=(7,4,6));columns=[0,2,3,5]
    result=probabilities(logits,columns);changed=logits.copy();changed[:,:,[1,4]]=1e6
    assert np.array_equal(result,probabilities(changed,columns))
    assert np.allclose(result,probabilities(np.roll(logits,1,axis=1),columns))
    selected=logits[:,:,columns];selected=np.exp(selected-selected.max(2,keepdims=True));expected=(selected/selected.sum(2,keepdims=True)).mean(1)
    assert result.shape==(7,4) and np.allclose(result,expected) and np.allclose(result.sum(1),1)


def test_teacher_bridge_ignores_head_payload_during_CCA_and_preserves_it(tmp_path):
    rng=np.random.default_rng(26);raw={key:rng.normal(size=(8,2*d)) for key,d in [('z_id',160),('fft',96),('t_emb',160),('f_emb',160),('pa_local',160),('sig_views',336)]}
    raw['teacher_phase_logits']=rng.normal(size=(8,4,6));cca=OrbitCCA([(np.zeros(2*d),np.zeros(2*d),np.eye(2*d,64),np.eye(2*d,64)) for d in DIMS]);bridge=TeacherBridge(cca)
    bridge.save(tmp_path/'model.npz');loaded=TeacherBridge.load(tmp_path/'model.npz');out=loaded.transform(raw,restored=False,old_columns=[0,2,3,5])
    original=cca.transform({key:value for key,value in raw.items() if key!='teacher_phase_logits'},restored=False)
    assert all(np.array_equal(out[key],value) for key,value in original.items())
    assert np.array_equal(out['teacher_probs'],probabilities(raw['teacher_phase_logits'],[0,2,3,5]))
    with np.load(tmp_path/'model.npz') as z:assert str(z['schema'])=='source_phase_orbit_cca_v1' and not any('teacher' in key for key in z.files)


def test_teacher_kernel_PSD_K1_reference_and_query_independence():
    x=examples(27,8);y=np.arange(8);state=model.fit(x,y,list(map(str,y)),fixed=rule())
    sim=model.similarity(state.features,state.features,state.config[0]);K=model.base.make_kernel(sim,state.tau,state.config[2],state.config[3])
    assert np.linalg.eigvalsh(K).min()>-1e-10 and np.allclose(np.diag(K),1)
    query=examples(28,5);scores=state.score(query)
    assert np.allclose(scores,np.concatenate([state.score({key:value[i:i+1] for key,value in query.items()}) for i in range(5)]))
    assert state.ridge==pytest.approx(.1*8/6) and state.alpha.shape==(8,8)
    with pytest.raises(ValueError,match='weight'):model.fit(x,y,list(map(str,y)),fixed=dict(rule(),teacher_weight=.9))


def test_physical_OOF_teacher_kernel_excludes_held_support_from_scale(monkeypatch):
    x=examples(29,24);y=np.repeat(np.arange(8),3);ids=list(map(str,range(24)));original=model.context;seen=[]
    def capture(features,labels,config):seen.append(len(labels));return original(features,labels,config)
    monkeypatch.setattr(model,'context',capture);state=model.fit(x,y,ids,fixed=rule())
    assert set(seen)=={16,24} and seen.count(24)==1
    assert state.selection['physical_OOF'] and state.selection['candidate_count']==120
    assert state.selection['old_head_columns']==6


def test_teacher_method_uses_its_custom_fit():
    from experiments.pamf_phase2_practical.milkr_cv_screen import fit_function
    assert fit_function({'kernel_method':'teacher_kernel'}) is model.fit


def test_opt_in_actual_phase_logit_payload_preserves_C4(monkeypatch):
    from experiments.pamf_phase2_practical import phase_orbit as orbit,teacher_features
    from tests.test_milkr_phase_orbit import fake_capture
    monkeypatch.setattr(orbit,'capture',fake_capture);x=np.random.default_rng(30).normal(size=(5,2,256)).astype(np.float32)
    features,_=teacher_features.extract(None,x,'cpu',batch=3)
    shifted,_=teacher_features.extract(None,orbit.rotate(x,1),'cpu',batch=2)
    assert features['teacher_phase_logits'].shape==(5,4,6)
    assert np.allclose(probabilities(features['teacher_phase_logits']),probabilities(shifted['teacher_phase_logits']))
    assert all(np.allclose(value,shifted[key],atol=1e-7) for key,value in features.items() if key!='teacher_phase_logits')
