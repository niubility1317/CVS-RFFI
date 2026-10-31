import json
import numpy as np
import pytest
from tests.test_milkr_orbit_ensemble import features,rule
from experiments.pamf_phase2_practical.milkr import adaptive_residual as model,orbit_ensemble as core


def example(seed,n,identical=False):
    full=features(seed,n);base=full if identical else features(seed+2,n)
    return dict(full,**{'BASE_'+k:base[k] for k in model.RF})


def fixed():return dict(rule(),support_residual_strengths=list(model.STRENGTHS),K1_residual_strength=1.)


def test_endpoints_K1_identical_to_frozen_full_model_and_query_independence():
    x=example(910,8);y=np.arange(8);ids=list(map(str,y));query=example(914,5)
    state=model.fit(x,y,ids,fixed=fixed());reference=core.fit(model.blend(x,1.),y,ids,fixed=fixed())
    assert state.strength==1 and np.array_equal(state.alpha,reference.alpha) and state.selection['candidate_count']==1
    assert np.array_equal(state.score(query),reference.score(model.blend(query,1.)))
    assert all(np.array_equal(model.blend(x,0.)[k],x['BASE_'+k]) for k in model.RF)
    rows=np.concatenate([state.score({k:v[i:i+1] for k,v in query.items()}) for i in range(5)])
    assert np.allclose(rows,state.score(query)) and not state.alpha.flags.writeable
    with pytest.raises(ValueError,match='schema'):model.blend({k:v for k,v in x.items() if not k.startswith('BASE_')},.5)


def test_all_strengths_use_same_physical_train_folds_and_joint_rank(monkeypatch):
    calls=[];original=core.context
    def capture(features,labels,config,**kwargs):calls.append(len(labels));return original(features,labels,config,**kwargs)
    monkeypatch.setattr(core,'context',capture);x=example(916,24);y=np.repeat(np.arange(8),3)
    state=model.fit(x,y,list(map(str,range(24))),fixed=fixed());a=state.selection['adaptive_residual']
    assert len(calls)==5*73 and calls.count(24)==5 and set(calls)=={16,24}
    assert state.selection['candidate_count']==600 and len(state.selection['records'])==600
    winner=max(a['choices'],key=lambda r:(r['physical_OOF_H'],-r['OOF_loss'],r['ridge'],-r['strength']))
    assert state.strength==winner['strength'] and a['unselected_models_retained']==0 and not a['query_fit']
    assert np.allclose(state.score(example(918,3)).sum(1),1)


def test_identical_endpoints_prefer_zero_residual_and_keep_original_solution():
    x=example(920,24,True);y=np.repeat(np.arange(8),3);ids=list(map(str,range(24)))
    state=model.fit(x,y,ids,fixed=fixed());reference=core.fit(model.blend(x,0.),y,ids,fixed=fixed())
    assert state.strength==0 and np.array_equal(state.alpha,reference.alpha)
    with pytest.raises(ValueError,match='physical OOF'):model.fit(x,y,ids,fixed=fixed(),conditional_loo=False)


def test_source_parent_role_check_and_freeze_have_no_new_training(tmp_path):
    from experiments.pamf_phase2_practical.milkr_adaptive_source import parent_metadata,freeze
    parent_rt=dict(run_id='source-parent',kernel_method='crossrx_residual_ensemble',run_root=str(tmp_path/'parent'),source_cache='legalL',source_contract='sourceF0',old_classes=list('abcdef'),model_seeds=[1],PAMF_sources={'1':'sourceF'},DURA=dict(checkpoint='sourceG',source_contract='sourceG0',roles='LU'))
    audit=dict(steps=600,target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0)
    parent=dict(status='SOURCE_FROZEN',target_eligible=True,K1_rule=rule(),source_metric_training=[dict(C4=audit,C8=audit) for _ in range(6)],source_K1_grid=[],source_summary=[],source_physical_rows=6300,source_paired_observations=12600,source_RX_TX_folds=[],source_correlation_weighting={},target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0)
    p=tmp_path/'runtime.json';s=tmp_path/'selection.json';p.write_text(json.dumps(parent_rt));s.write_text(json.dumps(parent))
    rt=dict(parent_rt,run_root=str(tmp_path/'new'),kernel_method='adaptive_residual_ensemble',source_reuse_runtime=str(p),source_reuse_selection=str(s),source_reuse_checkpoint=str(tmp_path/'parent/source_orbit_cca.npz'),support_residual_strengths=list(model.STRENGTHS),K1_residual_strength=1.)
    prior_rt,prior=parent_metadata(rt);(tmp_path/'new').mkdir()
    class Frozen:
        def save(self,path):path.write_text('test source checkpoint')
    freeze(rt,(prior_rt,prior,Frozen()));result=json.loads((tmp_path/'new/source_selection.json').read_text())
    assert result['source_new_optimizer_updates']==0 and result['source_model_fitting_rows']==0 and result['source_metric_training']==[]
    assert result['K1_rule']['config']==parent['K1_rule']['config'] and result['K1_rule']['K1_residual_strength']==1
    assert result['source_summary']==[] and result['source_RX_TX_folds_inherited']
    parent['query_reads']=1;s.write_text(json.dumps(parent))
    with pytest.raises(Exception,match='prohibited inputs'):parent_metadata(rt)
    parent['query_reads']=0;s.write_text(json.dumps(parent));rt['source_contract']='wrongrole'
    with pytest.raises(Exception,match='role differs'):parent_metadata(rt)


def test_adaptive_fit_and_encoder_routes(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen,teacher_features
    assert screen.fit_function({'kernel_method':'adaptive_residual_ensemble'}) is model.fit
    assert teacher_features.bridge_type('adaptive_residual_ensemble') is teacher_features.AdaptiveResidualBridge
    marker=object();monkeypatch.setattr(teacher_features,'extract_orbit_ensemble',lambda *args:marker)
    assert screen.extract_for_method({'kernel_method':'adaptive_residual_ensemble'},None,None,'cpu') is marker


def test_frozen_bridge_adds_linear_base_views_without_retraining(tmp_path):
    from experiments.pamf_phase2_practical import teacher_features as tf
    from experiments.pamf_phase2_practical.milkr import crossrx_residual as residual
    from tests.test_milkr_residual_skip import payload
    def dual(seed):
        a=payload(seed,170);b=payload(seed+1,170);a.update({'C4_'+k:v for k,v in b.items() if k!='teacher_phase_logits'})
        for prefix in ('','C4_'):a[prefix+'fft'][:,96:]=0;a[prefix+'f_emb'][:,160:]=0
        return a
    raw=dual(924);restored=dual(926);base=tf.CorrelationEnsembleBridge.fit(raw,restored);encoders=[]
    for encoder in base.encoders:
        neural=residual.export(residual.networks('cpu'))
        for path in neural:
            for p in path:p[3].fill(.3)
        encoders.append(residual.CrossRXResidualCCA(encoder.parameters,neural))
    old=tf.CrossRXResidualBridge(encoders);old.save(tmp_path/'model.npz');new=tf.AdaptiveResidualBridge.load(tmp_path/'model.npz')
    for path in (False,True):
        views=new.transform(raw,restored=path,old_columns=[0,2,4,5]);full=old.transform(raw,restored=path,old_columns=[0,2,4,5]);linear=base.transform(raw,restored=path,old_columns=[0,2,4,5])
        assert all(np.array_equal(model.blend(views,1)[k],v) for k,v in full.items())
        assert all(np.array_equal(model.blend(views,0)[k],v) for k,v in linear.items())
        single=new.transform({k:v[2:3] for k,v in raw.items()},restored=path,old_columns=[0,2,4,5])
        assert all(np.allclose(v[2],single[k][0]) for k,v in views.items())
        for strength in model.STRENGTHS:
            normalized=core.normalize(model.blend(views,strength));assert np.isfinite(normalized['z_id']).all()
    assert all(not a.flags.writeable for e in new.encoders for path in e.neural for p in path for a in p)
