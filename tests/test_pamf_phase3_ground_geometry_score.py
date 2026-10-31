"""Fixed col3 ViM primary, formal95 goals and pre-truth completeness."""
import importlib
import json
import numpy as np
import pytest


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.ground_geometry_score')


def path(s,condition):
    def row():return dict(C_K=.96,R_U=.96,original_closed_minus_new_open=.01,coverage=dict(known_complete=True,unknown_complete=True,complete=True))
    result=dict(overall=row(),by_scene={k:row() for k in (('clean',) if 'clean' in condition else s.common.SCENES)})
    if condition.startswith('source'):result['groups']={name:row() for name in s.risk.GROUP_NAMES}
    return s._strict_path(result,condition)


def test_fixed_domain_vim_requires_all_formal95_conditions(subtests):
    s=api();paths={c:path(s,c) for c in s.CONDITIONS};cal={'methods':{s.PRIMARY:{'constraints_passed':True},'F0_vim':{'constraints_passed':True}}}
    assert s.PRIMARY=='F0_domain_vim' and s._decision(paths,cal)==(True,True,True,True)
    for c in s.CONDITIONS:
        with subtests.test(condition=c):
            p={k:path(s,k) for k in s.CONDITIONS};p[c]['goals']['passed']=False
            assert not s._decision(p,cal)[-1]
    with subtests.test(no_standard_vim_swap=True):
        cal['methods'][s.PRIMARY]['constraints_passed']=False
        assert s._decision(paths,cal)==(False,True,False,False)
    with subtests.test(legacy90_is_diagnostic_only=True):
        a=path(s,'cross_star_DURA');a['overall']['C_K']=.94;a['overall']['R_U']=.94
        assert not s._strict_path(a,'cross_star_DURA')['goals']['passed']


def test_domain_vim_col3_compares_three_actual_control_columns(subtests):
    s=api();t=dict(ids=np.asarray(['a','b','u','v']),known=np.asarray([True,True,False,False]),y=np.asarray([0,1,-1,-1],np.int64),
        tx=np.asarray(['14-10','14-7','19-15','19-16']),rx_i=np.ones(4,np.int64),day_i=np.ones(4,np.int64),scenario=np.asarray(['clean']*4))
    a=dict(ids=t['ids'],closed=np.asarray([[0,0,1,1],[1,0,1,1],[0,0,0,0],[0,0,0,0]],np.int64),
        scores=np.asarray([[1.,-1.,-1.,1.],[1.,1.,-1.,1.],[1.,-1.,1.,-1.],[-1.,-1.,1.,1.]],np.float64))
    expected=((0,1,1,1),(1,0,0,1),(1,0,1,0))
    for j,counts in enumerate(expected):
        with subtests.test(reference_column=j):
            p=s._paired(a,t,3,j,True)['counts']
            assert (p['known_correct_accepted_gained'],p['known_correct_accepted_lost'],p['unknown_rejected_gained'],p['unknown_rejected_lost'])==counts
            assert p['known_closed_prediction_changed']==(0 if j==2 else 1 if j==0 else 2)


def test_no_truth_for_partial_or_twelve_instead_of_sixteen_points(tmp_path,monkeypatch,subtests):
    s=api();root=tmp_path/'predict';root.mkdir()
    for key in ('_source_known_truth','_source_truth','_future_truth'):
        monkeypatch.setattr(s,key,lambda *a:pytest.fail('All sixteen points must precede every private truth read'))
    for marker in ({'status':'PARTIAL'},{'status':'GROUND_GEOMETRY_PREDICTIONS_COMPLETE','run_id':s.RUN_ID,'all_12_points_complete':True,'all_16_points_complete':False}):
        with subtests.test(marker=marker['status']):
            (root/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
            with pytest.raises(ValueError):s.score(root,'known','novel','clean','star',tmp_path/'score')


def test_new_readouts_select_vim_column_without_refitting_or_normalizing(monkeypatch,subtests):
    s=api();from experiments.pamf_phase3_open_world import ground_geometry_core as core
    def state(mode,alpha):
        return dict(schema=np.asarray(core.SCHEMA),mode=np.asarray(mode),origin=np.zeros(160,np.float64),
            residual_basis=np.eye(160,dtype=np.float64)[:,:80],nuisance_basis=np.eye(160,dtype=np.float64)[:,159:] if mode=='domain_orthogonal' else np.empty((160,0),np.float64),alpha=np.asarray(alpha,np.float64))
    a,b=state('standard',1.),state('domain_orthogonal',2.)
    x=np.zeros((3,160),np.float32);x[0,0]=1.;x[1,0]=.5
    logits=np.zeros((3,6),np.float32);energy=np.full(3,np.log(6.));residual=np.asarray([1.,.5,0.])
    monkeypatch.setattr(np.linalg,'eigh',lambda *args:pytest.fail('CPU scorer must not refit eigenspace'))
    monkeypatch.setattr(np.linalg,'svd',lambda *args:pytest.fail('CPU scorer must not refit nuisance space'))
    result=s._geometry_raw(a,b,x,logits)
    np.testing.assert_allclose(result,np.column_stack((energy-residual,energy-2*residual)),rtol=0,atol=1e-15)
    for defect in ('mode','basis','source_bank'):
        with subtests.test(defect=defect):
            bad={k:v.copy() for k,v in b.items()}
            if defect=='mode':bad['mode']=np.asarray('standard')
            elif defect=='basis':bad['residual_basis'][:,1]=bad['residual_basis'][:,0]
            else:bad['source_rows']=x.copy()
            with pytest.raises(ValueError):s._geometry_raw(a,bad,x,logits)


def test_actual_sixteen_points_geometry_replay_four_truths_and_corruptions(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_ground_geometry_run import build_prediction_fixture
    f=build_prediction_fixture(tmp_path,monkeypatch);m=f['module'];root=f['root'];truth=f['truths'];s=api()
    assert f['done']['all_16_points_complete'] and f['done']['source_subspace_fit_count']==2
    assert len(f['fit_calls'])==len(f['calibration_calls'])==2
    reads=[];captured={};known=s._source_known_truth;novel=s._source_truth;future=s._future_truth;legacy=s._legacy_score
    monkeypatch.setattr(s,'_source_known_truth',lambda *a:(reads.append('known_TEST'),known(*a))[1])
    monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source_NV'),novel(*a))[1])
    def future_once(path,ids):
        reads.append('future');v=future(path,ids);captured[str(path)]=v;return v
    monkeypatch.setattr(s,'_future_truth',future_once)
    def old_score_after_truth(c):
        assert reads==['known_TEST','source_NV','future','future']
        return legacy(c)
    monkeypatch.setattr(s,'_legacy_score',old_score_after_truth)
    monkeypatch.setattr(m.core,'fit',lambda *a:pytest.fail('Scorer cannot refit source subspaces'))
    monkeypatch.setattr(np.linalg,'eigh',lambda *a:pytest.fail('Scorer cannot eigendecompose query/Us'))
    monkeypatch.setattr(np.linalg,'svd',lambda *a:pytest.fail('Scorer cannot refit nuisance span'))
    monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('Scorer cannot recalibrate'))
    monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Scorer cannot load NN or checkpoint'))
    for key in ('load_role','load_query'):
        monkeypatch.setattr(m.data,key,lambda *a:pytest.fail('Scorer cannot read Us or source IQ'))
    _,_,rows,_=s._prediction(root)
    assert not f['done']['classification_unchanged_from_cached_F0']  # Deliberately different toy old/new F0 graphs, not real replay evidence.
    args=(root,truth['source_known'],truth['source_novel'],truth['clean'],truth['star'])
    result=s.score(*args,tmp_path/'geometry_score')
    assert reads==['known_TEST','source_NV','future','future'] and result['truth_last'] and result['all_16_points_before_truth']
    assert result['legacy_energy_overall_exact'] and result['primary_method']=='F0_domain_vim'
    assert result['source_statistical_learning'] and result['new_neural_updates']==0 and result['adaptation_success'] is None and not result['new_target_DA']
    assert result['ground_stage_success']==(result['ground_source_success'] and result['cross_clean_success'])
    assert 'ground_training_success' not in result and result['source_TX_labels_before_CAL']==0
    assert result['replay_diagnostics']==f['done']['replay_diagnostics'] and not result['classification_unchanged_from_cached_F0']
    for name in m.METHODS:
        assert len(result['methods'][name]['paths'])==4
        for condition in m.CONDITIONS:
            g=result['methods'][name]['paths'][condition]['goals'];assert g['known_minimum']==g['unknown_minimum']==.95
    assert result['methods']['F0_domain_vim']['paths']['cross_star_DURA']['by_day']['2021_03_23']['A_K'] is None
    for c in m.CONDITIONS:
        assert result['paired_primary_vs_controls']['F0_vim'][c]['counts']['known_closed_prediction_changed']==0
    a=rows['cross_star_DURA'];t=captured[str(truth['star'])];k=t['y']<6;u=~k
    for j,name in enumerate(m.METHODS[:3]):
        nc=(a['closed'][k,3]==t['y'][k])&(a['scores'][k,3]>=0);oc=(a['closed'][k,j]==t['y'][k])&(a['scores'][k,j]>=0)
        nr=a['scores'][u,3]<0;orr=a['scores'][u,j]<0;p=result['paired_primary_vs_controls'][name]['cross_star_DURA']
        assert p['primary_column']==3 and p['reference_column']==j
        assert p['counts']['known_correct_accepted_gained']==int((nc&~oc).sum()) and p['counts']['known_correct_accepted_lost']==int((oc&~nc).sum())
        assert p['counts']['unknown_rejected_gained']==int((nr&~orr).sum()) and p['counts']['unknown_rejected_lost']==int((orr&~nr).sum())
    from pathlib import Path
    root=Path(root);marker=root/'complete.json';main=root/'queries/cross_star_DURA/predictions.npz';source=root/'queries/source_clean_DURA/predictions.npz'
    tau=root/'thresholds.npz';cal=root/'calibration/complete.json';model=root/'models/complete.json';state=root/'models/domain_orthogonal/state.npz';head=root/'classifier_head.npz'
    saved={p:p.read_bytes() for p in (marker,main,source,tau,cal,model,state,head)}
    for defect in ('partial','missing_Q','missing_source','private_Q','old_F0_control','old_SSL_control','standard_raw','domain_raw','residual','coordinates','fresh_logits','original_baseline',
            'old_tau','new_tau','geometry_basis','nuisance_overlap','classifier_head','source_role','CAL_private','replay_forged','private_config'):
        with subtests.test(defect=defect):
            reads.clear()
            if defect=='missing_Q':main.unlink()
            elif defect=='missing_source':source.unlink()
            elif defect in ('partial','source_role','CAL_private','replay_forged','private_config'):
                p=model if defect=='source_role' else cal if defect=='CAL_private' else marker;v=json.loads(p.read_text(encoding='utf-8'))
                if defect=='partial':v['all_16_points_complete']=False
                elif defect=='source_role':v['source_role_manifest']['role']='L'
                elif defect=='CAL_private':v['test_inputs_used']=True
                elif defect=='replay_forged':v['replay_diagnostics']['cross_star_DURA']['closed_mismatch_count']+=1
                else:v['config']['U_truth']='private'
                p.write_text(json.dumps(v),encoding='utf-8')
            else:
                p=tau if defect in ('old_tau','new_tau') else state if defect in ('geometry_basis','nuisance_overlap') else head if defect=='classifier_head' else main
                with np.load(p,allow_pickle=False) as z:v={key:z[key].copy() for key in z.files}
                if defect=='private_Q':v['y']=np.zeros(len(v['ids']),np.int64)
                elif defect=='old_F0_control':v['scores'][0,0]+=.1
                elif defect=='old_SSL_control':v['scores'][0,1]+=.1
                elif defect=='standard_raw':v['raw_scores'][0,2]+=.1
                elif defect=='domain_raw':v['raw_scores'][0,3]+=.1
                elif defect=='residual':v['residual_norms'][0,1]+=.1
                elif defect=='coordinates':v['classifier_coordinates'][0]*=-1  # Same norm/residual; only the native-head oracle detects sign inconsistency.
                elif defect=='fresh_logits':
                    v['fresh_F0_logits'][0,0]+=.1;v['default_logits'][0,2:,0]+=.1
                elif defect=='original_baseline':v['baseline_closed'][0]=(v['baseline_closed'][0]+1)%6
                elif defect=='old_tau':v['conditional'][0,0]+=.1
                elif defect=='new_tau':v['conditional'][3,0]=np.nan
                elif defect=='geometry_basis':v['residual_basis'][:,1]=v['residual_basis'][:,0]
                elif defect=='nuisance_overlap':v['nuisance_basis'][:,0]=v['residual_basis'][:,0]
                else:v['normalized_weight'][0]*=-1
                np.savez(p,**v)
            with pytest.raises((ValueError,FileNotFoundError)):s.score(*args,tmp_path/defect)
            assert reads==[]
            for p,value in saved.items():p.write_bytes(value)
    with pytest.raises(FileExistsError):s.score(*args,tmp_path/'geometry_score')
