"""Three-condition truth-last density scoring with fixed P1 identity."""
import importlib
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_class_density_score')


def truth_files(root,s,m,output):
    novel=m._rows(output,'novel');grid=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for scene in s.common.SCENES]
    data=dict(ids=novel['ids'],rx_i=novel['rx_i'],day_i=novel['day_i'],tx=np.asarray([v[0] for v in grid]),scenario=np.asarray([v[2] for v in grid]))
    paths=[root/'source_truth.npz'];np.savez(paths[0],**{k:v[::-1] for k,v in data.items()})
    for name in s.CONDITIONS:
        a=m._target_rows(output,name);n=len(a['ids']);p=root/(name+'_truth.npz');paths.append(p)
        data=dict(ids=a['ids'],y=np.arange(n,dtype=np.int64)%9,receiver=np.full(n,26,np.int64),scenario=np.asarray(['clean' if name=='clean_DURA' else 'practical_low_urban']*n))
        np.savez(p,**{k:v[::-1] for k,v in data.items()})
    return paths


def target(condition='clean_DURA',count=100):
    y=np.repeat(np.arange(9,dtype=np.int64),count);n=len(y);closed=np.minimum(y,5)
    a=dict(ids=np.asarray(['q'+str(i) for i in range(n)]),closed=closed,baseline_closed=closed.copy(),scores=np.ones((n,4),np.float64),rx_i=np.full(n,26,np.int64),day_i=np.full(n,2,np.int64))
    a['scores'][y>=6]=-1.
    truth=dict(y=y,receiver=a['rx_i'],scenario=np.asarray(['clean']*n))
    return a,truth


def test_fixed_primary_and_clean95_star90_acceptance_identity_boundaries():
    s=api();assert s.METHODS==('pooled_known','pooled_ratio','predicted_known','predicted_ratio') and s.PRIMARY=='predicted_ratio'
    a,t=target();a['scores'][np.flatnonzero(t['y']==6)[:15],3]=0.
    clean=s._target(a,t,'clean_DURA',3);assert clean['overall']['R_U']==pytest.approx(.95) and clean['goals']['passed']
    a['scores'][np.flatnonzero(t['y']==6)[15],3]=0.
    assert not s._target(a,t,'clean_DURA',3)['goals']['passed']
    a,t=target();a['scores'][np.flatnonzero(t['y']==6)[:30],3]=0.
    t['scenario']=np.asarray(['practical_low_urban']*len(t['y']))
    star=s._target(a,t,'star_DURA',3)
    assert star['overall']['R_U']==pytest.approx(.90) and star['goals']['overall_passed']
    assert not star['goals']['passed'] and not star['goals']['required_scene_coverage_complete']
    assert star['by_scene']['practical_high']['R_U'] is None
    a,t=target();a['scores'][np.flatnonzero(t['y']==0)[:12],3]=-1.
    assert s._target(a,t,'clean_DURA',3)['goals']['passed']
    a['scores'][np.flatnonzero(t['y']==0)[12],3]=-1.
    r=s._target(a,t,'clean_DURA',3)
    assert r['overall']['C_K']>.90 and r['overall']['original_closed_minus_new_open']>.02 and not r['goals']['passed']


def test_all_three_prediction_conditions_precede_any_truth_read(tmp_path,monkeypatch):
    s=api();reads=[]
    monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source'))
    monkeypatch.setattr(s,'_target_truth',lambda *a:reads.append('Q'))
    monkeypatch.setattr(s,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('missing star predictions')))
    with pytest.raises(ValueError,match='missing star'):s.score(tmp_path/'prediction','source','clean','star',tmp_path/'score')
    assert reads==[] and not (tmp_path/'score').exists()


def test_source_unknown_is_diagnostic_and_missing_target_scene_cannot_be_rescued(tmp_path,monkeypatch):
    s=api();rxs=s.common.RECEIVERS;scenes=s.common.SCENES
    kg=[(y,r,scene) for y in range(6) for r in rxs for scene in scenes];ng=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in rxs for scene in scenes]
    k=dict(ids=np.asarray(['k'+str(i) for i in range(len(kg))]),y=np.asarray([v[0] for v in kg],np.int64),rx_i=np.asarray([v[1] for v in kg],np.int64),day_i=np.arange(len(kg),dtype=np.int64)%3+1,scenario=np.asarray([v[2] for v in kg]),scores=np.ones((len(kg),4),np.float64))
    k['closed']=k['y'].copy();k['baseline_closed']=k['y'].copy()
    n=dict(ids=np.asarray(['n'+str(i) for i in range(len(ng))]),rx_i=np.asarray([v[1] for v in ng],np.int64),day_i=np.arange(len(ng),dtype=np.int64)%3+1,scores=np.ones((len(ng),4),np.float64))
    nt=dict(tx=np.asarray([v[0] for v in ng]),scenario=np.asarray([v[2] for v in ng]))
    clean,ct=target(count=6);star,st=target(count=6);st['scenario']=np.tile(np.asarray(scenes),9)
    rows=dict(known=k,novel=n,clean_DURA=clean,star_DURA=star)
    marker=dict(code_commit='e'*40,real_signal_forward_rows=1728,real_F_forward_rows=1728,G_forward_rows=1728,synthetic_signal_forward_rows=2,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2)
    cal=dict(methods={name:dict(constraints_passed=True) for name in s.METHODS});reads=[]
    monkeypatch.setattr(s,'_prediction',lambda *a:(marker,cal,rows));monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source') or nt)
    monkeypatch.setattr(s,'_target_truth',lambda path,*a:reads.append(path) or (ct if path=='clean' else st))
    result=s.score('prediction','source','clean','star',tmp_path/'pass')
    assert reads==['source','clean','star'] and result['development_gate_passed']
    assert not result['source_unknown_90_diagnostic_passed'] and result['source_known_12_passed']
    assert result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete'] and result['A_N'] is result['H_OW'] is None
    for name in s.METHODS:
        assert len(result['methods'][name]['source']['groups'])==42 and len(result['methods'][name]['star_DURA']['by_scene'])==6
    assert set(result['paired'])=={'conditional_known_vs_pooled','conditional_ratio_vs_pooled','pooled_ratio_vs_known','conditional_ratio_vs_known'}
    star['scores'][st['y']>=6,3]=1.
    result=s.score('prediction','source','clean','star',tmp_path/'main_fail')
    assert all(result['methods'][name]['star_DURA']['goals']['passed'] for name in s.METHODS[:3])
    assert not result['development_gate_passed'] and result['next_step']=='NO_PROMOTION'
    star['scores'][st['y']>=6,3]=-1.;st['scenario'][(st['y']==6)&(st['scenario']=='practical_high')]='practical_mid'
    result=s.score('prediction','source','clean','star',tmp_path/'missing')
    row=result['methods'][s.PRIMARY]['star_DURA'];assert row['overall']['R_U']==1. and row['by_scene']['practical_high']['R_U'] is None
    assert not row['goals']['required_scene_coverage_complete'] and not result['eligible_primary']


def test_actual_three_DURA_predictions_to_independent_CPU_score_no_refit(tmp_path,monkeypatch):
    from test_pamf_phase3_star_class_density_run import fixture
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,pm,backend,public,events,seen):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        paths=truth_files(tmp_path,s,m,c['output_root']);private=[];source=s._source_truth;target_load=s._target_truth
        def first(path,*args):private.append('source');return source(path,*args)
        def query(path,*args):private.append(Path(path).name);return target_load(path,*args)
        monkeypatch.setattr(s,'_source_truth',first);monkeypatch.setattr(s,'_target_truth',query)
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Independent CPU scorer cannot load neural checkpoints'))
        monkeypatch.setattr(m.core.density,'fit_known',lambda *a:pytest.fail('No density refit'))
        monkeypatch.setattr(m.core.density,'fit_background',lambda *a:pytest.fail('No density refit'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No recalibration'))
        result=s.score(c['output_root'],*paths,tmp_path/'score')
        assert private==['source','clean_DURA_truth.npz','star_DURA_truth.npz'] and result['truth_last'] and result['all_three_predictions_before_truth']
        assert result['classification_path']=='P1' and result['identity_baseline_path']=='same_DURA_output_P1' and result['clean_also_uses_DURA']
        assert result['costs']['real_F_forward_rows']==result['costs']['G_forward_rows']==result['costs']['real_signal_forward_rows']==16
        assert result['methods'][s.PRIMARY]['clean_DURA']['overall']['R_U'] is None and not result['development_gate_passed']
        assert result['source_truth_reads']==1 and result['target_truth_reads']==2 and result['final_phase3_complete'] is False
        with pytest.raises(FileExistsError):s.score(c['output_root'],*paths,tmp_path/'score')


def test_missing_test_P0_identity_changed_control_and_class_route_fail_before_truth(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_class_density_run import fixture
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,pm,backend,public,events,seen):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT);out=Path(c['output_root'])
        paths=[out/'complete.json',out/'thresholds.npz',out/'models/density.npz',out/'novel_predictions.npz',out/'targets/clean_DURA/predictions.npz',out/'targets/star_DURA/predictions.npz']
        saved={p:p.read_bytes() for p in paths}
        monkeypatch.setattr(s,'_source_truth',lambda *a:pytest.fail('Source truth before complete three-test validation'))
        monkeypatch.setattr(s,'_target_truth',lambda *a:pytest.fail('Query truth before complete three-test validation'))
        for defect in ('P0_identity','control_tau','density','source_control','conditional_route','old_clean_baseline','private_Q','missing_star'):
            with subtests.test(defect=defect):
                for p,value in saved.items():p.write_bytes(value)
                if defect=='P0_identity':
                    p=out/'complete.json';a=m._read(p);a['model_identity']=pm['model_identity'];p.unlink();m._write(p,a)
                elif defect=='missing_star':(out/'targets/star_DURA/predictions.npz').unlink()
                else:
                    p=(out/'thresholds.npz' if defect=='control_tau' else out/'models/density.npz' if defect=='density' else out/'novel_predictions.npz' if defect=='source_control' else out/'targets/clean_DURA/predictions.npz')
                    with np.load(p) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='control_tau':a['conditional'][0,0]+=.1
                    elif defect=='density':a['known_means'][0,0]+=.1
                    elif defect=='source_control':a['scores'][0,0]+=.1
                    elif defect=='conditional_route':a['raw_scores'][0,3]+=.1
                    elif defect=='old_clean_baseline':a['baseline_closed'][0]=(a['closed'][0]+1)%6
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,'source','clean','star',tmp_path/'score')
                assert not (tmp_path/'score/complete.json').exists()
