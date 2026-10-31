"""Strict rejection goals, independent F0 baseline and truth-last tail readout."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.ground_tail_score')


def metric(ck=.96,ru=.96):
    return dict(C_K=ck,R_U=ru,original_closed_minus_new_open=.01,coverage={'known_complete':True,'unknown_complete':True,'complete':True})


def path(s,condition):
    scenes=('clean',) if 'clean' in condition else s.common.SCENES
    a=dict(overall=metric(),by_scene={name:metric() for name in scenes})
    if condition.startswith('source'):a.update(groups={name:metric() for name in s.risk.GROUP_NAMES},scene_groups={name:metric() for name in s.common.GROUP_NAMES})
    return a


def test_every_formal_condition_uses95_and_source_alone_never_passes(subtests):
    s=api();paths={c:s._strict_path(path(s,c),c) for c in s.CONDITIONS};cal=dict(methods={s.PRIMARY:dict(constraints_passed=True)})
    assert s._decision(paths,cal)==(True,True,True,True)
    for condition in s.CONDITIONS:
        with subtests.test(old90_is_not_formal=condition):
            a=path(s,condition);a['overall']=metric(.94,.94);bad=dict(paths);bad[condition]=s._strict_path(a,condition)
            assert not s._decision(bad,cal)[-1]
            assert bad[condition]['goals']['known_minimum']==bad[condition]['goals']['unknown_minimum']==.95
    with subtests.test(rare_source_group=True):
        a=path(s,s.CONDITIONS[0]);a['groups']['rxday/3/3']=metric(.94,.99)
        assert not s._strict_path(a,s.CONDITIONS[0])['goals']['passed']
    with subtests.test(required_star_scene=True):
        a=path(s,s.CONDITIONS[3]);a['by_scene'][s.common.SCENES[-1]]=metric(.99,.94)
        assert not s._strict_path(a,s.CONDITIONS[3])['goals']['passed']
    with subtests.test(missing_class_is_not_present_only=True):
        a=path(s,s.CONDITIONS[3]);a['by_scene'][s.common.SCENES[0]]['coverage']['complete']=False
        assert not s._strict_path(a,s.CONDITIONS[3])['goals']['passed']
    with subtests.test(no_primary_control_swap=True):
        cal['methods'][s.PRIMARY]['constraints_passed']=False
        assert s._decision(paths,cal)==(False,True,False,False)


def test_old_SSL_column_is_not_the_original_F0_baseline(subtests):
    s=api();logits=np.zeros((2,3,6),np.float32)
    for j in range(3):logits[:,j,j+1]=3.+j
    closed=logits.argmax(2).astype(np.int64);tau=np.arange(18,dtype=np.float64).reshape(3,6)/10
    raw=np.column_stack([s.risk.energy(logits[:,j]) for j in range(3)])
    baseline=np.zeros((2,6),np.float32);baseline[:,0]=10
    a=dict(ids=np.asarray(['a','b']),default_logits=logits,closed=closed,baseline_logits=baseline,baseline_closed=np.asarray([0,0],np.int64),raw_scores=raw,
        scores=np.column_stack([raw[:,j]-tau[j,closed[:,j]] for j in range(3)]))
    s._reconstruct(a,tau)
    assert not np.array_equal(a['baseline_closed'],a['closed'][:,0])
    parent=dict(ids=a['ids'],default_logits=np.zeros_like(logits),closed=np.zeros_like(closed),baseline_closed=a['baseline_closed'].copy(),raw_scores=np.zeros_like(raw),scores=np.zeros_like(raw))
    parent['default_logits'][:,0]=baseline
    for key in ('default_logits','closed','raw_scores','scores'):parent[key][:,2]=a[key][:,0]
    s._exact_control(a,parent)
    for defect in ('old_logits','old_margin','original_F0_baseline','own_route','energy'):
        with subtests.test(defect=defect):
            b={k:v.copy() for k,v in a.items()}
            if defect=='old_logits':b['default_logits'][0,0,0]+=.1
            elif defect=='old_margin':b['scores'][0,0]+=.1
            elif defect=='original_F0_baseline':b['baseline_closed']=b['closed'][:,0].copy()
            elif defect=='own_route':b['scores'][:,2]=b['raw_scores'][:,2]-tau[2,b['closed'][:,0]]
            else:b['raw_scores'][0,2]+=.1
            with pytest.raises(ValueError):
                s._reconstruct(b,tau);s._exact_control(b,parent)


def test_partial_prediction_rejects_before_all_four_truths(tmp_path,monkeypatch):
    s=api();root=tmp_path/'predict';root.mkdir();(root/'complete.json').write_text(json.dumps({'status':'PARTIAL'}),encoding='utf-8')
    for key in ('_source_known_truth','_source_truth','_future_truth'):
        monkeypatch.setattr(s,key,lambda *a:pytest.fail('Partial tail predictions cannot connect truth'))
    with pytest.raises(ValueError):s.score(root,'known','novel','clean','star',tmp_path/'score')


def test_actual_twelve_tail_points_four_truths_strict_metrics_and_corruptions(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_ground_tail_run import build_prediction_fixture
    f=build_prediction_fixture(tmp_path,monkeypatch);m=f['module'];root=Path(f['root']);truths=f['truths'];s=api()
    assert f['done']['all_12_points_complete'] and f['done']['new_threshold_values']==12 and f['done']['reused_threshold_values']==6
    reads=[];captured={};known=s._source_known_truth;novel=s._source_truth;future=s._future_truth;legacy=s._legacy_score
    monkeypatch.setattr(s,'_source_known_truth',lambda *a:(reads.append('known_TEST'),known(*a))[1])
    monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source_NV'),novel(*a))[1])
    def read_future(path,ids):
        reads.append('future');a=future(path,ids);captured[str(path)]=a;return a
    monkeypatch.setattr(s,'_future_truth',read_future)
    def legacy_after_truth(config):
        assert reads==['known_TEST','source_NV','future','future'],'Old scored results must not enter any pre-truth model/threshold reconstruction'
        return legacy(config)
    monkeypatch.setattr(s,'_legacy_score',legacy_after_truth)
    monkeypatch.setattr(m.calibration,'calibrate_known',lambda *a:pytest.fail('CPU scoring cannot recalibrate'))
    for key in ('load_role','load_query'):
        monkeypatch.setattr(m.data,key,lambda *a:pytest.fail('CPU scoring cannot load IQ or Us roles'))
    monkeypatch.setattr(m._train(),'_backend',lambda *a:pytest.fail('CPU scoring cannot load NN'))
    monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('CPU scoring cannot load checkpoint'))
    _,_,rows,_=s._prediction(root)
    assert any(np.any(a['baseline_closed']!=a['closed'][:,0]) for a in rows.values())
    args=(root,truths['source_known'],truths['source_novel'],truths['clean'],truths['star'])
    result=s.score(*args,tmp_path/'tail_score')
    assert reads==['known_TEST','source_NV','future','future'] and result['truth_last'] and result['all_12_points_before_truth']
    assert result['legacy_SSL_overall_exact'] and result['primary_method']=='DTAIL_energy'
    assert result['adaptation_success'] is None and not result['new_target_DA'] and result['full_goal_status']=='ACTIVE'
    assert result['ground_training_success']==(result['ground_source_success'] and result['cross_clean_success'])
    assert result['scientific_success']==result['frozen_rejection_success']
    for name in m.METHODS:
        for condition in m.CONDITIONS:
            goals=result['methods'][name]['paths'][condition]['goals']
            assert goals['known_minimum']==goals['unknown_minimum']==.95
    assert result['methods']['DTAIL_energy']['paths']['cross_star_DURA']['by_day']['2021_03_23']['A_K'] is None
    t=captured[str(truths['star'])];a=rows['cross_star_DURA'];k=t['y']<6;u=~k
    for j,name in enumerate(m.METHODS[:2]):
        nc=(a['closed'][k,2]==t['y'][k])&(a['scores'][k,2]>=0);oc=(a['closed'][k,j]==t['y'][k])&(a['scores'][k,j]>=0)
        nr=a['scores'][u,2]<0;orr=a['scores'][u,j]<0;p=result['paired_primary_vs_controls'][name]['cross_star_DURA']['counts']
        assert p['known_correct_accepted_gained']==int((nc&~oc).sum()) and p['known_correct_accepted_lost']==int((oc&~nc).sum())
        assert p['unknown_rejected_gained']==int((nr&~orr).sum()) and p['unknown_rejected_lost']==int((orr&~nr).sum())
    marker=root/'complete.json';main=root/'queries/cross_star_DURA/predictions.npz';source=root/'queries/source_clean_DURA/predictions.npz'
    tau=root/'thresholds.npz';cal=root/'calibration/complete.json';saved={p:p.read_bytes() for p in (marker,main,source,tau,cal)}
    for defect in ('partial','missing_Q','missing_source','private_Q','old_control','energy','own_route','F0_logits','F0_baseline','old_tau','new_tau','CAL_private','native_ancestor','private_config','training_cost'):
        with subtests.test(defect=defect):
            reads.clear()
            if defect=='missing_Q':main.unlink()
            elif defect=='missing_source':source.unlink()
            elif defect in ('partial','CAL_private','native_ancestor','private_config','training_cost'):
                path=cal if defect=='CAL_private' else marker;v=json.loads(path.read_text(encoding='utf-8'))
                if defect=='partial':v['all_thresholds_frozen_before_Q']=False
                elif defect=='CAL_private':v['test_inputs_used']=True
                elif defect=='native_ancestor':v['model_identities']['DTAIL']['source_U_s_used']=True
                elif defect=='training_cost':v['training_costs'].pop('G_forward_rows')
                else:v['config']['truth']='private'
                path.write_text(json.dumps(v),encoding='utf-8')
            else:
                path=tau if defect in ('old_tau','new_tau') else main
                with np.load(path,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
                if defect=='private_Q':v['y']=np.zeros(len(v['ids']),np.int64)
                elif defect=='old_control':v['scores'][0,0]+=.1
                elif defect=='energy':v['raw_scores'][0,2]+=.1
                elif defect=='own_route':v['scores'][0,2]+=.1
                elif defect=='F0_logits':v['baseline_logits'][0,0]+=.1
                elif defect=='F0_baseline':v['baseline_closed'][0]=(v['baseline_closed'][0]+1)%6
                elif defect=='old_tau':v['conditional'][0,0]+=.1
                else:v['conditional'][2,0]=np.nan
                np.savez(path,**v)
            with pytest.raises((ValueError,FileNotFoundError)):s.score(*args,tmp_path/defect)
            assert reads==[]
            for path,value in saved.items():path.write_bytes(value)
    with pytest.raises(FileExistsError):s.score(*args,tmp_path/'tail_score')
