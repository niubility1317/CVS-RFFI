"""Centered default-logit evidence, exact DMEAN control and truth-last scoring."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.ground_centered_score')


def metric(ck=.96,ru=.96):
    return dict(C_K=ck,R_U=ru,original_closed_minus_new_open=.01,coverage=dict(known_complete=True,unknown_complete=True,complete=True))


def path(s,condition):
    scenes=('clean',) if 'clean' in condition else s.common.SCENES
    result=dict(overall=metric(),by_scene={name:metric() for name in scenes})
    if condition.startswith('source'):
        result.update(groups={name:metric() for name in s.risk.GROUP_NAMES},scene_groups={name:metric() for name in s.common.GROUP_NAMES})
    return result


def test_centered_primary_keeps_formal95_four_paths_and_complete_classes(subtests):
    s=api();assert s.PRIMARY=='CPAIR_centered' and len(s.METHODS)==4
    paths={c:s._strict_path(path(s,c),c) for c in s.CONDITIONS}
    cal=dict(methods={s.PRIMARY:dict(constraints_passed=True),'DMEAN_energy':dict(constraints_passed=True)})
    assert s._decision(paths,cal)==(True,True,True,True)
    for condition in s.CONDITIONS:
        with subtests.test(condition=condition):
            v=path(s,condition);v['overall']=metric(.949,.949);bad=dict(paths);bad[condition]=s._strict_path(v,condition)
            assert not s._decision(bad,cal)[-1]
            assert bad[condition]['goals']['known_minimum']==bad[condition]['goals']['unknown_minimum']==.95
    for defect in ('rare_RXday','missing_scene_class','F0_loss','primary_CAL_failed'):
        with subtests.test(defect=defect):
            bad=dict(paths);c=s.CONDITIONS[0] if defect=='rare_RXday' else s.CONDITIONS[-1];v=path(s,c)
            if defect=='rare_RXday':v['groups']['rxday/3/3']=metric(.94,.99)
            elif defect=='missing_scene_class':v['by_scene'][s.common.SCENES[0]]['coverage']['complete']=False
            elif defect=='F0_loss':v['overall']['original_closed_minus_new_open']=.021
            else:
                badcal=json.loads(json.dumps(cal));badcal['methods'][s.PRIMARY]['constraints_passed']=False
                assert s._decision(paths,badcal)==(False,True,False,False);continue
            bad[c]=s._strict_path(v,c);assert not s._decision(bad,cal)[-1]


def test_four_raw_columns_use_energy_or_centered_and_each_own_class(subtests):
    s=api();n=3;logits=np.zeros((n,4,6),np.float32)
    logits[:,0]=np.asarray([1.,4.,-2.,3.,0.,2.],np.float32);logits[:,1]=logits[:,0]
    logits[:,2]=np.asarray([6.,1.,-4.,0.,2.,3.],np.float32);logits[:,3]=logits[:,2]
    logits[1]+=np.float32(8.);logits[2]-=np.float32(2.)
    closed=logits.argmax(2).astype(np.int64);tau=np.arange(24,dtype=np.float64).reshape(4,6)/10
    energy=np.column_stack([s.risk.energy(logits[:,j]) for j in range(4)])
    raw=energy.copy()
    for j in (1,3):
        relative=logits[:,j].astype(np.float64);relative-=relative.max(1,keepdims=True)
        raw[:,j]=np.log(np.exp(relative).sum(1))-relative.mean(1)-np.log(6.)
    baseline=np.zeros((n,6),np.float32);baseline[:,5]=9.
    a=dict(ids=np.asarray(['a','b','c']),default_logits=logits,closed=closed,baseline_logits=baseline,baseline_closed=baseline.argmax(1).astype(np.int64),
        raw_scores=raw,scores=np.column_stack([raw[:,j]-tau[j,closed[:,j]] for j in range(4)]))
    s._reconstruct(a,tau);assert not np.array_equal(a['baseline_closed'],closed[:,0])
    np.testing.assert_allclose(raw[:,1],raw[0,1],rtol=0,atol=2e-15)
    assert not np.array_equal(raw[:,0],raw[:,1])
    parent=dict(ids=a['ids'],baseline_logits=baseline.copy(),baseline_closed=a['baseline_closed'].copy(),
        default_logits=logits[:,:3].copy(),closed=closed[:,:3].copy(),raw_scores=raw[:,:3].copy(),scores=a['scores'][:,:3].copy())
    for key in ('default_logits','closed','raw_scores','scores'):parent[key][:,1]=a[key][:,0]
    s._exact_control(a,parent)
    for defect in ('energy_as_centered','CPAIR_raw','own_route','old_margin','F0_baseline','wrong_F0_logits','same_model_logits','nonfinite_tau'):
        with subtests.test(defect=defect):
            b={k:v.copy() for k,v in a.items()};bad_tau=tau.copy()
            if defect=='energy_as_centered':b['raw_scores'][:,3]=energy[:,3]
            elif defect=='CPAIR_raw':b['raw_scores'][0,2]+=.1
            elif defect=='own_route':b['scores'][:,3]=raw[:,3]-tau[3,closed[:,0]]
            elif defect=='old_margin':b['scores'][0,0]+=.1
            elif defect=='F0_baseline':b['baseline_closed']=closed[:,0].copy()
            elif defect=='wrong_F0_logits':b['baseline_logits'][0,0]+=.1
            elif defect=='same_model_logits':b['default_logits'][0,1,0]+=.1
            else:bad_tau[1,0]=np.nan
            with pytest.raises(ValueError):s._reconstruct(b,bad_tau);s._exact_control(b,parent)


def test_incomplete_sixteen_points_cannot_read_any_private_truth(tmp_path,monkeypatch):
    s=api();root=tmp_path/'predict';root.mkdir();(root/'complete.json').write_text(json.dumps(dict(status='PARTIAL')),encoding='utf-8')
    for key in ('_source_known_truth','_source_truth','_future_truth'):
        monkeypatch.setattr(s,key,lambda *a:pytest.fail('Incomplete predictions cannot connect four truths'))
    with pytest.raises(ValueError):s.score(root,'known','novel','clean','star',tmp_path/'score')


def test_actual_sixteen_centered_predictions_four_truths_and_protocol_damage(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_ground_centered_run import build_prediction_fixture
    f=build_prediction_fixture(tmp_path,monkeypatch);m=f['module'];root=Path(f['root']);truth=f['truths'];s=api()
    assert f['done']['all_16_points_complete'] and f['done']['new_threshold_values']==18 and f['done']['reused_threshold_values']==6
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
    monkeypatch.setattr(s.risk,'calibrate_known',lambda *a:pytest.fail('CPU cannot fit CAL thresholds'))
    for key in ('load_role','load_query'):
        monkeypatch.setattr(m.data,key,lambda *a:pytest.fail('CPU cannot open source/Us/query IQ'))
    monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('CPU cannot load a neural checkpoint'))
    monkeypatch.setattr(m._train(),'_backend',lambda *a:pytest.fail('CPU cannot construct original or new NN'))
    _,_,rows,_=s._prediction(root)
    assert any(np.any(a['baseline_closed']!=a['closed'][:,0]) for a in rows.values())
    args=(root,truth['source_known'],truth['source_novel'],truth['clean'],truth['star'])
    result=s.score(*args,tmp_path/'centered_score')
    assert reads==['known_TEST','source_NV','future','future'] and result['all_16_points_before_truth'] and result['truth_last']
    assert result['legacy_energy_overall_exact'] and result['primary_method']=='CPAIR_centered'
    assert result['ground_stage_success']==(result['ground_source_success'] and result['cross_clean_success'])
    assert result['scientific_success']==result['frozen_rejection_success']
    assert result['adaptation_success'] is None and not result['new_target_DA'] and result['full_goal_status']=='ACTIVE'
    assert result['combined_new_real_student_forward_rows']==f['done']['training_costs']['real_student_forward_rows']+f['done']['real_student_forward_rows']
    for name in s.METHODS:
        assert len(result['methods'][name]['paths'])==4
        for condition in s.CONDITIONS:
            assert result['methods'][name]['paths'][condition]['goals']['known_minimum']==.95
            assert result['methods'][name]['paths'][condition]['goals']['unknown_minimum']==.95
    assert result['methods'][s.PRIMARY]['paths']['cross_star_DURA']['by_day']['2021_03_23']['A_K'] is None
    a=rows['cross_star_DURA'];t=captured[str(truth['star'])];k=t['y']<6;u=~k
    for j,name in enumerate(s.METHODS[:3]):
        new=(a['closed'][k,3]==t['y'][k])&(a['scores'][k,3]>=0);ref=(a['closed'][k,j]==t['y'][k])&(a['scores'][k,j]>=0)
        nr=a['scores'][u,3]<0;rr=a['scores'][u,j]<0;p=result['paired_primary_vs_controls'][name]['cross_star_DURA']
        assert p['primary_column']==3 and p['reference_column']==j
        assert p['counts']['known_correct_accepted_gained']==int((new&~ref).sum()) and p['counts']['known_correct_accepted_lost']==int((ref&~new).sum())
        assert p['counts']['unknown_rejected_gained']==int((nr&~rr).sum()) and p['counts']['unknown_rejected_lost']==int((rr&~nr).sum())
    for c in s.CONDITIONS:assert result['paired_primary_vs_controls']['CPAIR_energy'][c]['counts']['known_closed_prediction_changed']==0
    marker=root/'complete.json';query=root/'queries/cross_star_DURA/predictions.npz';source=root/'queries/source_clean_DURA/predictions.npz'
    tau=root/'thresholds.npz';cal=root/'calibration/complete.json';saved={p:p.read_bytes() for p in (marker,query,source,tau,cal)}
    for defect in ('partial','missing_Q','missing_source','private_Q','old_energy','legacy_centered','new_energy','new_centered','own_route','F0_logits','F0_baseline',
            'old_tau','new_tau','CAL_private','native_ancestor','private_config','training_cost'):
        with subtests.test(defect=defect):
            reads.clear()
            if defect=='missing_Q':query.unlink()
            elif defect=='missing_source':source.unlink()
            elif defect in ('partial','CAL_private','native_ancestor','private_config','training_cost'):
                p=cal if defect=='CAL_private' else marker;v=json.loads(p.read_text(encoding='utf-8'))
                if defect=='partial':v['all_thresholds_frozen_before_Q']=False
                elif defect=='CAL_private':v['test_inputs_used']=True
                elif defect=='native_ancestor':v['model_identities']['CPAIR']['target_U_reads']=1
                elif defect=='training_cost':v['training_costs'].pop('G_forward_rows')
                else:v['config']['truth']='private'
                p.write_text(json.dumps(v),encoding='utf-8')
            else:
                p=tau if defect in ('old_tau','new_tau') else query
                with np.load(p,allow_pickle=False) as z:v={key:z[key].copy() for key in z.files}
                if defect=='private_Q':v['y']=np.zeros(len(v['ids']),np.int64)
                elif defect=='old_energy':v['scores'][0,0]+=.1
                elif defect=='legacy_centered':v['raw_scores'][0,1]+=.1
                elif defect=='new_energy':v['raw_scores'][0,2]+=.1
                elif defect=='new_centered':v['raw_scores'][0,3]+=.1
                elif defect=='own_route':v['scores'][0,3]+=.1
                elif defect=='F0_logits':v['baseline_logits'][0,0]+=.1
                elif defect=='F0_baseline':v['baseline_closed'][0]=(v['baseline_closed'][0]+1)%6
                elif defect=='old_tau':v['conditional'][0,0]+=.1
                else:v['conditional'][3,0]=np.nan
                np.savez(p,**v)
            with pytest.raises((ValueError,FileNotFoundError)):s.score(*args,tmp_path/defect)
            assert reads==[]
            for p,value in saved.items():p.write_bytes(value)
    with pytest.raises(FileExistsError):s.score(*args,tmp_path/'centered_score')
