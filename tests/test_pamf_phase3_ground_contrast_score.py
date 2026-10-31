"""Six native readouts, exact CPAIR controls and four-truth isolation."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.ground_contrast_score')


def metric(ck=.96,ru=.96):
    return dict(C_K=ck,R_U=ru,original_closed_minus_new_open=.01,coverage=dict(known_complete=True,unknown_complete=True,complete=True))


def path(s,c):
    a=dict(overall=metric(),by_scene={x:metric() for x in (('clean',) if 'clean' in c else s.common.SCENES)})
    if c.startswith('source'):a.update(groups={x:metric() for x in s.risk.GROUP_NAMES},scene_groups={x:metric() for x in s.common.GROUP_NAMES})
    return a


def test_fixed_OECON_primary_requires_all95_paths_and_complete_rosters(subtests):
    s=api();assert s.PRIMARY=='OECON_centered' and s.METHODS[5]==s.PRIMARY
    paths={c:s._strict_path(path(s,c),c) for c in s.CONDITIONS};cal=dict(methods={s.PRIMARY:dict(constraints_passed=True)})
    assert s._decision(paths,cal)==(True,True,True,True)
    for c in s.CONDITIONS:
        with subtests.test(condition=c):
            v=path(s,c);v['overall']=metric(.94,.94);bad=dict(paths);bad[c]=s._strict_path(v,c)
            assert not s._decision(bad,cal)[-1] and bad[c]['goals']['unknown_minimum']==.95
    for defect in ('rare_source_group','missing_star_class','F0_loss','control_cannot_rescue_CAL'):
        with subtests.test(defect=defect):
            if defect=='control_cannot_rescue_CAL':
                badcal=dict(methods={s.PRIMARY:dict(constraints_passed=False),'KCON_centered':dict(constraints_passed=True)})
                assert s._decision(paths,badcal)==(False,True,False,False);continue
            c=s.CONDITIONS[0] if defect=='rare_source_group' else s.CONDITIONS[-1];v=path(s,c)
            if defect=='rare_source_group':v['groups']['rxday/3/3']=metric(.94,.99)
            elif defect=='missing_star_class':v['by_scene'][s.common.SCENES[0]]['coverage']['complete']=False
            else:v['overall']['original_closed_minus_new_open']=.021
            bad=dict(paths);bad[c]=s._strict_path(v,c);assert not s._decision(bad,cal)[-1]


def test_six_energy_centered_columns_use_own_classes_and_old_CPAIR2_3(subtests):
    s=api();n=3;logits=np.zeros((n,6,6),np.float32)
    for j,values in enumerate(([1.,4.,-2.,3.,0.,2.],[6.,1.,-4.,0.,2.,3.],[0.,-4.,8.,1.,2.,3.])):
        logits[:,2*j]=np.asarray(values,np.float32);logits[:,2*j+1]=logits[:,2*j]
    logits[1]+=np.float32(8.);logits[2]-=np.float32(2.);closed=logits.argmax(2).astype(np.int64)
    tau=np.arange(36,dtype=np.float64).reshape(6,6)/10;energy=np.column_stack([s.risk.energy(logits[:,j]) for j in range(6)]);raw=energy.copy()
    for j in (1,3,5):
        relative=logits[:,j].astype(np.float64);relative-=relative.max(1,keepdims=True)
        raw[:,j]=np.log(np.exp(relative).sum(1))-relative.mean(1)-np.log(6.)
    baseline=np.zeros((n,6),np.float32);baseline[:,5]=9.
    a=dict(ids=np.asarray(['a','b','c']),default_logits=logits,closed=closed,baseline_logits=baseline,baseline_closed=baseline.argmax(1).astype(np.int64),
        raw_scores=raw,scores=np.column_stack([raw[:,j]-tau[j,closed[:,j]] for j in range(6)]))
    s._reconstruct(a,tau)
    parent=dict(ids=a['ids'],baseline_logits=baseline.copy(),baseline_closed=a['baseline_closed'].copy())
    for key in ('default_logits','closed','raw_scores','scores'):
        parent[key]=np.concatenate((a[key][:,2:4],a[key][:,:2]),axis=1)
    s._exact_control(a,parent);assert not np.array_equal(closed[:,0],a['baseline_closed'])
    for defect in ('energy_as_centered','KCON_raw','OECON_raw','own_route','old_energy','old_centered','F0_baseline','F0_logits','same_model_pair','nonfinite_tau','wrong_old_columns'):
        with subtests.test(defect=defect):
            b={k:v.copy() for k,v in a.items()};p={k:v.copy() for k,v in parent.items()};bad_tau=tau.copy()
            if defect=='energy_as_centered':b['raw_scores'][:,5]=energy[:,5]
            elif defect=='KCON_raw':b['raw_scores'][0,2]+=.1
            elif defect=='OECON_raw':b['raw_scores'][0,4]+=.1
            elif defect=='own_route':b['scores'][:,5]=raw[:,5]-tau[5,closed[:,0]]
            elif defect=='old_energy':b['scores'][0,0]+=.1
            elif defect=='old_centered':b['scores'][0,1]+=.1
            elif defect=='F0_baseline':b['baseline_closed']=closed[:,0].copy()
            elif defect=='F0_logits':b['baseline_logits'][0,0]+=.1
            elif defect=='same_model_pair':b['default_logits'][0,3,0]+=.1
            elif defect=='wrong_old_columns':
                for key in ('default_logits','closed','raw_scores','scores'):b[key][:,:2]=p[key][:,:2]
            else:bad_tau[5,0]=np.nan
            with pytest.raises(ValueError):s._reconstruct(b,bad_tau);s._exact_control(b,p)


def test_partial_twentyfour_points_keeps_all_four_truths_closed(tmp_path,monkeypatch):
    s=api();root=tmp_path/'predict';root.mkdir();(root/'complete.json').write_text(json.dumps(dict(status='PARTIAL')),encoding='utf-8')
    for key in ('_source_known_truth','_source_truth','_future_truth'):monkeypatch.setattr(s,key,lambda *a:pytest.fail('No truth before24 complete points'))
    with pytest.raises(ValueError):s.score(root,'known','novel','clean','star',tmp_path/'score')


def test_actual_twentyfour_predictions_four_truths_five_controls_and_damage(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_ground_contrast_run import build_prediction_fixture
    f=build_prediction_fixture(tmp_path,monkeypatch);m=f['module'];root=Path(f['root']);truth=f['truths'];s=api()
    assert f['done']['all_24_points_complete'] and f['done']['new_threshold_values']==24 and f['done']['reused_threshold_values']==12
    reads=[];captured={};known=s._source_known_truth;novel=s._source_truth;future=s._future_truth;legacy=s._legacy_score
    monkeypatch.setattr(s,'_source_known_truth',lambda *a:(reads.append('known_TEST'),known(*a))[1])
    monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source_NV'),novel(*a))[1])
    def future_once(path,ids):
        reads.append('future');v=future(path,ids);captured[str(path)]=v;return v
    monkeypatch.setattr(s,'_future_truth',future_once)
    def legacy_after_truth(c):
        assert reads==['known_TEST','source_NV','future','future']
        return legacy(c)
    monkeypatch.setattr(s,'_legacy_score',legacy_after_truth)
    monkeypatch.setattr(s.risk,'calibrate_known',lambda *a:pytest.fail('CPU cannot fit source CAL thresholds'))
    for key in ('load_role','load_query'):monkeypatch.setattr(m.data,key,lambda *a:pytest.fail('CPU cannot reopen source/Us/query IQ'))
    monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('CPU cannot load a neural checkpoint'))
    monkeypatch.setattr(m._train(),'_backend',lambda *a:pytest.fail('CPU cannot construct original or new NN'))
    _,_,rows,_=s._prediction(root)
    assert any(np.any(a['baseline_closed']!=a['closed'][:,0]) for a in rows.values())
    args=(root,truth['source_known'],truth['source_novel'],truth['clean'],truth['star'])
    result=s.score(*args,tmp_path/'contrast_score')
    assert reads==['known_TEST','source_NV','future','future'] and result['truth_last'] and result['all_24_points_before_truth']
    assert result['legacy_energy_overall_exact'] and result['primary_method']=='OECON_centered'
    assert result['ground_stage_success']==(result['ground_source_success'] and result['cross_clean_success'])
    assert result['scientific_success']==result['frozen_rejection_success'] and result['adaptation_success'] is None and not result['new_target_DA']
    assert result['full_goal_status']=='ACTIVE' and result['combined_new_real_student_forward_rows']==f['done']['training_costs']['real_student_forward_rows']+f['done']['real_student_forward_rows']
    assert f['done']['real_legacy_forward_rows']==result['prediction_costs']['real_legacy_forward_rows']==0
    for name in s.METHODS:
        assert len(result['methods'][name]['paths'])==4
        for c in s.CONDITIONS:assert result['methods'][name]['paths'][c]['goals']['known_minimum']==result['methods'][name]['paths'][c]['goals']['unknown_minimum']==.95
    assert result['methods'][s.PRIMARY]['paths']['cross_star_DURA']['by_day']['2021_03_23']['A_K'] is None
    a=rows['cross_star_DURA'];t=captured[str(truth['star'])];k=t['y']<6;u=~k
    for j,name in enumerate(s.METHODS[:5]):
        nr=(a['closed'][k,5]==t['y'][k])&(a['scores'][k,5]>=0);orr=(a['closed'][k,j]==t['y'][k])&(a['scores'][k,j]>=0)
        nu=a['scores'][u,5]<0;ou=a['scores'][u,j]<0;p=result['paired_primary_vs_controls'][name]['cross_star_DURA']
        assert p['primary_column']==5 and p['reference_column']==j
        assert p['counts']['known_correct_accepted_gained']==int((nr&~orr).sum()) and p['counts']['known_correct_accepted_lost']==int((orr&~nr).sum())
        assert p['counts']['unknown_rejected_gained']==int((nu&~ou).sum()) and p['counts']['unknown_rejected_lost']==int((ou&~nu).sum())
    for c in s.CONDITIONS:assert result['paired_primary_vs_controls']['OECON_energy'][c]['counts']['known_closed_prediction_changed']==0
    marker=root/'complete.json';query=root/'queries/cross_star_DURA/predictions.npz';source=root/'queries/source_clean_DURA/predictions.npz'
    tau=root/'thresholds.npz';cal=root/'calibration/complete.json';saved={p:p.read_bytes() for p in (marker,query,source,tau,cal)}
    defects=('partial','missing_Q','missing_source','private_Q','old_energy','old_centered','KCON_energy','KCON_centered','OECON_energy','OECON_centered',
        'own_route','F0_logits','F0_baseline','old_energy_tau','old_centered_tau','new_tau','CAL_private','native_ancestor','private_config','training_cost')
    for defect in defects:
        with subtests.test(defect=defect):
            reads.clear()
            if defect=='missing_Q':query.unlink()
            elif defect=='missing_source':source.unlink()
            elif defect in ('partial','CAL_private','native_ancestor','private_config','training_cost'):
                p=cal if defect=='CAL_private' else marker;v=json.loads(p.read_text(encoding='utf-8'))
                if defect=='partial':v['all_thresholds_frozen_before_Q']=False
                elif defect=='CAL_private':v['test_inputs_used']=True
                elif defect=='native_ancestor':v['model_identities']['OECON']['target_U_reads']=1
                elif defect=='training_cost':v['training_costs'].pop('G_forward_rows')
                else:v['config']['truth']='private'
                p.write_text(json.dumps(v),encoding='utf-8')
            else:
                p=tau if defect in ('old_energy_tau','old_centered_tau','new_tau') else query
                with np.load(p,allow_pickle=False) as z:v={key:z[key].copy() for key in z.files}
                if defect=='private_Q':v['y']=np.zeros(len(v['ids']),np.int64)
                elif defect=='old_energy':v['scores'][0,0]+=.1
                elif defect=='old_centered':v['scores'][0,1]+=.1
                elif defect in ('KCON_energy','KCON_centered','OECON_energy','OECON_centered'):v['raw_scores'][0,s.METHODS.index(defect)]+=.1
                elif defect=='own_route':v['scores'][0,5]+=.1
                elif defect=='F0_logits':v['baseline_logits'][0,0]+=.1
                elif defect=='F0_baseline':v['baseline_closed'][0]=(v['baseline_closed'][0]+1)%6
                elif defect=='old_energy_tau':v['conditional'][0,0]+=.1
                elif defect=='old_centered_tau':v['conditional'][1,0]+=.1
                else:v['conditional'][5,0]=np.nan
                np.savez(p,**v)
            with pytest.raises((ValueError,FileNotFoundError)):s.score(*args,tmp_path/defect)
            assert reads==[]
            for p,value in saved.items():p.write_bytes(value)
    with pytest.raises(FileExistsError):s.score(*args,tmp_path/'contrast_score')
