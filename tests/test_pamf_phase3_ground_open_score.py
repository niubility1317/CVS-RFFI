"""Ground gates, true-class coverage and truth-last energy scoring; synthetic only."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.ground_open_score')


def test_ground_95_gate_and_fixed_primary_are_independent_of_star_diagnostics(subtests):
    s=api();row=dict(C_K=.94,R_U=.96,original_closed_minus_new_open=.01,coverage={'complete':True})
    assert not s._passed(row,.95,.95) and s._passed(row,.90,.95)
    paths={name:dict(goals={'passed':True}) for name in s.CONDITIONS}
    cal=dict(methods={s.PRIMARY:dict(constraints_passed=True)})
    assert s._decision(paths,cal)==(True,True,True)
    with subtests.test(star_diagnostic_only=True):
        paths['source_star_DURA']['goals']['passed']=False;paths['cross_star_DURA']['goals']['passed']=False
        assert s._decision(paths,cal)==(True,True,True)
    for condition in ('source_clean_DURA','cross_clean_DURA'):
        with subtests.test(primary_failure=condition):
            paths[condition]['goals']['passed']=False
            assert s._decision(paths,cal)[-1] is False
            paths[condition]['goals']['passed']=True
    with subtests.test(missing_class=True):
        row.update(C_K=1.,coverage={'complete':False});assert not s._passed(row,.95,.95)
    with subtests.test(calibration_failure=True):
        cal['methods'][s.PRIMARY]['constraints_passed']=False
        assert s._decision(paths,cal)[-1] is False


def test_energy_uses_each_native_head_and_threshold_class(subtests):
    s=api();logits=np.zeros((2,3,6),np.float32)
    for j in range(3):logits[:,j,j]=np.asarray([3.,4.],np.float32)
    tau=np.arange(18,dtype=np.float64).reshape(3,6)/10
    a=s.evaluation._stack(np.asarray(['a','b']),logits[:,0],(logits[:,1],logits[:,2]),tau)
    s._reconstruct(a,tau)
    np.testing.assert_array_equal(a['closed'],np.tile(np.arange(3),(2,1)))
    for defect in ('energy','old_head_routing','baseline','closed'):
        with subtests.test(defect=defect):
            b={k:v.copy() for k,v in a.items()}
            if defect=='energy':b['raw_scores'][0,2]+=.01
            elif defect=='old_head_routing':b['scores'][:,2]=b['raw_scores'][:,2]-tau[2,b['closed'][:,0]]
            elif defect=='baseline':b['baseline_closed'][0]=5
            else:b['closed'][0,2]=5
            with pytest.raises(ValueError):s._reconstruct(b,tau)


def test_true_TX_macro_missing_classes_and_source_namespace(tmp_path,subtests):
    s=api();y=np.asarray([0]*9+[1,2,3,4,5],np.int64);closed=y.copy();closed[:9]=1
    row=s._metrics(y,closed,np.ones(len(y)),np.asarray(s.common.UNKNOWN_TX),-np.ones(3),y)
    assert row['A_K']==pytest.approx(5/6) and row['known_correct_accepted']==5
    with subtests.test(missing_known_is_NA=True):
        z=y!=5;r=s._metrics(y[z],closed[z],np.ones(z.sum()),np.asarray(s.common.UNKNOWN_TX),-np.ones(3),y[z])
        assert r['A_K'] is None and r['C_K'] is None and r['H_score'] is None and r['R_U']==1.
    path=tmp_path/'known.npz';ids=np.asarray(['k'+str(i) for i in range(6)])
    values=dict(ids=ids[::-1],y=np.arange(6,dtype=np.int64)[::-1],rx_i=np.ones(6,np.int64),day_i=np.ones(6,np.int64),star_scenario=np.asarray([s.common.SCENES[0]]*6))
    np.savez(path,**values);truth=s._source_known_truth(path)
    a=dict(ids=np.concatenate((ids,np.asarray(['n0','n1','n2']))),rx_i=np.ones(9,np.int64),day_i=np.ones(9,np.int64))
    nt=dict(ids=np.asarray(['n0','n1','n2']),tx=np.asarray(s.common.UNKNOWN_TX),rx_i=np.ones(3,np.int64),day_i=np.ones(3,np.int64),scenario=np.asarray([s.common.SCENES[0]]*3))
    t=s._join_source(a,truth,nt,False)
    np.testing.assert_array_equal(t['y'][:6],np.arange(6));np.testing.assert_array_equal(t['tx'][6:],s.common.UNKNOWN_TX)
    assert (t['y'][6:]==-1).all() and t['known'].sum()==6
    with subtests.test(target_namespace_forbidden=True):
        bad=dict(nt,tx=np.asarray([6,7,8],np.int64))
        with pytest.raises(ValueError):s._join_source(a,truth,bad,False)
    with subtests.test(extra_supervision_forbidden=True):
        np.savez(path,**values,receiver=np.ones(6,np.int64))
        with pytest.raises(ValueError):s._source_known_truth(path)


def test_partial_prediction_never_opens_any_of_four_truths(tmp_path,monkeypatch):
    s=api();root=tmp_path/'predict';root.mkdir()
    (root/'complete.json').write_text(json.dumps(dict(status='PARTIAL')),encoding='utf-8')
    for name in ('_source_known_truth','_source_truth','_future_truth'):
        monkeypatch.setattr(s,name,lambda *a:pytest.fail('No private read before all four conditions pass'))
    with pytest.raises(ValueError):s.score(root,'known','novel','clean','star',tmp_path/'score')


def test_actual_twelve_readouts_before_four_truths_and_artifact_corruptions(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_ground_open_run import build_prediction_fixture
    f=build_prediction_fixture(tmp_path,monkeypatch);m=f['module'];root=f['root'];s=api();truth=f['truths']
    assert f['done']['all_12_points_complete'] and f['done']['new_threshold_values']==18
    assert len(f['calibration_calls'])==3
    reads=[];future_truths={};kt=s._source_known_truth;nt=s._source_truth;ft=s._future_truth
    monkeypatch.setattr(s,'_source_known_truth',lambda *a:(reads.append('source_known'),kt(*a))[1])
    monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source_novel'),nt(*a))[1])
    def future(path,ids):
        reads.append('future');a=ft(path,ids);future_truths[str(path)]=a;return a
    monkeypatch.setattr(s,'_future_truth',future)
    monkeypatch.setattr(m.core,'calibrate_known',lambda *a:pytest.fail('Independent CPU scorer cannot recalibrate'))
    for name in ('load_role','load_query'):
        monkeypatch.setattr(m.data,name,lambda *a:pytest.fail('Independent CPU scorer cannot read IQ roles'))
    monkeypatch.setattr(m._train(),'_backend',lambda *a:pytest.fail('Independent CPU scorer cannot load NN'))
    monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('Independent CPU scorer cannot load checkpoint'))
    _,_,rows,_=s._prediction(root)
    assert any(np.any(a['closed'][:,2]!=a['baseline_closed']) for a in rows.values())
    args=(root,truth['source_known'],truth['source_novel'],truth['clean'],truth['star'])
    result=s.score(*args,tmp_path/'score')
    assert reads==['source_known','source_novel','future','future'] and result['truth_last'] and result['all_four_predictions_before_truth']
    assert len(f['truth_reads'])==4 and len(set(f['truth_reads']))==4
    assert result['primary_method']=='SSL_energy' and result['adaptation_success'] is None and not result['new_target_DA']
    assert len(result['methods'])==3 and all(len(r['paths'])==4 for r in result['methods'].values())
    assert result['methods']['SSL_energy']['paths']['cross_star_DURA']['by_day']['2021_03_23']['A_K'] is None
    assert len(result['methods']['SSL_energy']['paths']['source_clean_DURA']['groups'])==24
    assert len(result['methods']['SSL_energy']['paths']['source_star_DURA']['scene_groups'])==42
    assert result['ground_training_success']==(result['ground_source_success'] and result['cross_clean_success'])
    # Literal per-model selections distinguish identity corrections from accepted
    # correctness and reject changes, despite each model having a different head.
    t=future_truths[str(truth['star'])];a=rows['cross_star_DURA'];k=t['y']<6;u=~k
    for j,label in enumerate(m.METHODS[:2]):
        nc=(a['closed'][k,2]==t['y'][k])&(a['scores'][k,2]>=0);oc=(a['closed'][k,j]==t['y'][k])&(a['scores'][k,j]>=0)
        nr=a['scores'][u,2]<0;orr=a['scores'][u,j]<0
        p=result['paired_primary_vs_controls'][label]['cross_star_DURA'];counts=p['counts']
        assert p['primary_column']==2 and p['reference_column']==j
        assert counts['known_correct_accepted_gained']==int((nc&~oc).sum()) and counts['known_correct_accepted_lost']==int((oc&~nc).sum())
        assert counts['unknown_rejected_gained']==int((nr&~orr).sum()) and counts['unknown_rejected_lost']==int((orr&~nr).sum())
    marker=Path(root)/'complete.json';cal=Path(root)/'calibration/complete.json';tau=Path(root)/'thresholds.npz'
    main=Path(root)/'queries/cross_star_DURA/predictions.npz';source=Path(root)/'queries/source_clean_DURA/predictions.npz'
    saved={p:p.read_bytes() for p in (marker,cal,tau,main,source)}
    for defect in ('partial','missing_Q','private_Q','energy','routing','baseline','own_head','tau','calibration','ancestor','private_config','source_metadata'):
        with subtests.test(defect=defect):
            reads.clear()
            if defect=='missing_Q':main.unlink()
            elif defect in ('partial','calibration','ancestor','private_config'):
                path=cal if defect=='calibration' else marker;v=json.loads(path.read_text(encoding='utf-8'))
                if defect=='partial':v['all_thresholds_frozen_before_Q']=False
                elif defect=='calibration':v['unknown_inputs_used']=True
                elif defect=='ancestor':v['model_identities']['SSL']['source_only']=False
                else:v['config']['source_novel_truth']='private'
                path.write_text(json.dumps(v),encoding='utf-8')
            else:
                path=tau if defect=='tau' else source if defect=='source_metadata' else main
                with np.load(path,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
                if defect=='private_Q':v['y']=np.zeros(len(v['ids']),np.int64)
                elif defect=='energy':v['raw_scores'][0,2]+=.1
                elif defect=='routing':v['scores'][0,2]+=.1
                elif defect=='baseline':v['baseline_closed'][0]=(v['baseline_closed'][0]+1)%6
                elif defect=='own_head':v['closed'][0,2]=(v['closed'][0,2]+1)%6
                elif defect=='source_metadata':v['day_i'][0]=1 if v['day_i'][0]!=1 else 2
                else:v['conditional'][2,0]=np.nan
                np.savez(path,**v)
            with pytest.raises((ValueError,FileNotFoundError)):s.score(*args,tmp_path/defect)
            assert reads==[]
            for path,value in saved.items():path.write_bytes(value)
    with pytest.raises(FileExistsError):s.score(*args,tmp_path/'score')
