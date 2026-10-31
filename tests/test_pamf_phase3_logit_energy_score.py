"""Energy/ratio column routing and truth-last producer handoff, synthetic only."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.logit_energy_score')


def test_paired_main_energy_and_control_ratio_use_different_columns(subtests):
    s=api()
    new=dict(ids=np.asarray(['a','b']),y=np.asarray([0,1]),closed=np.asarray([0,0]),scores=np.asarray([[1.,-1.],[1.,-1.]]))
    old=dict(ids=new['ids'],y=new['y'],closed=np.asarray([0,1]),scores=np.asarray([[1.,-1.],[1.,1.]]))
    nn=dict(ids=np.asarray(['u','v']),scores=np.asarray([[-1.,1.],[1.,-1.]]))
    on=dict(ids=nn['ids'],scores=np.asarray([[-1.,1.],[1.,-1.]]))
    # Main energy accepts a correctly and b wrongly; control ratio only b correctly.
    p=s._paired(new,old,nn,on,0,1)
    assert (p['known_correct_accepted_gained'],p['known_correct_accepted_lost'],p['known_correct_accepted_net'])==(1,1,0)
    assert (p['unknown_rejected_gained'],p['unknown_rejected_lost'])==(1,1)
    assert (p['known_closed_correct_gained'],p['known_closed_correct_lost'])==(0,1)
    with subtests.test(control_energy=True):
        q=s._paired(new,old,nn,on,0,0)
        assert q['known_correct_accepted_gained']==0 and q['known_correct_accepted_lost']==1
        assert q['unknown_rejected_gained']==q['unknown_rejected_lost']==0
    with subtests.test(masked=True):
        q=s._paired(new,old,nn,on,0,1,np.asarray([True,False]),np.asarray([True,False]))
        assert q['known_correct_accepted_gained']==q['unknown_rejected_gained']==1


def test_only_primary_energy_can_pass_scientific_gate(subtests):
    s=api();primary=dict(source_DURA=dict(known_12_groups_passed=True),clean_DURA=dict(goals={'passed':True}),star_DURA=dict(goals={'passed':True}))
    cal=dict(methods={s.PRIMARY:dict(constraints_passed=True)})
    pairs={name:dict(star_DURA=dict(delta=dict(A_K=.01,R_U=0.))) for name in s.CONTROL_METHODS}
    assert s._decision(primary,cal,pairs)==(True,True,True,True,True)
    for name in ('UNIFORM_energy','OLD_PCD_predicted_ratio'):
        with subtests.test(known_gain=name):
            pairs[name]['star_DURA']['delta']['A_K']=0.
            assert s._decision(primary,cal,pairs)[-1] is False
            pairs[name]['star_DURA']['delta']['A_K']=.01
    with subtests.test(weak_rejection=False):
        pairs['E0_predicted_ratio']['star_DURA']['delta']['R_U']=-.01
        assert s._decision(primary,cal,pairs)[-1] is False
    pairs['E0_predicted_ratio']['star_DURA']['delta']['R_U']=0.
    with subtests.test(missing_scene=True):
        primary['star_DURA']['goals']['passed']=False
        assert s._decision(primary,cal,pairs)[-1] is False
    primary['star_DURA']['goals']['passed']=True
    with subtests.test(source_guard=False):
        cal['methods'][s.PRIMARY]['constraints_passed']=False
        assert s._decision(primary,cal,pairs)[-1] is False


def test_partial_prediction_metadata_never_connects_truth(tmp_path,monkeypatch):
    s=api();root=tmp_path/'predict';root.mkdir();(root/'complete.json').write_text(json.dumps(dict(schema='pamf_phase3_logit_energy_prediction_v1',status='PARTIAL')),encoding='utf-8')
    monkeypatch.setattr(s,'_source_truth',lambda *a:pytest.fail('Partial predictions must reject before truth'))
    monkeypatch.setattr(s,'_future_truth',lambda *a:pytest.fail('Partial predictions must reject before truth'))
    with pytest.raises(ValueError):s.score(root,'source','clean','star',tmp_path/'score')


def test_actual_eight_readouts_energy_gate_and_CPU_truth_handoff(tmp_path,monkeypatch,subtests):
    # The source-only training completion/loader is explicitly nominal in this
    # fixture. Actual producer, source density fits, risk calibration and gate run.
    from test_pamf_phase3_logit_energy_run import fixture
    with fixture(tmp_path,monkeypatch) as (m,c,pm,flow,calibrated):
        done=m.run(c,'cpu');out=Path(c['output_root']);s=api()
        assert flow==['smoke','L','OE','V','calibrate','calibrate','calibrate','calibrate','novel','clean_DURA','star_DURA']
        assert len(calibrated)==4 and done['source_density_models_fitted']==2 and done['new_threshold_sets_fitted']==4
        assert done['threshold_values']==48 and done['threshold_bytes']==384 and done['primary_method']=='RXPAIR_energy'
        assert done['primary_deployed_neural_parameters']==499944 and done['new_candidate_optimizer_updates']==4440
        assert done['target_U_reads']==done['U_ID_metadata_reads']==0 and not done['classification_unchanged']
        monkeypatch.setattr(s,'COUNTS',dict(m.COUNTS));monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('CPU scorer must not recalibrate'))
        _,_,rows=s._prediction(out)
        for condition in m.CONDITIONS:
            for arm in m.NEW_ARMS:
                a=rows[condition][arm]['known'] if condition=='source_DURA' else rows[condition][arm]
                np.testing.assert_array_equal(a['raw_scores'][:,0],np.logaddexp.reduce(a['default_logits'].astype(np.float64),axis=1))
                assert np.any(a['closed']!=a['baseline_closed'])
        nv=rows['source_DURA']['E0']['novel'];grid=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for scene in s.common.SCENES]
        source=tmp_path/'source_truth.npz';truths=[source]
        np.savez(source,ids=nv['ids'],rx_i=nv['rx_i'],day_i=nv['day_i'],tx=np.asarray([v[0] for v in grid]),scenario=np.asarray([v[2] for v in grid]))
        y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);days=np.asarray(['2021_03_15']*10+['2021_03_23']*8)
        for condition in m.CONDITIONS[1:]:
            a=rows[condition]['E0'];path=tmp_path/(condition+'_truth.npz');truths.append(path)
            np.savez(path,ids=a['ids'],y=y,tx=np.asarray(s.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),day=days,
                scenario=np.asarray(['clean' if condition=='clean_DURA' else 'practical_low_urban']*len(y)))
        reads=[];st=s._source_truth;ft=s._future_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),st(*a))[1]);monkeypatch.setattr(s,'_future_truth',lambda *a:(reads.append('future'),ft(*a))[1])
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('CPU scoring must not load NN'))
        monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('CPU scoring must not load checkpoint'))
        monkeypatch.setattr(m.density,'fit_known',lambda *a:pytest.fail('CPU scoring must not fit known density'))
        monkeypatch.setattr(m.density,'fit_background',lambda *a:pytest.fail('CPU scoring must not fit background'))
        result=s.score(out,*truths,tmp_path/'score')
        assert reads==['source','future','future'] and result['truth_last'] and result['all_three_predictions_before_truth']
        assert result['primary_method']=='RXPAIR_energy' and result['adaptation_success'] is None and not result['new_target_DA']
        assert result['new_projection_updates_total']==0 and result['new_identity_head_updates_total']==4440
        assert result['methods']['RXPAIR_energy']['paths']['star_DURA']['by_day']['2021_03_23']['A_K'] is None
        # Independent literal column choices catch accidental main ratio/index1.
        for label,arm,old_column in (('E0_predicted_ratio','E0',1),('OLD_PCD_predicted_ratio','OLD_PCD',1),('UNIFORM_energy','UNIFORM',0)):
            a,b=rows['star_DURA']['RXPAIR'],rows['star_DURA'][arm];k=y<6;u=~k
            nc=(a['closed'][k]==y[k])&(a['scores'][k,0]>=0);oc=(b['closed'][k]==y[k])&(b['scores'][k,old_column]>=0)
            nr=a['scores'][u,0]<0;orr=b['scores'][u,old_column]<0
            p=result['paired_primary_vs_controls'][label]['star_DURA'];counts=p['counts']
            assert p['primary_column']==0 and p['reference_column']==old_column
            assert counts['known_correct_accepted_gained']==int((nc&~oc).sum()) and counts['known_correct_accepted_lost']==int((oc&~nc).sum())
            assert counts['unknown_rejected_gained']==int((nr&~orr).sum()) and counts['unknown_rejected_lost']==int((orr&~nr).sum())
        main=out/'targets/star_DURA/RXPAIR/predictions.npz';old=out/'targets/star_DURA/OLD_PCD/predictions.npz';marker=out/'complete.json'
        tau=out/'thresholds.npz';fit=out/'models/complete.json';saved={p:p.read_bytes() for p in (main,old,marker,tau,fit)}
        for defect in ('partial','missing_Q','private_Q','old_control','new_energy','new_ratio','original_baseline','own_closed','tau','model_fit','private_config','deployment_count'):
            with subtests.test(defect=defect):
                reads.clear()
                if defect=='missing_Q':main.unlink()
                elif defect in ('partial','private_config','deployment_count','model_fit'):
                    path=fit if defect=='model_fit' else marker;v=json.loads(path.read_text(encoding='utf-8'))
                    if defect=='partial':v['all_thresholds_frozen_before_novel_and_Q']=False
                    elif defect=='private_config':v['config']['U_truth']='private'
                    elif defect=='deployment_count':v['primary_deployed_neural_parameters']=602632
                    else:v['target_Q_fit']=True
                    path.write_text(json.dumps(v),encoding='utf-8')
                else:
                    path=old if defect=='old_control' else tau if defect=='tau' else main
                    with np.load(path,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
                    if defect=='private_Q':v['y']=np.zeros(m.Q_COUNT,np.int64)
                    elif defect=='old_control':v['scores'][0,1]+=.1
                    elif defect=='new_energy':v['raw_scores'][0,0]+=.1
                    elif defect=='new_ratio':v['raw_scores'][0,1]+=.1
                    elif defect=='original_baseline':v['baseline_closed'][0]=(v['baseline_closed'][0]+1)%6
                    elif defect=='own_closed':v['closed'][0]=(v['closed'][0]+1)%6
                    else:v['conditional'][6,0]=np.nan
                    np.savez(path,**v)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*truths,tmp_path/defect)
                assert reads==[]
                for path,value in saved.items():path.write_bytes(value)
        with pytest.raises(FileExistsError):s.score(out,*truths,tmp_path/'score')
