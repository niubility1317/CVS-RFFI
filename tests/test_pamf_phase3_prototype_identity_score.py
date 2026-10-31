"""Prototype identity keeps independent F0 protection and truth-last scoring."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.prototype_identity_score')


def test_prototype_closed_is_not_original_F0_and_missing_date_stays_NA():
    s=api();y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);n=len(y)
    a=dict(closed=(np.minimum(y,5)+1)%6,baseline_closed=np.minimum(y,5),scores=np.where(y[:,None]<6,1.,-1.)*np.ones((n,2)))
    truth=dict(y=y,receiver=np.full(n,34,np.int64),day=np.asarray(['2021_03_15']*10+['2021_03_23']*8),scenario=np.asarray(['clean']*n))
    result=s.fresh._target(a,truth,'clean_DURA',1)
    assert result['overall']['original_F0_macro_closed_accuracy']==1.
    assert result['overall']['candidate_macro_closed_accuracy']==0. and result['overall']['original_closed_minus_new_open']==1.
    assert result['by_day']['2021_03_23']['A_K'] is result['by_day']['2021_03_23']['C_K'] is result['by_day']['2021_03_23']['H_score'] is None
    assert result['by_day']['2021_03_23']['R_U']==1.


def test_macro12_new_closed_and_original_baseline_remain_separate():
    s=api();from experiments.pamf_phase3_open_world import star_macro12_risk as risk
    y=np.tile(np.arange(6,dtype=np.int64),30);rx=np.repeat(np.asarray(s.common.RECEIVERS,np.int64),36)
    scene=np.tile(np.repeat(np.asarray(s.common.SCENES),6),5);closed=(y+1)%6;score=closed.astype(np.float64)
    ids=np.asarray(['known-'+str(i) for i in range(len(y))])
    _,actual=risk.calibrate_known(ids,y,closed,y,score,rx,scene)
    _,wrong=risk.calibrate_known(ids,y,closed,closed,score,rx,scene)
    assert not actual['constraints_passed'] and actual['zero_rejection_fallback']
    assert actual['groups']['overall']['original_closed_minus_new_open']==1.
    assert wrong['constraints_passed']  # Replacing F0 with the poorer prototype would conceal the loss.


def fixture(root):
    s=api();scenes=s.common.SCENES
    y=np.tile(np.arange(6,dtype=np.int64),30);rx=np.repeat(np.asarray(s.common.RECEIVERS,np.int64),36)
    scene=np.tile(np.repeat(np.asarray(scenes),6),5)
    k=dict(ids=np.asarray(['known-'+str(i) for i in range(len(y))]),y=y,rx_i=rx,day_i=np.tile(np.repeat([1,2,3],12),5).astype(np.int64),scenario=scene,
        closed=y.copy(),baseline_closed=y.copy(),scores=np.ones((len(y),2)))
    uy=np.tile(np.arange(3),30);nv=dict(ids=np.asarray(['novel-'+str(i) for i in range(len(uy))]),rx_i=np.repeat(np.asarray(s.common.RECEIVERS,np.int64),18),
        day_i=np.tile(np.repeat([1,2,3],6),5).astype(np.int64),closed=np.zeros(len(uy),np.int64),baseline_closed=np.zeros(len(uy),np.int64),scores=-np.ones((len(uy),2)))
    nt=root/'source_truth.npz';np.savez(nt,ids=nv['ids'],tx=np.asarray(s.common.UNKNOWN_TX)[uy],rx_i=nv['rx_i'],day_i=nv['day_i'],scenario=np.tile(np.repeat(np.asarray(scenes),3),5))
    ys=[];ss=[];days=[]
    for d in s.fresh.DAYS:
        for sc in scenes:
            for c in range(10):
                if d=='2021_03_23' and c in (3,5):continue
                ys.append(c);ss.append(sc);days.append(d)
    qy=np.asarray(ys,np.int64);ids=np.asarray(['Q-'+str(i) for i in range(len(qy))]);closed=np.minimum(qy,5).astype(np.int64)
    old_closed=closed.copy();old_closed[np.flatnonzero(qy==0)[0]]=1
    q=dict(ids=ids,closed=closed,baseline_closed=old_closed,scores=np.where(qy[:,None]<6,1.,-1.)*np.ones((len(qy),2)))
    old=dict(q,closed=old_closed);rows=dict(source_V=dict(OLD_PCD=k,PROTO=dict(k)),source_NV=dict(OLD_PCD=nv,PROTO=dict(nv)))
    paths=[nt]
    for condition in ('clean_DURA','star_DURA'):
        rows['Q_'+condition]=dict(OLD_PCD=dict(old),PROTO=dict(q));p=root/(condition+'_truth.npz');paths.append(p)
        np.savez(p,ids=ids,y=qy,tx=np.asarray(s.fresh.TX_BY_LABEL)[qy],receiver=np.full(len(qy),34,np.int64),day=np.asarray(days),scenario=np.asarray(['clean']*len(qy)) if condition=='clean_DURA' else np.asarray(ss))
    done=dict(run_id=s.RUN_ID,code_commit='a'*40,method_names=list(s.METHODS),primary_method=s.PRIMARY)
    cal=dict(methods={name:dict(constraints_passed=True) for name in s.METHODS})
    return s,done,cal,rows,paths


def test_all_four_readouts_before_three_truths_positive_and_negative(tmp_path,monkeypatch,subtests):
    s,done,cal,rows,paths=fixture(tmp_path);monkeypatch.setattr(s,'_prediction',lambda root:(done,cal,rows))
    reads=[];sr=s._source_truth;tr=s._future_truth
    monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),sr(*a))[1]);monkeypatch.setattr(s,'_future_truth',lambda *a:(reads.append('future'),tr(*a))[1])
    result=s.score(tmp_path/'predict',*paths,tmp_path/'score')
    assert reads==['source','future','future'] and result['truth_last'] and result['all_three_predictions_before_truth']
    assert result['development_gate_passed'] and result['primary_composite_success'] and result['adaptation_success'] is None
    assert result['A_N'] is result['H_OW'] is None and result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete']
    assert set(s.common.UNKNOWN_TX).issubset(result['methods'][s.PRIMARY]['paths']['source_DURA']['by_TX'])
    assert set(s.fresh.TX_BY_LABEL[6:]).issubset(result['methods'][s.PRIMARY]['paths']['star_DURA']['by_TX'])
    counts=result['paired_primary_vs_OLD_PCD_predicted_ratio']['star_DURA']['counts']
    assert counts['known_closed_correct_net']==1 and counts['known_correct_accepted_net']==1
    assert result['methods'][s.PRIMARY]['paths']['star_DURA']['by_day']['2021_03_23']['A_K'] is None
    # One entire unknown class accepted violates absolute RU and the fixed control comparison.
    a=rows['Q_star_DURA']['PROTO'];qtruth=s._future_truth(paths[2],a['ids']);a['scores']=a['scores'].copy();a['scores'][qtruth['y']==9]=1.
    failed=s.score(tmp_path/'predict',*paths,tmp_path/'failed')
    assert not failed['development_gate_passed'] and failed['next_step']=='NO_PROMOTION'
    for defect in ('partial_Q','prototype_route','source_geometry_ancestry','changed_control'):
        with subtests.test(defect=defect):
            reads.clear();monkeypatch.setattr(s,'_prediction',lambda root:(_ for _ in ()).throw(ValueError(defect)))
            with pytest.raises(ValueError):s.score(tmp_path/'predict',*paths,tmp_path/defect)
            assert reads==[]
    with pytest.raises(FileExistsError):s.score(tmp_path/'predict',*paths,tmp_path/'score')


def test_actual_producer_gate_own_identity_before_three_truths_and_terminal(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_prototype_identity_run import fixture as producer
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with producer(tmp_path,monkeypatch) as (p,c,*_):
        done=p.run(c,'cpu');out=Path(c['output_root']);_,_,_,_,_,rows=p._prediction_gate(out)
        proxy=type('Rows',(),dict(_rows=staticmethod(lambda root,role:rows['source_NV']['OLD_PCD'])))
        source=truth_files(tmp_path,type('Score',(),dict(common=s.common,CONDITIONS=())),proxy,out)[0];paths=[source]
        y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);days=np.asarray(['2021_03_15']*10+['2021_03_23']*8)
        for condition in ('clean_DURA','star_DURA'):
            a=rows['Q_'+condition]['OLD_PCD'];path=tmp_path/(condition+'_truth.npz');paths.append(path)
            np.savez(path,ids=a['ids'],y=y,tx=np.asarray(s.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),day=days,scenario=np.asarray(['clean' if condition=='clean_DURA' else 'practical_low_urban']*len(y)))
        reads=[];sr=s._source_truth;tr=s._future_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),sr(*a))[1]);monkeypatch.setattr(s,'_future_truth',lambda *a:(reads.append('future'),tr(*a))[1])
        monkeypatch.setattr(p,'run',lambda *a:pytest.fail('Independent scorer cannot run producer'))
        monkeypatch.setattr(p,'_backend',lambda *a:pytest.fail('Independent scorer cannot load neural weights'))
        result=s.score(out,*paths,tmp_path/'actual_score')
        assert reads==['source','future','future'] and result['status']==s.STATUS and result['adaptation_success'] is None
        assert result['prediction_costs']['real_signal_forward_rows']==done['real_signal_forward_rows']==0
        from experiments.pamf_phase3_open_world import prototype_identity_launch as launch
        with monkeypatch.context() as handoff:
            handoff.setattr(launch,'_read',lambda *a:s._read(tmp_path/'actual_score/complete.json'))
            assert launch._terminal(dict(run_root=str(tmp_path/'terminal'),code_commit=c['code_commit']),'score')['status']==s.STATUS
        path=out/'targets/star_DURA/PROTO/predictions.npz';saved=path.read_bytes()
        summary=Path(c['source_geometry_root'])/'summary.json';summary_saved=summary.read_bytes()
        old_path=out/'targets/star_DURA/OLD_PCD/predictions.npz';old_saved=old_path.read_bytes()
        for defect in ('partial_Q','prototype_closed','wrong_density_route','changed_control','wrong_geometry_ancestor'):
            with subtests.test(defect=defect):
                reads.clear();path.write_bytes(saved)
                if defect=='partial_Q':path.unlink()
                elif defect=='wrong_geometry_ancestor':
                    value=s._read(summary);value['checkpoint_identities']['PCD']['updates']=445;summary.unlink();s._write(summary,value)
                else:
                    changed=old_path if defect=='changed_control' else path
                    with np.load(changed,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='prototype_closed':a['closed'][0]=(a['closed'][0]+1)%6
                    elif defect=='wrong_density_route':a['raw_scores'][0,1]+=.1
                    else:a['scores'][0,1]+=.1
                    np.savez(changed,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/defect)
                assert reads==[]
                summary.write_bytes(summary_saved);old_path.write_bytes(old_saved)
        path.write_bytes(saved)
