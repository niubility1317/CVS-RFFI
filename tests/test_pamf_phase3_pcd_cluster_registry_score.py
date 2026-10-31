"""U-only majority matching and future registry scoring keep cohort boundaries."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.pcd_cluster_registry_score')


def test_U_strict_majority_known_clusters_and_one_to_one(subtests):
    m=api();labels=np.asarray([2,6,7,8,9],np.int64)
    y=np.asarray([6,6,0,0,6,6,0,6,6,7,7,7,7,7,8],np.int64)
    assignment=np.asarray([0,0,1,1,1,1,2,2,2,3,3,3,4,4,-1],np.int64)
    mapping,diag=m._matching(y,assignment,labels)
    assert mapping=={1:6,2:7}  # anonymous0 is a 50% tie; anonymous3 loses the one-to-one match.
    assert diag['unmatched_true_arrivals']==[8] and diag['known_associated_cluster_count']==1
    assert diag['strict_majority_required'] and diag['Q_truth_used'] is False
    raw=np.asarray([0,6,7,8,9,-1],np.int64)
    np.testing.assert_array_equal(m._mapped(raw,mapping),[0,-2,6,7,-2,-1])
    for a in (np.asarray([],np.int64),np.asarray([0,1],np.int64)):
        with subtests.test(labels=a.tolist()):
            sy=np.asarray([0,6]);assign=np.full(2,-1,np.int64);result,_=m._matching(sy,assign,a);assert result=={}


def test_source_unknown_namespace_and_missed_arrival_denominator():
    m=api();old=np.arange(6,dtype=np.int64);base=np.r_[old,[0,0,0]].astype(np.int64);raw=np.r_[old,[6,7,8]].astype(np.int64)
    source_y=np.r_[old,[100,101,102]].astype(np.int64)
    source=m._metrics(source_y,raw,m._mapped(raw,{}),base,np.ones(9),True,{},source=True)
    assert source['R_U']==0 and source['A_N'] is source['H_KN'] is source['H_OW'] is None
    target_y=np.r_[old,[6,7,8]].astype(np.int64);mapping={0:6,1:7}
    target=m._metrics(target_y,raw,m._mapped(raw,mapping),base,np.ones(9),True,mapping)
    assert target['A_N']==pytest.approx(2/3) and target['new_macro_denominator']==3 and target['R_U'] is None
    assert target['new_per_class'][-1]['correct_accepted']==0 and target['unmatched_anonymous_accepted']==1


def test_missing_known_date_macro_NA_and_same_later_cohort():
    m=api();y=np.asarray([0,1,2,4,6,7,8,9],np.int64);raw=np.asarray([0,1,2,4,6,7,8,-1],np.int64);base=np.minimum(y,5)
    row=m._metrics(y,raw,raw.copy(),base,np.where(raw<0,-1.,1.),True,{0:6,1:7,2:8})
    assert row['A_K'] is row['C_K'] is row['H_score'] is row['registered_macro_acceptance'] is None
    assert row['R_U']==1. and row['planned_later_macro_rejection']==1.
    reg0=m._metrics(y,raw,np.where(raw>=6,-2,raw),base,np.where(raw<0,-1.,1.),False,{})
    assert reg0['R_U']==.25 and reg0['planned_later_macro_rejection']==1.


def fixture(tmp_path):
    m=api();nclass=10;scenes=m.common.SCENES;ys=[];ss=[];days=[]
    for d in ('2021_03_15','2021_03_23'):
        for scene in scenes:
            for c in range(nclass):
                if d=='2021_03_23' and c in (3,5):continue
                ys.append(c);ss.append(scene);days.append(d)
    y=np.asarray(ys,np.int64);n=len(y);ids=np.asarray(['future-'+str(i) for i in range(n)]);closed=np.minimum(y,5).astype(np.int64)
    pred=np.stack([np.where(y<6,y,-1),np.where(y<6,y,-1),np.where(y<9,y,-1),np.where(y<9,y,-1)],axis=1).astype(np.int64)
    # Registered third arrival was missed by U matching and must remain wrong on future Q.
    pred[y==8,2:]=8
    decisions=np.where(pred>=0,1.,-1.).astype(np.float64)
    q=dict(ids=ids,closed=closed,baseline_closed=closed.copy(),raw_prediction=pred,decision_score=decisions)
    source_y=np.tile(np.arange(6,dtype=np.int64),30);rx=np.repeat(np.asarray(m.common.RECEIVERS,np.int64),36);scene=np.tile(np.repeat(np.asarray(scenes),6),5)
    k=dict(ids=np.asarray(['known-'+str(i) for i in range(len(source_y))]),y=source_y,rx_i=rx,day_i=np.tile(np.repeat([1,2,3],12),5).astype(np.int64),scenario=scene,
        closed=source_y.copy(),baseline_closed=source_y.copy(),raw_prediction=np.repeat(source_y[:,None],4,axis=1),decision_score=np.ones((len(source_y),4),np.float64))
    nvlabels=np.tile(np.arange(3),len(scenes)*5);nvids=np.asarray(['novel-'+str(i) for i in range(len(nvlabels))]);nv=dict(ids=nvids,closed=np.zeros(len(nvlabels),np.int64),baseline_closed=np.zeros(len(nvlabels),np.int64),
        rx_i=np.repeat(np.asarray(m.common.RECEIVERS,np.int64),len(scenes)*3),day_i=np.tile(np.repeat([1,2,3],6),5).astype(np.int64),raw_prediction=np.full((len(nvlabels),4),-1,np.int64),decision_score=-np.ones((len(nvlabels),4)))
    u_y=np.repeat(np.arange(9,dtype=np.int64),2);uids=np.asarray(['U-'+str(i) for i in range(len(u_y))]);assignment=np.where(u_y<6,0,u_y-5).astype(np.int64)
    # cluster0 known-associated; anon0->6 and anon1->7, no cluster for true8.
    assignment[u_y==8]=-1
    U=dict(ids=uids,assignment=assignment);registry=dict(labels=np.asarray([0,6,7],np.int64),anonymous_ids=np.asarray(['anon-0001','anon-0002']))
    root=tmp_path/'predict';root.mkdir();config=dict(output_root=str(root));cfg=tmp_path/'predict.json';cfg.write_text(json.dumps(config))
    done=dict(run_id=m.RUN_ID,code_commit='a'*40,method_names=list(m.METHODS),primary_method=m.PRIMARY,source_only=False,final_phase3_complete=False)
    rows=dict(source_V=k,source_NV=nv,Q_clean_DURA=dict(q),Q_star_DURA=dict(q))
    ut=tmp_path/'U_truth.npz';np.savez(ut,ids=uids,y=u_y,tx=np.asarray(m.fresh.TX_BY_LABEL)[u_y],receiver=np.full(len(u_y),34,np.int64),day=np.asarray(['2021_03_01']*len(u_y)),scenario=np.asarray([scenes[0]]*len(u_y)))
    nt=tmp_path/'source_truth.npz';np.savez(nt,ids=nvids,tx=np.asarray(m.common.UNKNOWN_TX)[nvlabels],rx_i=nv['rx_i'],day_i=nv['day_i'],scenario=np.tile(np.repeat(np.asarray(scenes),3),5))
    paths=[ut,nt]
    for condition in ('clean','star'):
        path=tmp_path/(condition+'_truth.npz');paths.append(path)
        np.savez(path,ids=ids,y=y,tx=np.asarray(m.fresh.TX_BY_LABEL)[y],receiver=np.full(n,34,np.int64),day=np.asarray(days),scenario=np.asarray(['clean']*n) if condition=='clean' else np.asarray(ss))
    return m,cfg,done,U,registry,rows,paths


def test_complete_all_predictions_before_four_truths_and_full_gates(tmp_path,monkeypatch,subtests):
    m,cfg,done,U,registry,rows,paths=fixture(tmp_path)
    reads=[];gate=lambda c:(done,{},registry,U,rows);monkeypatch.setattr(m,'_prediction_gate',gate)
    original=m._arrays
    def read(path):reads.append(Path(path).name);return original(path)
    monkeypatch.setattr(m,'_arrays',read)
    result=m.score(cfg,*paths,tmp_path/'score')
    assert reads==['U_truth.npz','source_truth.npz','clean_truth.npz','star_truth.npz']
    assert result['truth_last'] and result['all_three_predictions_before_truth'] and result['full_goal_status']=='ACTIVE'
    assert not result['development_gate_passed'] and result['next_step']=='NO_PROMOTION'  # missed 8 yields AN2/3.
    main=result['methods'][m.PRIMARY]
    assert main['source']['overall']['A_N'] is None and main['star_DURA']['overall']['new_macro_denominator']==3
    assert main['star_DURA']['overall']['R_U_labels']==[9]
    assert main['star_DURA']['by_day']['2021_03_23']['A_K'] is None
    # The producer gate must fail before any private IO on partial/corrupt artifact.
    for defect in ('partial_Q','wrong_source_namespace','misaligned_U'):
        with subtests.test(defect=defect):
            reads.clear()
            monkeypatch.setattr(m,'_prediction_gate',lambda c:(_ for _ in ()).throw(ValueError(defect)))
            with pytest.raises(ValueError):m.score(cfg,*paths,tmp_path/defect)
            assert reads==[]
    with pytest.raises(FileExistsError):m.score(cfg,*paths,tmp_path/'score')


def test_all_three_arrivals_pass_then_one_later_false_accept_fails(tmp_path,monkeypatch):
    m,cfg,done,U,registry,rows,paths=fixture(tmp_path)
    registry['labels']=np.asarray([0,6,7,8],np.int64);registry['anonymous_ids']=np.asarray(['anon-0001','anon-0002','anon-0003'])
    U['assignment'][-2:]=3
    monkeypatch.setattr(m,'_prediction_gate',lambda c:(done,{},registry,U,rows))
    result=m.score(cfg,*paths,tmp_path/'positive')
    assert result['development_gate_passed'] and result['main_registration_gates_passed'] and result['predicted_REG0_gates_passed']
    assert result['primary_summary']['star_DURA']['A_N']==1. and result['primary_summary']['star_DURA']['R_U']==1.
    # Positive science is still development evidence, never full Phase3 completion.
    assert result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete']
    from experiments.pamf_phase3_open_world import pcd_cluster_registry_launch as launch
    read=launch._read
    with monkeypatch.context() as handoff:
        handoff.setattr(launch,'_read',lambda *a:m._read(tmp_path/'positive/complete.json'))
        assert launch._terminal(dict(run_root=str(tmp_path/'terminal'),code_commit=done['code_commit']),'score')['development_gate_passed']
    star=rows['Q_star_DURA'];later=np.flatnonzero(star['raw_prediction'][:,3]==-1)[0]
    star['raw_prediction'][later,3]=6;star['decision_score'][later,3]=1.
    failed=m.score(cfg,*paths,tmp_path/'one_later_false_accept')
    assert not failed['development_gate_passed'] and failed['next_step']=='NO_PROMOTION'
    assert failed['paired']['predicted_REG1_vs_predicted_REG0']['star_DURA']['same_later_cohort']['gain']<0


def test_actual_producer_gate_to_four_truths_and_partial_evidence_rejection(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_pcd_cluster_registry_run import fixture as producer
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with producer(tmp_path,monkeypatch) as (p,c,*_):
        done=p.run(c,'cpu');out=Path(c['output_root']);cfg=tmp_path/'run.json';cfg.write_text(json.dumps(c))
        _,_,registry,U,rows=p._prediction_gate(out)
        uy=np.repeat(np.arange(9,dtype=np.int64),64);up=tmp_path/'U_truth.npz'
        np.savez(up,ids=U['ids'],y=uy,tx=np.asarray(s.fresh.TX_BY_LABEL)[uy],receiver=np.full(len(uy),34,np.int64),day=np.asarray(['2021_03_01']*len(uy)),scenario=np.asarray([s.common.SCENES[0]]*len(uy)))
        proxy=type('Rows',(),dict(_rows=staticmethod(lambda root,role:rows['source_NV'])))
        nt=truth_files(tmp_path,type('Scoring',(),dict(common=s.common,CONDITIONS=())),proxy,out)[0]
        paths=[up,nt];y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);day=np.asarray(['2021_03_15']*10+['2021_03_23']*8)
        for condition in ('clean_DURA','star_DURA'):
            a=rows['Q_'+condition];path=tmp_path/(condition+'_truth.npz');paths.append(path)
            np.savez(path,ids=a['ids'],y=y,tx=np.asarray(s.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),day=day,scenario=np.asarray(['clean' if condition=='clean_DURA' else 'practical_low_urban']*len(y)))
        reads=[];read=s._arrays
        monkeypatch.setattr(s,'_arrays',lambda path:(reads.append(Path(path).name),read(path))[1])
        monkeypatch.setattr(p,'run',lambda *a:pytest.fail('Scorer cannot run producer'))
        monkeypatch.setattr(p.core,'discover',lambda *a:pytest.fail('Scorer cannot recluster U'))
        result=s.score(cfg,*paths,tmp_path/'actual_score')
        assert result['status']=='PCD_CLUSTER_REGISTRY_SCORES_COMPLETE' and len(reads)==4 and result['truth_last']
        assert result['prediction_costs']['real_signal_forward_rows']==done['real_signal_forward_rows']
        from experiments.pamf_phase3_open_world import pcd_cluster_registry_launch as launch
        with monkeypatch.context() as handoff:
            handoff.setattr(launch,'_read',lambda *a:s._read(tmp_path/'actual_score/complete.json'))
            assert launch._terminal(dict(run_root=str(tmp_path/'terminal'),code_commit=c['code_commit']),'score')['status']==s.STATUS
        path=out/'predictions/Q_star_DURA.npz';saved=path.read_bytes()
        for defect in ('partial','changed_decision'):
            with subtests.test(defect=defect):
                reads.clear();path.write_bytes(saved)
                if defect=='partial':path.unlink()
                else:
                    a=read(path);a['raw_prediction'][0,3]=0 if a['raw_prediction'][0,3]<0 else -1;np.savez(path,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(cfg,*paths,tmp_path/defect)
                assert reads==[]
        path.write_bytes(saved)
