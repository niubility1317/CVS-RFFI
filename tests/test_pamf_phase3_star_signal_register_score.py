"""Truth-last registration metrics and one synthetic producer/CPU-score flow."""
from pathlib import Path
import importlib
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_signal_register_score')


def test_U_only_match_missing_arrival_and_unmatched_acceptance():
    s=api();uy=np.asarray([6,6,7,7,0],np.int64);assigned=np.asarray([0,0,0,1,2],np.int64)
    mapping,diag=s.prior._matching(uy,assigned,assigned>=0,3,[6,7])
    assert mapping=={0:6,1:7} and diag['unmatched_cluster_indices']==[2]
    y=np.arange(9,dtype=np.int64);raw=np.asarray([0,1,2,3,4,5,6,-1,8],np.int64)
    mapped=s.prior._mapped(raw,mapping)
    row=s._metrics(y,raw,mapped,y,np.ones(9),y,True,[6,7],[8],[8])
    assert row['A_K']==row['C_K']==1 and row['A_N']==.5
    assert row['R_U']==0 and row['H_score']==0 and row['unmatched_anonymous_accepted']==1
    assert row['registered_macro_acceptance']==7/8 and row['registered_macro_accuracy']==7/8
    assert row['H_KN']==pytest.approx(2/3) and row['H_OW']==0
    keep=y!=6
    missing=s._metrics(y[keep],raw[keep],mapped[keep],y[keep],np.ones(8),y[keep],True,[6,7],[8],[8])
    assert missing['A_N'] is missing['registered_macro_acceptance'] is missing['H_KN'] is missing['H_OW'] is None
    assert missing['A_K']==1 and missing['R_U']==0 and missing['H_score']==0
    assert not missing['coverage']['complete']
    before=s._metrics(y[keep],raw[keep],raw[keep],y[keep],np.ones(8),y[keep],False,[6,7],[8],[6,7,8])
    assert before['R_U'] is before['H_score'] is before['A_N'] is None


def test_zero_registration_K_does_not_fake_new_accuracy_or_protection():
    s=api();y=np.arange(9,dtype=np.int64);raw=np.asarray([0,1,2,3,4,5,-1,-1,-1],np.int64)
    row=s._metrics(y,raw,raw,y,np.ones(9),y,True,[6,7],[8],[6,7,8])
    assert row['A_N']==0 and row['registered_macro_acceptance']==.75
    assert row['A_K']==row['C_K']==row['R_U']==row['H_score']==1
    assert row['H_KN']==row['H_OW']==0
    row['old_loss_from_REG0']=0
    checks=s._checks(row,.9,overall=True)
    assert not checks['arrival_accuracy80'] and not checks['registered_CK90']


def test_pair_counts_keep_later_denominator_and_accepted_wrong():
    s=api();y=np.asarray([0,0,6,7,8,8]);left=np.asarray([0,6,6,-1,-1,8]);right=np.asarray([-1,0,-1,-1,8,-1])
    lm=s.prior._mapped(left,{0:6});rm=right.copy()
    row=s._counts(y,left,right,lm,rm,[6,7],[8])
    assert row['old_correct_accept_gain']==row['old_correct_accept_loss']==1
    assert row['old_correct_accept_net']==0 and row['old_accept_net']==1
    assert row['arrival_correct_accept_net']==1
    assert row['later_reject_gain']==row['later_reject_loss']==1 and row['later_reject_net']==0


def test_positive_anonymous_full_U_moments_and_joint_reconstruction(monkeypatch):
    from test_pamf_phase3_star_class_density_core import state
    s=api();d=state();N=32;monkeypatch.setattr(s,'U_COUNT',N)
    z=np.repeat(np.eye(160,dtype=np.float32)[:2],16,axis=0);closed=np.zeros(N,np.int64);tau=np.full((2,6),1e6,np.float64)
    raw,base=s.evaluation._base(d,z,closed,tau);labels=np.repeat(np.arange(2,dtype=np.int64),16)
    U=dict(ids=np.asarray(['u'+str(i) for i in range(N)]),default_logits=np.zeros((N,6),np.float32),closed=closed,signal_features=z,
        base_scores=base,assignments=labels.copy(),dbscan_labels=labels.copy(),rejected=np.ones(N,bool))
    white=(s.evaluation.parent.core.density.unit(z)-d['origin'])@d['whitener']
    means=np.asarray([white[labels==j].mean(0) for j in range(2)],np.float64)
    anon=dict(schema=np.asarray(s.core.ANONYMOUS_SCHEMA),means=means,variances=np.full(2,160/176,np.float64),anonymous_ids=np.asarray(['anon-0001','anon-0002']))
    source=dict(schema=np.asarray(s.core.SOURCE_SCHEMA),eps=np.asarray(.1,np.float64))
    s._U_state(U,d,source,anon,tau)
    ratios=s.core.anonymous_ratios(d,anon,z);registration_tau=np.asarray([ratios.min()-1]*2,np.float64)
    predicted,decision=s.core.predict(closed,base,ratios,registration_tau)
    a=dict(ids=U['ids'],default_logits=U['default_logits'],closed=closed,signal_features=z,raw_scores=raw,scores=base,
        predictions=predicted,decision_scores=decision,anonymous_raw_scores=ratios)
    s._reconstruct(a,d,anon,tau,registration_tau)
    assert np.all(a['predictions'][:,2:]>=6)
    anon['means'][0,0]+=.1
    with pytest.raises(ValueError,match='all assigned U'):s._U_state(U,d,source,anon,tau)
    anon['means'][0,0]-=.1;U['assignments'][0]=-1
    with pytest.raises(ValueError,match='complete DBSCAN'):s._U_state(U,d,source,anon,tau)


def test_fixed_primary_cannot_borrow_pooled_or_hide_scene_missing_arrival(tmp_path,monkeypatch):
    s=api();kg=[(c,r,sc) for c in range(6) for r in s.common.RECEIVERS for sc in s.common.SCENES]
    ng=[(c,r,sc) for c in range(6,9) for r in s.common.RECEIVERS for sc in s.common.SCENES]
    def predictions(y,prefix):
        n=len(y);old=y<6;closed=np.where(old,y,0).astype(np.int64);p=np.tile(np.where(old,closed,-1)[:,None],(1,4))
        for j in (2,3):p[np.isin(y,(6,7)),j]=y[np.isin(y,(6,7))]
        return dict(ids=np.asarray([prefix+str(i) for i in range(n)]),closed=closed,baseline_closed=closed.copy(),predictions=p,
            decision_scores=np.tile(np.where(old,1.,-1.)[:,None],(1,4)))
    k=predictions(np.asarray([g[0] for g in kg]),'k');k.update(y=np.asarray([g[0] for g in kg]),rx_i=np.asarray([g[1] for g in kg]),
        day_i=np.arange(len(kg))%3+1,scenario=np.asarray([g[2] for g in kg]))
    n=predictions(np.asarray([g[0] for g in ng]),'n');n.update(rx_i=np.asarray([g[1] for g in ng]),day_i=np.arange(len(ng))%3+1)
    nt=dict(tx=np.asarray([s.common.UNKNOWN_TX[g[0]-6] for g in ng]),scenario=np.asarray([g[2] for g in ng]))
    y=np.repeat(np.arange(9),6);clean=predictions(y,'c');star=predictions(y,'s')
    ct=dict(y=y.copy(),receiver=np.full(len(y),26),scenario=np.asarray(['clean']*len(y)))
    st=dict(y=y.copy(),receiver=np.full(len(y),26),scenario=np.tile(np.asarray(s.common.SCENES),9))
    U=dict(ids=np.asarray(['u0','u1','u2','u3']),assignments=np.asarray([0,0,1,1],np.int64))
    ut=dict(y=np.asarray([6,6,7,7]),receiver=np.full(4,26),scenario=np.asarray([s.common.SCENES[0]]*4))
    m=dict(code_commit='a'*40,target_U_reads=4,**{key:0 for key in ('real_signal_forward_rows','real_F_forward_rows','G_forward_rows','synthetic_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows')})
    cal=dict(report=dict(methods={name:dict(constraints_passed=True) for name in s.METHODS[2:]}));reads=[]
    monkeypatch.setattr(s,'_prediction',lambda *a:(m,cal,dict(known=k,novel=n,clean_DURA=clean,star_DURA=star),U,dict(means=np.zeros((2,160)))))
    monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source') or nt)
    monkeypatch.setattr(s,'_truth',lambda path,*a:reads.append(path) or {'U':ut,'clean':ct,'star':st}[path])
    result=s.score('prediction','U','source','clean','star',tmp_path/'pass')
    assert reads==['U','source','clean','star'] and result['registration_success']
    assert result['U_true_arrival_labels']==[6,7] and result['planned_later_labels']==[8]
    pair=result['paired']['predicted_REG1_minus_REG0']['conditions']['star_DURA']
    assert pair['delta']['R_U'] is pair['delta']['H_score'] is None and pair['delta']['planned_later_macro_rejection']==0
    assert all(len(method['source']['groups'])==42 for method in result['methods'].values())
    star['predictions'][np.isin(y,(6,7)),3]=-1
    failed=s.score('prediction','U','source','clean','star',tmp_path/'main_fail')
    assert failed['methods']['pooled_REG1']['star_DURA']['goals']['passed']
    assert failed['methods'][s.PRIMARY]['star_DURA']['overall']['A_N']==0 and not failed['registration_success']
    assert failed['next_step']=='NO_PROMOTION' and failed['full_goal_status']=='ACTIVE' and not failed['final_phase3_complete']
    star['predictions'][np.isin(y,(6,7)),3]=y[np.isin(y,(6,7))]
    st['scenario'][(y==6)&(st['scenario']==s.common.SCENES[0])]=s.common.SCENES[1]
    missing=s.score('prediction','U','source','clean','star',tmp_path/'missing')
    high=missing['methods'][s.PRIMARY]['star_DURA']['by_scene'][s.common.SCENES[0]]
    assert high['A_N'] is high['registered_macro_acceptance'] is high['H_KN'] is high['H_OW'] is None
    assert high['A_K']==high['R_U']==high['H_score']==1 and not missing['development_gate_passed']


def test_actual_producer_CPU_four_truths_and_corruptions(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_signal_register_run import fixture
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,backend,U_public):
        m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','L','radius','U','discover','known','novel','clean_DURA','star_DURA']
        monkeypatch.setattr(s,'COUNTS',counts);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT);monkeypatch.setattr(s,'U_COUNT',m.U_COUNT)
        U=s._load(out/'registration/U.npz');nv=s._load(out/'novel_predictions.npz');grid=[(tx,r,sc) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for sc in s.common.SCENES]
        paths=[tmp_path/'U_truth.npz',tmp_path/'source_truth.npz',tmp_path/'clean_truth.npz',tmp_path/'star_truth.npz']
        def write_truth(path,values):np.savez(path,**{key:value[::-1] for key,value in values.items()})
        write_truth(paths[0],dict(ids=U['ids'],y=np.arange(m.U_COUNT,dtype=np.int64)%8,receiver=np.full(m.U_COUNT,26,np.int64),scenario=np.asarray([s.common.SCENES[0]]*m.U_COUNT)))
        write_truth(paths[1],dict(ids=nv['ids'],rx_i=nv['rx_i'],day_i=nv['day_i'],tx=np.asarray([x[0] for x in grid]),scenario=np.asarray([x[2] for x in grid])))
        for name,path in zip(s.CONDITIONS[1:],paths[2:]):
            a=s._load(out/'targets'/name/'predictions.npz');N=len(a['ids'])
            write_truth(path,dict(ids=a['ids'],y=np.arange(N,dtype=np.int64)%9,receiver=np.full(N,26,np.int64),scenario=np.asarray(['clean' if name=='clean_DURA' else s.common.SCENES[0]]*N)))
        reads=[];truth=s._truth;source=s._source_truth
        def read_truth(path,*a):reads.append(Path(path).name);return truth(path,*a)
        def source_truth(path,*a):reads.append(Path(path).name);return source(path,*a)
        monkeypatch.setattr(s,'_truth',read_truth);monkeypatch.setattr(s,'_source_truth',source_truth)
        monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('CPU scorer cannot load neural weights'))
        for name in ('source_radius','discover','calibrate'):
            monkeypatch.setattr(m.core,name,lambda *a:pytest.fail('CPU scorer cannot fit, cluster or recalibrate'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==[p.name for p in paths] and result['truth_last'] and result['all_three_predictions_before_truth']
        assert result['U_mapping_frozen_before_source_and_Q_truth'] and not result['Q_truth_used_for_cluster_mapping']
        assert result['target_REG'] and not result['target_DA'] and result['next_step']=='NO_PROMOTION'
        assert all(len(method['source']['groups'])==42 for method in result['methods'].values())
        with pytest.raises(FileExistsError):s.score(out,*paths,tmp_path/'score')
        files=[out/'complete.json',out/'models/density.npz',out/'models/anonymous.npz',out/'thresholds.npz',out/'registration/U.npz',
            out/'novel_predictions.npz',out/'targets/clean_DURA/predictions.npz',out/'targets/star_DURA/predictions.npz']
        saved={p:p.read_bytes() for p in files}
        for defect in ('missing_source','missing_clean','missing_star','partial_freeze','identity','density','old_tau','new_tau','anon','U_truth_field','old_control','default_head','joint_prediction','private_Q'):
            with subtests.test(defect=defect):
                for p,value in saved.items():p.write_bytes(value)
                reads.clear()
                if defect.startswith('missing_'):
                    p=out/'novel_predictions.npz' if defect=='missing_source' else out/'targets'/('clean_DURA' if defect=='missing_clean' else 'star_DURA')/'predictions.npz';p.unlink()
                elif defect in ('partial_freeze','identity'):
                    p=out/'complete.json';a=m._read(p)
                    if defect=='identity':a['model_identity']['updates']+=1
                    else:a['anonymous_frozen_before_V_and_Q']=False
                    p.unlink();m._write(p,a)
                else:
                    p=out/'models/density.npz' if defect=='density' else out/'models/anonymous.npz' if defect=='anon' else out/'thresholds.npz' if defect in ('old_tau','new_tau') else out/'registration/U.npz' if defect=='U_truth_field' else out/'novel_predictions.npz' if defect=='old_control' else out/'targets/star_DURA/predictions.npz'
                    a=s._load(p)
                    if defect=='density':a['known_means'][0,0]+=.1
                    elif defect=='anon':a['schema']=np.asarray('wrong')
                    elif defect=='old_tau':a['conditional'][0,0]+=.1
                    elif defect=='new_tau':a['registration'][1]=np.nan
                    elif defect=='U_truth_field':a['y']=np.zeros(m.U_COUNT,np.int64)
                    elif defect=='old_control':a['scores'][0,0]+=.1
                    elif defect=='default_head':a['closed'][0]=(a['closed'][0]+1)%6
                    elif defect=='joint_prediction':a['predictions'][0,3]=6+result['cluster_diagnostics']['prototype_count']
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/'invalid')
                assert not reads and not (tmp_path/'invalid').exists()
