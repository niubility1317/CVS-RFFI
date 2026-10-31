"""Moment adaptation scoring uses complete synthetic public predictions only."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.pcd_moment_score')


def future(condition='clean'):
    s=api();y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);n=len(y)
    a=dict(ids=np.asarray(['q'+str(i) for i in range(n)]),closed=np.minimum(y,5),baseline_closed=np.minimum(y,5),
        scores=np.where(y[:,None]<6,1.,-1.)*np.ones((n,8)))
    t=dict(ids=a['ids'],y=y,tx=np.asarray(s.fresh.TX_BY_LABEL)[y],receiver=np.full(n,34,np.int64),
        day=np.asarray(['2021_03_15']*10+['2021_03_23']*8),scenario=np.asarray(['clean' if condition=='clean' else 'practical_low_urban']*n))
    return a,t


def test_full_unknown_roster_and_unchanged_identity_pairs():
    s=api();a,t=future();row=s.fresh._target(a,t,'clean_DURA',7)
    assert row['overall']['A_K']==row['overall']['R_U']==1. and row['unknown_roster']==[6,7,8,9]
    assert row['by_day']['2021_03_23']['A_K'] is None
    a['scores'][t['y']==9,7]=1.
    assert s.fresh._target(a,t,'clean_DURA',7)['overall']['R_U']==.75
    y=np.asarray([0,1,2],np.int64);ids=np.asarray(['k0','k1','k2'])
    known=dict(ids=ids,y=y,closed=y,scores=np.ones((3,8)));novel=dict(ids=np.asarray(['u0','u1']),scores=-np.ones((2,8)))
    known['scores'][0,3]=-1.;known['scores'][1,7]=-1.;novel['scores'][0,3]=1.
    r=s._paired(known,novel,7,3)
    assert r['known_closed_prediction_changed']==r['known_closed_correct_net']==0
    assert r['known_correct_accepted_gained']==r['known_correct_accepted_lost']==1
    assert r['unknown_rejected_net']==1


def synthetic_rows():
    s=api();kg=[(c,r,sc) for c in range(6) for r in s.common.RECEIVERS for sc in s.common.SCENES]
    ng=[(tx,r,sc) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for sc in s.common.SCENES]
    k=dict(ids=np.asarray(['k'+str(i) for i in range(len(kg))]),y=np.asarray([r[0] for r in kg],np.int64),
        rx_i=np.asarray([r[1] for r in kg],np.int64),day_i=np.arange(len(kg),dtype=np.int64)%3+1,
        scenario=np.asarray([r[2] for r in kg]),scores=np.ones((len(kg),8)))
    k['closed']=k['y'].copy();k['baseline_closed']=k['closed'].copy()
    n=dict(ids=np.asarray(['n'+str(i) for i in range(len(ng))]),rx_i=np.asarray([r[1] for r in ng],np.int64),
        day_i=np.arange(len(ng),dtype=np.int64)%3+1,scores=-np.ones((len(ng),8)))
    nt=dict(ids=n['ids'],tx=np.asarray([r[0] for r in ng]),rx_i=n['rx_i'],day_i=n['day_i'],scenario=np.asarray([r[2] for r in ng]))
    rows={'source_DURA':dict(known=k,novel=n)};truth={}
    for base in ('clean','star'):rows[base+'_DURA'],truth[base]=future(base)
    done=dict(run_id=s.RUN_ID,code_commit='a'*40,target_U_reads=576,U_ID_metadata_reads=576,real_F_forward_rows=0,
        real_signal_forward_rows=0,G_forward_rows=0,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=4,G_synthetic_forward_rows=2,
        adaptation_status='FITTED')
    cal=dict(methods={name:dict(constraints_passed=True) for name in s.METHODS})
    return done,cal,rows,nt,truth


def test_three_truth_reads_and_scientific_failure_is_not_fit_success(tmp_path,monkeypatch):
    s=api();done,cal,rows,nt,truth=synthetic_rows();reads=[]
    monkeypatch.setattr(s,'_prediction',lambda *a:(done,cal,rows))
    monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),nt)[1])
    monkeypatch.setattr(s,'_future_truth',lambda path,*a:(reads.append(str(path)),truth[str(path)])[1])
    monkeypatch.setattr(s.fresh,'_prediction',lambda *a:pytest.fail('No old U-value/ID scorer gate'))
    result=s.score('public','NV','clean','star',tmp_path/'score')
    assert reads==['source','clean','star'] and result['all_three_predictions_before_truth']
    assert len(result['methods'])==8 and all(len(v['paths'])==3 for v in result['methods'].values())
    assert set(result['paired_primary_vs_controls'])=={'E0_predicted_ratio','PCD_predicted_ratio','SHIFT_predicted_ratio'}
    assert result['adaptation_success'] is result['scientific_success'] is False
    assert result['new_target_DA'] and result['next_step']=='NO_PROMOTION'
    assert result['scorer_U_ID_metadata_reads']==576 and result['scorer_U_feature_reads']==0
    assert result['primary_deployed_neural_parameters']==834665 and result['primary_adapter_numeric_bytes']==212480
    assert result['source_known_42_passed'] and result['A_N'] is result['H_OW'] is None
    assert not result['final_phase3_complete'] and result['full_goal_status']=='ACTIVE'
    with pytest.raises(FileExistsError):s.score('public','NV','clean','star',tmp_path/'score')


def test_actual_producer_gate_three_truths_and_tampering_is_rejected_before_truth(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_pcd_moment_run import fixture
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,identity,events,header):
        done=m.run(c,'cpu');out=Path(c['output_root']);s=api()
        monkeypatch.setattr(s,'COUNTS',dict(m.COUNTS));monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        monkeypatch.setattr(m.core,'fit_adapters',lambda *a,**kw:pytest.fail('Scorer must never re-estimate U statistics'))
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Independent CPU scoring cannot load neural checkpoints'))
        original=np.load;loads=[]
        def load(path,*a,**kw):
            p=str(path).replace('\\','/');loads.append(p)
            assert not p.endswith('/discovery/U.npz') and not p.endswith('/U_features.npz')
            return original(path,*a,**kw)
        monkeypatch.setattr(np,'load',load)
        _,_,rows=s._prediction(out)
        assert sum(p.endswith('/adapters/U_ids.npy') for p in loads)==1
        assert done['target_U_reads']==576 and done['new_neural_updates']==done['new_threshold_sets_fitted']==0
        n=rows['source_DURA']['novel'];grid=[(tx,r,sc) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for sc in s.common.SCENES]
        paths=[tmp_path/'source_truth.npz'];np.savez(paths[0],ids=n['ids'],rx_i=n['rx_i'],day_i=n['day_i'],tx=np.asarray([g[0] for g in grid]),scenario=np.asarray([g[2] for g in grid]))
        for base in ('clean','star'):
            _,t=future(base);t['ids']=rows[base+'_DURA']['ids'];p=tmp_path/(base+'_truth.npz');np.savez(p,**t);paths.append(p)
        reads=[];source=s._source_truth;target=s._future_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1])
        monkeypatch.setattr(s,'_future_truth',lambda *a:(reads.append('future'),target(*a))[1])
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','future','future'] and len(result['methods'])==8 and result['new_target_DA']
        assert result['methods'][s.PRIMARY]['paths']['clean_DURA']['by_day']['2021_03_23']['A_K'] is None
        assert result['adaptation_success']==result['scientific_success'] and result['scorer_U_feature_reads']==0
        q=out/'targets/star_DURA/predictions.npz';known=out/'known_predictions.npz';meta=out/'adapters/complete.json'
        files=(out/'complete.json',q,known,out/'thresholds.npz',out/'models/PCD.npz',out/'adapters/CORAL.npz',out/'adapters/U_ids.npy',meta)
        saved={p:p.read_bytes() for p in files}
        for defect in ('partial','early_completion','UQ_overlap','cache_identity','adapter_matrix','routing','source_adapter','old_control','tau','density','private_Q','NO_CORE_nonidentity'):
            with subtests.test(defect=defect):
                for p,data in saved.items():p.write_bytes(data)
                reads.clear()
                if defect=='partial':q.unlink()
                elif defect=='early_completion':
                    d=m._read(out/'complete.json');d['status']='PARTIAL';(out/'complete.json').write_text(json.dumps(d),encoding='utf-8')
                elif defect=='cache_identity':
                    d=m._read(meta);d['U_manifest']=dict(d['U_manifest'],capsule_id='wrong');meta.write_text(json.dumps(d),encoding='utf-8')
                elif defect=='UQ_overlap':
                    ids=original(out/'adapters/U_ids.npy',allow_pickle=False).tolist();ids[0]=str(rows['star_DURA']['ids'][0]);np.save(out/'adapters/U_ids.npy',np.asarray(ids))
                elif defect=='NO_CORE_nonidentity':
                    d=m._read(meta);d.update(fit_status='NO_CORE',adaptation_status='NO_CORE',fit_succeeded=False);d['diagnostics'].update(status='NO_CORE',fit_succeeded=False);meta.write_text(json.dumps(d),encoding='utf-8')
                    d=m._read(out/'complete.json');d.update(adaptation_status='NO_CORE',fit_succeeded=False);(out/'complete.json').write_text(json.dumps(d),encoding='utf-8')
                    p=out/'adapters/CORAL.npz'
                    with original(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    a['offset'][0,0]+=.01;np.savez(p,**a)
                else:
                    p=out/'thresholds.npz' if defect=='tau' else out/'models/PCD.npz' if defect=='density' else out/'adapters/CORAL.npz' if defect=='adapter_matrix' else known if defect in ('source_adapter','old_control') else q
                    with original(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='tau':a['conditional'][7,0]+=.1
                    elif defect=='density':a['origin'][0]+=.01
                    elif defect=='adapter_matrix':a['matrix'][0,0]+=.01
                    elif defect=='private_Q':a['y']=np.zeros(len(a['ids']),np.int64)
                    elif defect=='old_control':a['default_logits'][0]+=1
                    elif defect=='source_adapter':a['raw_scores'][0,7]+=.01;a['scores'][0,7]+=.01
                    else:a['scores'][0,7]+=.1
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/('bad_'+defect))
                assert reads==[] and not (tmp_path/('bad_'+defect)).exists()
        for p,data in saved.items():p.write_bytes(data)
