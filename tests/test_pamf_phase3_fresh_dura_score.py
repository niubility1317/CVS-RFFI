"""Four-unknown raw/DURA diagnostics use only synthetic completed artifacts."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.fresh_dura_score')


def future(condition='clean'):
    s=api();y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);n=len(y)
    a=dict(ids=np.asarray(['q'+str(i) for i in range(n)]),closed=np.minimum(y,5),baseline_closed=np.minimum(y,5),
           scores=np.where(y[:,None]<6,1.,-1.)*np.ones((n,4)))
    t=dict(ids=a['ids'],y=y,tx=np.asarray([s.fresh.TX_BY_LABEL[i] for i in y]),receiver=np.full(n,34,np.int64),
           day=np.asarray(['2021_03_15']*10+['2021_03_23']*8),scenario=np.asarray(['clean' if condition=='clean' else 'practical_low_urban']*n))
    return a,t


def test_raw_clean_mapping_full_four_unknowns_and_missing_date_knowns():
    s=api();a,t=future();row=s._target(a,t,'clean_raw',3)
    assert row['overall']['A_K']==row['overall']['R_U']==1. and row['goals']['unknown_minimum']==.95
    assert set(row['by_scene'])=={'clean'} and row['unknown_roster']==[6,7,8,9]
    late=row['by_day']['2021_03_23'];assert late['A_K'] is late['C_K'] is late['H_score'] is None and late['R_U']==1.
    a['scores'][t['y']==9,3]=1.
    assert s._target(a,t,'clean_raw',3)['overall']['R_U']==.75
    a,t=future('star');row=s._target(a,t,'star_raw',3)
    assert row['goals']['unknown_minimum']==.90 and not row['goals']['all_main_scenes_passed']
    with pytest.raises(ValueError):s._target(a,t,'source_raw',3)


def test_pairs_use_changed_actual_closed_predictions():
    s=api();ids=np.asarray(['a','b','c']);y=np.asarray([0,1,2],np.int64)
    old=dict(ids=ids,y=y,closed=np.asarray([1,1,2]),scores=np.asarray([[1.]*4,[1.]*4,[-1.]*4]))
    new=dict(ids=ids,y=y,closed=np.asarray([0,2,2]),scores=np.ones((3,4)))
    nn=dict(ids=np.asarray(['u0','u1']),scores=-np.ones((2,4)));on=dict(ids=nn['ids'],scores=np.ones((2,4)))
    r=s.best.paired_counts(new,old,nn,on,3)
    assert r['known_closed_prediction_changed']==2 and r['known_closed_correct_gained']==r['known_closed_correct_lost']==1
    assert r['known_correct_accepted_gained']==2 and r['known_correct_accepted_lost']==1
    assert r['known_wrong_accepted_gained']==r['known_wrong_accepted_lost']==1 and r['unknown_rejected_net']==2


def synthetic_rows():
    s=api();kg=[(c,rx,scene) for c in range(6) for rx in s.common.RECEIVERS for scene in s.common.SCENES]
    ng=[(tx,rx,scene) for tx in s.common.UNKNOWN_TX for rx in s.common.RECEIVERS for scene in s.common.SCENES]
    k=dict(ids=np.asarray(['k'+str(i) for i in range(len(kg))]),y=np.asarray([r[0] for r in kg],np.int64),
           rx_i=np.asarray([r[1] for r in kg],np.int64),day_i=np.arange(len(kg),dtype=np.int64)%3+1,
           scenario=np.asarray([r[2] for r in kg]),scores=np.ones((len(kg),4)))
    k['closed']=k['y'].copy();k['baseline_closed']=k['closed'].copy()
    n=dict(ids=np.asarray(['n'+str(i) for i in range(len(ng))]),rx_i=np.asarray([r[1] for r in ng],np.int64),
           day_i=np.arange(len(ng),dtype=np.int64)%3+1,scores=-np.ones((len(ng),4)))
    source=dict(ids=n['ids'],tx=np.asarray([r[0] for r in ng]),rx_i=n['rx_i'],day_i=n['day_i'],scenario=np.asarray([r[2] for r in ng]))
    rows={c:dict(known=k,novel=n) for c in ('source_raw','source_DURA')};truth={}
    for base in ('clean','star'):
        a,t=future(base);truth[base]=t
        for path in ('raw','DURA'):rows[base+'_'+path]=a
    done=dict(run_id=s.RUN_ID,code_commit='a'*40,real_F_forward_rows=1,real_signal_forward_rows=2,G_forward_rows=0,
        synthetic_F_forward_rows=2,synthetic_signal_forward_rows=4,G_synthetic_forward_rows=2)
    cal=dict(methods={name:dict(constraints_passed=True) for name in s.METHODS})
    return done,cal,rows,source,truth


def test_small_end_to_end_three_private_reads_and_fixed_no_promotion(tmp_path,monkeypatch):
    s=api();done,cal,rows,nt,truth=synthetic_rows();reads=[]
    monkeypatch.setattr(s,'_prediction',lambda *a:(done,cal,rows))
    monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),nt)[1])
    monkeypatch.setattr(s,'_future_truth',lambda path,*a:(reads.append(str(path)),truth[str(path)])[1])
    monkeypatch.setattr(s.fresh,'_prediction',lambda *a:pytest.fail('No fresh U-ID gate'))
    result=s.score('public','NV','clean','star',tmp_path/'score')
    assert reads==['source','clean','star'] and result['diagnostic_completed'] and result['all_six_predictions_before_truth']
    assert len(result['methods'])==4 and all(len(v['paths'])==6 for v in result['methods'].values())
    assert all(set(result['paired_DURA_minus_raw'][name])=={'source','clean','star'} for name in s.METHODS)
    assert result['primary_summary']['clean_DURA']['R_U']==1. and result['primary_summary']['clean_raw']['R_U']==1.
    assert result['adaptation_success'] is None and not result['new_target_DA'] and not result['eligible_primary']
    assert result['next_step']=='NO_PROMOTION' and not result['final_phase3_complete'] and result['full_goal_status']=='ACTIVE'
    with pytest.raises(FileExistsError):s.score('public','NV','clean','star',tmp_path/'score')


def test_wrong_source_and_target_private_namespaces_rejected(tmp_path,subtests):
    s=api();_,_,rows,nt,truth=synthetic_rows();source=tmp_path/'nv.npz';query=tmp_path/'q.npz'
    np.savez(source,**nt);np.savez(query,**truth['clean'])
    assert set(s._source_truth(source,rows['source_DURA']['novel'])['tx'])==set(s.common.UNKNOWN_TX)
    assert set(s._future_truth(query,truth['clean']['ids'])['y'])==set(range(10))
    wrong=dict(nt,tx=np.asarray(['6']*len(nt['ids'])));np.savez(source,**wrong)
    with pytest.raises(ValueError):s._source_truth(source,rows['source_DURA']['novel'])
    for key,value in (('receiver',np.full(18,26,np.int64)),('tx',np.asarray(['bad']*18)),('day',np.asarray(['2021_03_08']*18))):
        with subtests.test(key=key):
            np.savez(query,**dict(truth['clean'],**{key:value}))
            with pytest.raises(ValueError):s._future_truth(query,truth['clean']['ids'])


def test_actual_producer_gate_three_truths_then_bad_artifacts_never_read_truth(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_fresh_dura_run import fixture
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        done=m.run(c,'cpu');out=Path(c['output_root'])
        monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        proxy=type('Rows',(),dict(_rows=staticmethod(lambda root,role:m._source_rows(root,'source_DURA',role))))
        settings=type('Settings',(),dict(common=s.common,CONDITIONS=()))
        paths=truth_files(tmp_path,settings,proxy,out)
        for base in ('clean','star'):
            a,t=future(base);t['ids']=m._target_rows(out,base+'_DURA')['ids']
            p=tmp_path/(base+'_truth.npz');np.savez(p,**{k:v[::-1] for k,v in t.items()});paths.append(p)
        reads=[];source=s._source_truth;target=s._future_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1])
        monkeypatch.setattr(s,'_future_truth',lambda p,*a:(reads.append(Path(p).name),target(p,*a))[1])
        monkeypatch.setattr(s.fresh,'_prediction',lambda *a:pytest.fail('No fresh U-ID gate'))
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Independent CPU gate cannot load F0/G'))
        monkeypatch.setattr(m,'_load_model',lambda *a:pytest.fail('Independent CPU gate cannot load PCD'))
        monkeypatch.setattr(s.density,'fit_known',lambda *a:pytest.fail('No fitting'))
        monkeypatch.setattr(s.density,'fit_background',lambda *a:pytest.fail('No fitting'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_truth.npz','star_truth.npz']
        assert result['costs']['real_F_forward_rows']==done['real_F_forward_rows'] and result['costs']['G_forward_rows']==0
        assert result['diagnostic_completed'] and result['next_step']=='NO_PROMOTION' and result['adaptation_success'] is None
        assert result['methods'][s.PRIMARY]['paths']['clean_raw']['by_day']['2021_03_23']['A_K'] is None
        files=[out/'complete.json',out/'thresholds.npz',out/'models/E0.npz',out/'targets/clean_DURA/predictions.npz',
               out/'targets/clean_raw/predictions.npz',out/'targets/star_raw/predictions.npz']
        saved={p:p.read_bytes() for p in files}
        for defect in ('missing_complete','wrong_complete','raw_G_identity','new_U_read','tau','state','DURA_exact_control','raw_baseline','pair_ID','private_Q','missing_raw'):
            with subtests.test(defect=defect):
                for p,contents in saved.items():p.write_bytes(contents)
                reads.clear()
                if defect=='missing_complete':(out/'complete.json').unlink()
                elif defect=='missing_raw':(out/'targets/star_raw/predictions.npz').unlink()
                elif defect in ('wrong_complete','raw_G_identity','new_U_read'):
                    p=out/'complete.json';a=m._read(p)
                    if defect=='wrong_complete':a['all_six_paths_complete']=False
                    elif defect=='new_U_read':a['target_U_reads']=1
                    else:a['loaded_dura_identity']={'wrong':'G'}
                    p.unlink();m._write(p,a)
                else:
                    p=out/('thresholds.npz' if defect=='tau' else 'models/E0.npz' if defect=='state' else
                           'targets/clean_DURA/predictions.npz' if defect=='DURA_exact_control' else 'targets/clean_raw/predictions.npz')
                    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='tau':a['conditional'][3,0]+=.2
                    elif defect=='state':a['known_means'][0,0]+=.2
                    elif defect=='DURA_exact_control':a['default_logits'][0]+=1.
                    elif defect=='raw_baseline':a['baseline_closed'][0]=(a['closed'][0]+1)%6
                    elif defect=='pair_ID':a['ids'][0]='wrong'
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/('bad_'+defect))
                assert reads==[] and not (tmp_path/('bad_'+defect)).exists()
        for p,contents in saved.items():p.write_bytes(contents)
