"""U-only frozen mixture adapters, six readouts, and independent truth joins."""
import importlib
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_mixture_uda_score')


def test_fixed_primary_pairs_ratio_to_DA0_and_known_only_adapter():
    s=api();assert s.METHODS==('DA0_known','DA0_ratio','DA_K_known','DA_K_ratio','DA_M_known','DA_M_ratio') and s.PRIMARY=='DA_M_ratio'
    k=dict(y=np.asarray([0,1,2,3]),closed=np.asarray([0,0,2,3]),scores=np.ones((4,6),np.float64));k['scores'][:,1]=[1,-1,-1,1];k['scores'][:,3]=[-1,1,-1,1];k['scores'][:,5]=[1,1,-1,-1]
    n=dict(scores=np.ones((3,6),np.float64));n['scores'][:,1]=[-1,1,-1];n['scores'][:,3]=[1,-1,-1];n['scores'][:,5]=[-1,-1,1]
    a=s.paired_counts(k,n,np.ones(4,bool),np.ones(3,bool),primary_index=5,reference_index=1);b=s.paired_counts(k,n,np.ones(4,bool),np.ones(3,bool),primary_index=5,reference_index=3)
    assert a['known_correct_accepted_net']==-1 and a['known_accepted_net']==0 and a['unknown_rejected_net']==0
    assert b['known_correct_accepted_net']==0 and b['known_accepted_net']==0 and b['unknown_rejected_net']==0


def test_no_private_truth_if_an_adapter_or_prediction_is_incomplete(tmp_path,monkeypatch):
    s=api();reads=[];monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source'));monkeypatch.setattr(s,'_target_truth',lambda *a:reads.append('Q'))
    monkeypatch.setattr(s,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('adapter not frozen')))
    with pytest.raises(ValueError,match='adapter not frozen'):s.score('prediction','source','clean','star',tmp_path/'score')
    assert reads==[] and not (tmp_path/'score').exists()


def test_adaptation_requires_star_correct_gain_all_target_protection_and_scene_coverage(tmp_path,monkeypatch):
    s=api();scenes=s.common.SCENES
    kg=[(c,r,sc) for c in range(6) for r in s.common.RECEIVERS for sc in scenes];ng=[(c,r,sc) for c in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for sc in scenes]
    k=dict(ids=np.asarray(['k'+str(i) for i in range(len(kg))]),y=np.asarray([v[0] for v in kg],np.int64),rx_i=np.asarray([v[1] for v in kg],np.int64),day_i=np.arange(len(kg),dtype=np.int64)%3+1,scenario=np.asarray([v[2] for v in kg]),scores=np.ones((len(kg),6),np.float64))
    k['closed']=k['y'].copy();k['baseline_closed']=k['y'].copy()
    n=dict(ids=np.asarray(['n'+str(i) for i in range(len(ng))]),rx_i=np.asarray([v[1] for v in ng],np.int64),day_i=np.arange(len(ng),dtype=np.int64)%3+1,scores=np.ones((len(ng),6),np.float64))
    nt=dict(tx=np.asarray([v[0] for v in ng]),scenario=np.asarray([v[2] for v in ng]))
    y=np.repeat(np.arange(9,dtype=np.int64),6);q=dict(ids=np.asarray(['q'+str(i) for i in range(len(y))]),closed=np.minimum(y,5),scores=np.ones((len(y),6),np.float64));q['baseline_closed']=q['closed'].copy();q['scores'][y>=6]=-1.
    # DA0 loses one correct record per class; DA_M changes only rejection.
    q['scores'][np.arange(0,36,6),1]=-1.
    ct=dict(y=y,receiver=np.full(len(y),26,np.int64),scenario=np.asarray(['clean']*len(y)));st=dict(ct,scenario=np.tile(np.asarray(scenes),9))
    rows=dict(source_DURA=dict(known=k,novel=n),clean_DURA=deepcopy(q),star_DURA=deepcopy(q))
    marker=dict(code_commit='e'*40,real_F_forward_rows=384,real_signal_forward_rows=384,G_forward_rows=384,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,target_U_reads=384,target_Q_IQ_reads=0,source_IQ_reads=0)
    adapters={a:dict(delta=np.zeros(160),coefficients=np.zeros(1)) for a in ('DA_K','DA_M')};diag=dict(arms={a:dict(trace=[dict(objective=0.),dict(objective=1.)],group_prior=[1.,0.] if a=='DA_K' else [.5,.5]) for a in adapters})
    fit=dict(prior=dict(basis=np.eye(1,160),radius=np.asarray(1.)),adapters=adapters,diagnostics=diag);reads=[]
    monkeypatch.setattr(s,'_prediction',lambda *a:(marker,{},rows,fit));monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source') or nt);monkeypatch.setattr(s,'_target_truth',lambda p,*a:reads.append(p) or (ct if p=='clean' else st))
    r=s.score('prediction','source','clean','star',tmp_path/'pass')
    assert reads==['source','clean','star'] and r['adaptation_success'] and r['development_gate_passed'] and r['star_A_K_gain']>0
    assert not r['methods'][s.PRIMARY]['paths']['source_DURA']['source_unknown_90_diagnostic_passed']
    assert r['paired_primary_vs_DA0_ratio']['star_DURA']['counts']['known_correct_accepted_net']==6
    assert r['A_N'] is r['H_OW'] is None and not r['final_phase3_complete'] and r['full_goal_status']=='ACTIVE'
    rows['star_DURA']['scores'][:,1]=rows['star_DURA']['scores'][:,5]
    r=s.score('prediction','source','clean','star',tmp_path/'no_gain');assert r['target_protection_passed'] and not r['adaptation_success'] and r['star_A_K_gain']==0
    rows['star_DURA']['scores'][np.arange(0,36,6),1]=-1.;rows['star_DURA']['scores'][y>=6,5]=1.
    r=s.score('prediction','source','clean','star',tmp_path/'main_fail')
    assert r['methods']['DA_K_ratio']['paths']['star_DURA']['goals']['passed'] and not r['development_gate_passed'] and r['next_step']=='NO_PROMOTION'
    rows['star_DURA']['scores'][y>=6,5]=-1.;st['scenario'][(y==6)&(st['scenario']=='practical_high')]='practical_mid'
    r=s.score('prediction','source','clean','star',tmp_path/'missing')
    assert r['primary_summary']['star_DURA']['R_U']==1. and r['methods'][s.PRIMARY]['paths']['star_DURA']['by_scene']['practical_high']['R_U'] is None and not r['eligible_primary']


def test_actual_checkpoint_U_only_fits_to_independent_six_column_score_and_invalid_boundaries(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_mixture_uda_run import fixture
    from test_pamf_phase3_star_dura_best_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,u,events,seen,public):
        m.run(c,'cpu');out=Path(c['output_root']);monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT);monkeypatch.setattr(s,'U_COUNT',m.U_COUNT)
        paths=truth_files(tmp_path,s,m,out);reads=[];source=s._source_truth;target=s._target_truth
        def first(path,*a):reads.append('source');return source(path,*a)
        def query(path,*a):reads.append(Path(path).name);return target(path,*a)
        monkeypatch.setattr(s,'_source_truth',first);monkeypatch.setattr(s,'_target_truth',query)
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Independent score cannot load NN checkpoints'))
        monkeypatch.setattr(s.core,'fit_adapters',lambda *a:pytest.fail('Independent score cannot refit U'))
        monkeypatch.setattr(s.core,'build_prior',lambda *a:pytest.fail('Independent score cannot refit source prior'))
        monkeypatch.setattr(m.best.density,'fit_known',lambda *a:pytest.fail('No source density fitting'))
        monkeypatch.setattr(m.best.density,'fit_background',lambda *a:pytest.fail('No source density fitting'))
        monkeypatch.setattr(m.best.reference.risk,'calibrate_known',lambda *a:pytest.fail('No threshold fitting'))
        r=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_truth.npz','star_truth.npz'] and r['truth_last'] and r['all_three_predictions_before_truth'] and r['U_truth_reads']==0
        assert r['target_statistical_iterations_total']==40 and all(r['adapter_summary'][a]['iterations']==20 for a in ('DA_K','DA_M'))
        assert r['costs']['real_F_forward_rows']==r['costs']['real_signal_forward_rows']==r['costs']['G_forward_rows']==16
        assert r['new_neural_updates']==r['source_density_fit_updates']==r['threshold_fit_updates']==0 and not r['development_gate_passed']
        for name in s.METHODS:assert set(r['methods'][name]['paths'])==set(s.CONDITIONS) and len(r['methods'][name]['paths']['source_DURA']['groups'])==42
        for key in ('paired_primary_vs_DA0_ratio','paired_primary_vs_DA_K_ratio'):
            assert r[key]['source_DURA']['counts']['known_correct_accepted_net']==r[key]['source_DURA']['counts']['unknown_rejected_net']==0
        with pytest.raises(FileExistsError):s.score(out,*paths,tmp_path/'score')
        monkeypatch.setattr(s,'_source_truth',lambda *a:pytest.fail('Truth before complete adapter and prediction verification'))
        monkeypatch.setattr(s,'_target_truth',lambda *a:pytest.fail('Truth before complete adapter and prediction verification'))
        files=[out/'complete.json',out/'thresholds.npz',out/'adaptation/DA_M.npz',out/'adaptation/fit_diagnostics.json',out/'adaptation/U_features.npz',out/'targets/star_DURA/predictions.npz',out/'source/source_DURA/known_predictions.npz']
        originals={p:p.read_bytes() for p in files}
        for defect in ('unfrozen','tau','delta_bound','incomplete_EM','objective_decrease','private_U_TX','source_shift','wrong_default_identity','missing_Q'):
            with subtests.test(defect=defect):
                for p,v in originals.items():p.write_bytes(v)
                if defect=='missing_Q':(out/'targets/star_DURA/predictions.npz').unlink()
                elif defect in ('unfrozen','incomplete_EM','objective_decrease'):
                    p=out/('complete.json' if defect=='unfrozen' else 'adaptation/fit_diagnostics.json');a=m._read(p)
                    if defect=='unfrozen':a['adapters_frozen_before_source_and_Q']=False
                    elif defect=='incomplete_EM':a['arms']['DA_M']['trace'].pop()
                    else:a['arms']['DA_M']['trace'][-1]['objective']=a['arms']['DA_M']['trace'][-2]['objective']-1.
                    p.unlink();m._write(p,a)
                else:
                    p=out/('thresholds.npz' if defect=='tau' else 'adaptation/DA_M.npz' if defect=='delta_bound' else 'adaptation/U_features.npz' if defect=='private_U_TX' else 'source/source_DURA/known_predictions.npz' if defect=='source_shift' else 'targets/star_DURA/predictions.npz')
                    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='tau':a['conditional'][1,0]+=.1
                    elif defect=='delta_bound':a['delta']*=0.;a['delta'][0]=1e10
                    elif defect=='private_U_TX':a['tx']=np.zeros(m.U_COUNT,np.int64)
                    elif defect=='source_shift':a['scores'][0,5]+=.1
                    else:a['baseline_closed'][0]=(a['closed'][0]+1)%6
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,'source','clean','star',tmp_path/'invalid_score')
                assert not (tmp_path/'invalid_score/complete.json').exists()
