"""Fixed P1 readout, six paths, and path-specific original-F0 pairing."""
import importlib
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_dura_best_score')


def truth_files(root,s,m,output):
    novel=m._source_rows(output,'source_DURA','novel');grid=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for scene in s.common.SCENES]
    data=dict(ids=novel['ids'],rx_i=novel['rx_i'],day_i=novel['day_i'],tx=np.asarray([v[0] for v in grid]),scenario=np.asarray([v[2] for v in grid]))
    paths=[root/'source_truth.npz'];np.savez(paths[0],**{k:v[::-1] for k,v in data.items()})
    for base in ('clean','star'):
        a=m._target_rows(output,base+'_DURA');n=len(a['ids']);p=root/(base+'_truth.npz');paths.append(p)
        data=dict(ids=a['ids'],y=np.arange(n,dtype=np.int64)%9,receiver=np.full(n,26,np.int64),scenario=np.asarray(['clean' if base=='clean' else 'practical_low_urban']*n))
        np.savez(p,**{k:v[::-1] for k,v in data.items()})
    return paths


def test_DURA_pair_uses_each_path_identity_and_zero_margin_is_accepted():
    s=api();y=np.asarray([0,1,2,3],np.int64);ids=np.asarray(['a','b','c','d'])
    raw=dict(ids=ids,y=y,closed=np.asarray([0,0,2,3]),scores=np.column_stack((np.zeros(4),[1,-1,-1,0])).astype(np.float64))
    new=dict(ids=ids,y=y,closed=np.asarray([1,1,2,3]),scores=np.column_stack((np.zeros(4),[1,1,-1,0])).astype(np.float64))
    raw_n=dict(ids=np.asarray(['u','v','w']),scores=np.column_stack((np.zeros(3),[-1,0,-1])).astype(np.float64))
    new_n=dict(ids=raw_n['ids'],scores=np.column_stack((np.zeros(3),[0,-1,-1])).astype(np.float64))
    pair=s.paired_counts(new,raw,new_n,raw_n,1)
    assert pair['known_correct_accepted_gained']==pair['known_correct_accepted_lost']==1 and pair['known_correct_accepted_net']==0
    assert pair['known_accepted_net']==1 and pair['known_closed_correct_net']==0 and pair['known_closed_prediction_changed']==2
    assert pair['known_correct_rejected_net']==0 and pair['known_wrong_accepted_net']==1
    assert pair['known_correct_accepted_net']==pair['known_closed_correct_net']-pair['known_correct_rejected_net']
    assert pair['unknown_rejected_gained']==pair['unknown_rejected_lost']==1 and pair['unknown_rejected_net']==0
    assert s.METHODS==('known_support','density_ratio') and s.PRIMARY=='density_ratio'


def test_all_six_paths_must_complete_before_any_of_three_truth_reads(tmp_path,monkeypatch):
    s=api();reads=[];monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source'));monkeypatch.setattr(s,'_target_truth',lambda *a:reads.append('Q'))
    monkeypatch.setattr(s,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('missing star_raw')))
    with pytest.raises(ValueError,match='missing star_raw'):s.score('prediction','source','clean','star',tmp_path/'score')
    assert reads==[] and not (tmp_path/'score').exists()


def test_DURA_primary_targets_require_coverage_and_cannot_be_rescued_by_raw_or_control(tmp_path,monkeypatch):
    s=api();rxs=s.common.RECEIVERS;scenes=s.common.SCENES;kg=[(c,r,sc) for c in range(6) for r in rxs for sc in scenes];ng=[(c,r,sc) for c in s.common.UNKNOWN_TX for r in rxs for sc in scenes]
    k=dict(ids=np.asarray(['k'+str(i) for i in range(len(kg))]),y=np.asarray([v[0] for v in kg],np.int64),rx_i=np.asarray([v[1] for v in kg],np.int64),day_i=np.arange(len(kg),dtype=np.int64)%3+1,scenario=np.asarray([v[2] for v in kg]),scores=np.ones((len(kg),2),np.float64))
    k['closed']=k['y'].copy();k['baseline_closed']=k['y'].copy()
    n=dict(ids=np.asarray(['n'+str(i) for i in range(len(ng))]),rx_i=np.asarray([v[1] for v in ng],np.int64),day_i=np.arange(len(ng),dtype=np.int64)%3+1,scores=np.ones((len(ng),2),np.float64))
    nt=dict(tx=np.asarray([v[0] for v in ng]),scenario=np.asarray([v[2] for v in ng]))
    y=np.repeat(np.arange(9,dtype=np.int64),6);Q=dict(ids=np.asarray(['q'+str(i) for i in range(len(y))]),closed=np.minimum(y,5),scores=np.ones((len(y),2),np.float64));Q['baseline_closed']=Q['closed'].copy();Q['scores'][y>=6]=-1.
    ct=dict(y=y,receiver=np.full(len(y),26,np.int64),scenario=np.asarray(['clean']*len(y)));st=dict(ct,scenario=np.tile(np.asarray(scenes),9))
    rows=dict(source_raw=dict(known=deepcopy(k),novel=deepcopy(n)),source_DURA=dict(known=deepcopy(k),novel=deepcopy(n)),**{c:deepcopy(Q) for c in s.CONDITIONS[2:]})
    marker=dict(code_commit='e'*40,reference_prediction_identity=dict(method_names=['P1_known12','P1_ratio12']),real_F_forward_rows=33606,real_signal_forward_rows=33606,G_forward_rows=1728,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2)
    cal=dict(methods={name:dict(constraints_passed=True) for name in ('P1_known12','P1_ratio12')});reads=[]
    monkeypatch.setattr(s,'_prediction',lambda *a:(marker,cal,rows));monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source') or nt)
    monkeypatch.setattr(s,'_target_truth',lambda path,*a:reads.append(path) or (ct if path=='clean' else st))
    result=s.score('prediction','source','clean','star',tmp_path/'pass')
    assert reads==['source','clean','star'] and result['development_gate_passed']
    assert not result['methods'][s.PRIMARY]['paths']['source_DURA']['source_unknown_90_diagnostic_passed']
    assert result['raw_paths_diagnostic_only'] and result['A_N'] is result['H_OW'] is None and not result['final_phase3_complete']
    assert len(result['paired_DURA_minus_raw'])==2 and result['full_goal_status']=='ACTIVE'
    rows['star_DURA']['scores'][y>=6,1]=1.
    result=s.score('prediction','source','clean','star',tmp_path/'main_fail')
    assert result['methods']['known_support']['paths']['star_DURA']['goals']['passed'] and result['methods'][s.PRIMARY]['paths']['star_raw']['goals']['passed']
    assert not result['development_gate_passed'] and result['next_step']=='NO_PROMOTION'
    rows['star_DURA']['scores'][y>=6,1]=-1.;st['scenario'][(y==6)&(st['scenario']=='practical_high')]='practical_mid'
    result=s.score('prediction','source','clean','star',tmp_path/'missing')
    assert result['primary_summary']['star_DURA']['R_U']==1. and result['methods'][s.PRIMARY]['paths']['star_DURA']['by_scene']['practical_high']['R_U'] is None
    assert not result['eligible_primary']


def test_clean95_and_original_F0_two_pp_are_separate_from_acceptance():
    s=api();y=np.repeat(np.arange(9,dtype=np.int64),100);n=len(y);a=dict(ids=np.asarray(['q'+str(i) for i in range(n)]),closed=np.minimum(y,5),scores=np.ones((n,2),np.float64));a['baseline_closed']=a['closed'].copy();a['scores'][y>=6]=-1.
    t=dict(y=y,receiver=np.full(n,26,np.int64),scenario=np.asarray(['clean']*n));a['scores'][np.flatnonzero(y==6)[:15],1]=0.
    assert s._target(a,t,'clean_DURA',1)['goals']['passed']
    a['scores'][np.flatnonzero(y==6)[15],1]=0.;assert not s._target(a,t,'clean_DURA',1)['goals']['passed']
    a['scores'][y>=6]=-1.;a['scores'][np.flatnonzero(y==0)[:13],1]=-1.;r=s._target(a,t,'clean_DURA',1)
    assert r['overall']['C_K']>.90 and r['overall']['original_closed_minus_new_open']>.02 and not r['goals']['passed']


def test_actual_fixed_P1_checkpoint_six_paths_to_three_truth_joins_without_refit(tmp_path,monkeypatch):
    from test_pamf_phase3_star_dura_best_run import fixture
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,public,events,seen,values):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT);paths=truth_files(tmp_path,s,m,c['output_root'])
        private=[];source=s._source_truth;target=s._target_truth
        def first(path,*a):private.append('source');return source(path,*a)
        def query(path,*a):private.append(Path(path).name);return target(path,*a)
        monkeypatch.setattr(s,'_source_truth',first);monkeypatch.setattr(s,'_target_truth',query)
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Independent CPU score cannot load checkpoints'))
        monkeypatch.setattr(s.density,'fit_known',lambda *a:pytest.fail('No density fitting'))
        monkeypatch.setattr(s.density,'fit_background',lambda *a:pytest.fail('No density fitting'))
        for risk in (m.reference.risk,m.signal_eval.risk):monkeypatch.setattr(risk,'calibrate_known',lambda *a:pytest.fail('No threshold fitting'))
        result=s.score(c['output_root'],*paths,tmp_path/'score')
        assert private==['source','clean_truth.npz','star_truth.npz'] and result['truth_last'] and result['all_six_predictions_before_truth']
        assert result['costs']['real_F_forward_rows']==result['costs']['real_signal_forward_rows']==m.COUNTS['V']+m.COUNTS['novel']+4*m.Q_COUNT
        assert result['costs']['G_forward_rows']==2*m.Q_COUNT and result['neural_updates']==result['density_fit_updates']==result['threshold_fit_updates']==0
        for name in s.METHODS:
            assert len(result['methods'][name]['paths'])==6 and len(result['methods'][name]['paths']['source_DURA']['groups'])==42
            assert set(result['paired_DURA_minus_raw'][name])=={'source','clean','star'}
        assert not result['development_gate_passed'] and result['primary_summary']['clean_DURA']['R_U'] is None
        assert result['source_truth_reads']==1 and result['target_truth_reads']==2 and not result['final_phase3_complete']
        with pytest.raises(FileExistsError):s.score(c['output_root'],*paths,tmp_path/'score')


def test_model_tau_control_pairing_baseline_and_complete_matrix_verify_before_truth(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_dura_best_run import fixture
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,public,events,seen,values):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT);out=Path(c['output_root'])
        files=[out/'complete.json',out/'thresholds.npz',out/'models/density.npz',out/'source/source_DURA/novel_predictions.npz',out/'source/source_raw/novel_predictions.npz',out/'targets/clean_raw/predictions.npz',out/'targets/star_raw/predictions.npz']
        originals={p:p.read_bytes() for p in files}
        monkeypatch.setattr(s,'_source_truth',lambda *a:pytest.fail('Private source labels before six-path validation'))
        monkeypatch.setattr(s,'_target_truth',lambda *a:pytest.fail('Private query labels before six-path validation'))
        for defect in ('P0_theta','tau','state','source_DURA_control','wrong_raw_baseline','pair_ID','private_Q','missing_raw'):
            with subtests.test(defect=defect):
                for p,v in originals.items():p.write_bytes(v)
                if defect=='P0_theta':
                    p=out/'complete.json';a=m._read(p);a['model_identity']=pm['model_identity'];p.unlink();m._write(p,a)
                elif defect=='missing_raw':(out/'targets/star_raw/predictions.npz').unlink()
                else:
                    p=out/('thresholds.npz' if defect=='tau' else 'models/density.npz' if defect=='state' else 'source/source_DURA/novel_predictions.npz' if defect=='source_DURA_control' else 'targets/clean_raw/predictions.npz')
                    with np.load(p) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='tau':a['conditional'][1,0]+=.1
                    elif defect=='state':a['known_means'][0,0]+=.1
                    elif defect=='source_DURA_control':a['scores'][0,0]+=.1
                    elif defect=='wrong_raw_baseline':a['baseline_closed'][0]=(a['closed'][0]+1)%6
                    elif defect=='pair_ID':a['ids'][0]='wrong'
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,'source','clean','star',tmp_path/'score')
                assert not (tmp_path/'score/complete.json').exists()
