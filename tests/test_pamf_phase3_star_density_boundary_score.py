"""Frozen shared density, complete source replay and independent three-truth score."""
import importlib
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_density_boundary_score')


def test_shared_conditional_density_reconstructs_three_feature_axes(subtests):
    s=api();rng=np.random.default_rng(31)
    state=dict(schema=np.asarray(s.density.SCHEMA),origin=np.zeros(160),whitener=np.eye(160),log_abs_det_whitener=np.asarray(0.),
        pooled_L_variance=np.asarray(1.),known_means=rng.normal(size=(90,160))*.1,known_variances=np.ones(90),known_weights=np.full(90,1/90),
        background_means=rng.normal(size=(2,160))*.1,background_variances=np.full(2,2.),background_weights=np.full(2,.5))
    n=4;z=rng.normal(size=(n,3,160));z=(z/np.linalg.norm(z,axis=2,keepdims=True)).astype(np.float32)
    closed=np.arange(n,dtype=np.int64);tau=np.arange(36,dtype=np.float64).reshape(6,6)/10
    raw=np.concatenate([s.conditional.raw_scores(state,z[:,i],closed)[:,[1,3]] for i in range(3)],axis=1)
    a=dict(ids=np.asarray(['q'+str(i) for i in range(n)]),default_logits=np.eye(6,dtype=np.float32)[:n],closed=closed,
        baseline_closed=closed.copy(),signal_features=z,raw_scores=raw,scores=raw-tau[:,closed].T)
    s._validate_rows(a,n,set(a),state,tau)
    for defect in ('feature_axis','method_axis','threshold_route','baseline'):
        with subtests.test(defect=defect):
            b={k:v.copy() for k,v in a.items()}
            if defect=='feature_axis':b['signal_features'][:,[1,2]]=b['signal_features'][:,[2,1]]
            elif defect=='method_axis':b['raw_scores'][:,[3,5]]=b['raw_scores'][:,[5,3]]
            elif defect=='threshold_route':b['scores'][0,5]+=.1
            else:b['baseline_closed'][0]=5
            with pytest.raises(ValueError):s._validate_rows(b,n,set(a),state,tau)


def test_complete_matched_training_trace_precedes_any_truth(tmp_path,monkeypatch,subtests):
    s=api();monkeypatch.setattr(s,'STEPS',2)
    monkeypatch.setattr(s,'COUNTS',dict(L=360,OE=90,V=180,novel=90))
    rows={}
    for arm in ('REPLAY','BOUNDARY'):
        rows[arm]=[dict(step=i,arm=arm,L_indices=list(range(192)),OE_indices=list(range(64)),source_indices_equal=True,
            source_loss=.5,point_anchor=.1,boundary_loss=.2,loss_total=.8,gradient_norms=None,gradient_norm_before_physical_clip=1.,
            sinc_lr=.1,sinc_optimizer_step=i,physical_clip_calls=i) for i in (1,2)]
        (tmp_path/arm).mkdir()
    def save():
        for arm,trace in rows.items():(tmp_path/arm/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace),encoding='utf-8')
    save();result=s._trace_summary(tmp_path)
    assert result['source_sampling_matched'] and result['total_optimizer_updates']==4
    assert result['arms']['BOUNDARY']['source_L_exposures']==384 and result['arms']['BOUNDARY']['source_OE_exposures']==128
    original=json.loads(json.dumps(rows))
    for defect in ('missing_step','unmatched_indices','nonfinite','wrong_clip'):
        with subtests.test(defect=defect):
            rows=json.loads(json.dumps(original))
            if defect=='missing_step':rows['BOUNDARY'].pop()
            elif defect=='unmatched_indices':rows['BOUNDARY'][1]['OE_indices'][0]=89
            elif defect=='nonfinite':rows['BOUNDARY'][1]['source_loss']=float('nan')
            else:rows['BOUNDARY'][1]['physical_clip_calls']=1
            save()
            with pytest.raises(ValueError):s._trace_summary(tmp_path)
    reads=[];monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source'));monkeypatch.setattr(s,'_target_truth',lambda *a:reads.append('Q'))
    monkeypatch.setattr(s,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('partial matrix')))
    with pytest.raises(ValueError,match='partial'):s.score('pred','source','clean','star',tmp_path/'score')
    assert not reads and not (tmp_path/'score').exists()


def test_primary_must_preserve_both_controls_even_when_absolute_goals_pass(tmp_path,monkeypatch):
    s=api();scenes=s.common.SCENES
    kg=[(c,r,sc) for c in range(6) for r in s.common.RECEIVERS for sc in scenes]
    ng=[(c,r,sc) for c in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for sc in scenes]
    k=dict(ids=np.asarray(['k'+str(i) for i in range(len(kg))]),y=np.asarray([x[0] for x in kg],np.int64),rx_i=np.asarray([x[1] for x in kg],np.int64),
        day_i=np.arange(len(kg),dtype=np.int64)%3+1,scenario=np.asarray([x[2] for x in kg]),scores=np.ones((len(kg),6),np.float64))
    k['closed']=k['y'].copy();k['baseline_closed']=k['y'].copy()
    n=dict(ids=np.asarray(['n'+str(i) for i in range(len(ng))]),rx_i=np.asarray([x[1] for x in ng],np.int64),day_i=np.arange(len(ng),dtype=np.int64)%3+1,scores=np.ones((len(ng),6),np.float64))
    nt=dict(tx=np.asarray([x[0] for x in ng]),scenario=np.asarray([x[2] for x in ng]))
    y=np.repeat(np.arange(9,dtype=np.int64),60);q=dict(ids=np.asarray(['q'+str(i) for i in range(len(y))]),closed=np.minimum(y,5),scores=np.ones((len(y),6),np.float64))
    q['baseline_closed']=q['closed'].copy();q['scores'][y>=6]=-1.
    ct=dict(y=y,receiver=np.full(len(y),26,np.int64),scenario=np.asarray(['clean']*len(y)));st=dict(ct,scenario=np.tile(np.asarray(scenes),len(y)//6))
    rows=dict(source_DURA=dict(known=k,novel=n),clean_DURA=deepcopy(q),star_DURA=deepcopy(q))
    marker=dict(code_commit='c'*40,scorer_training_trace=dict(source_sampling_matched=True),scorer_training_cost=dict(total_optimizer_updates=4440),
        **{key:0 for key in ('real_F_forward_rows','G_forward_rows','real_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows','synthetic_signal_forward_rows')})
    cal=dict(methods={name:dict(constraints_passed=True) for name in s.METHODS})
    monkeypatch.setattr(s,'_prediction',lambda *a:(marker,cal,rows));monkeypatch.setattr(s,'_source_truth',lambda *a:nt)
    monkeypatch.setattr(s,'_target_truth',lambda path,*a:ct if path=='clean' else st)
    result=s.score('pred','source','clean','star',tmp_path/'pass')
    assert result['development_gate_passed'] and result['source_known_12_passed']
    assert not result['methods'][s.PRIMARY]['paths']['source_DURA']['source_unknown_90_diagnostic_passed']
    # Match E0, but lose one correct acceptance against the same-budget REPLAY.
    rows['star_DURA']['scores'][0,[1,5]]=-1.
    result=s.score('pred','source','clean','star',tmp_path/'fail')
    assert result['target_protection_passed'] and not result['star_not_worse_than_both_controls']
    assert result['paired_primary_vs_E0_predicted']['star_DURA']['counts']['known_correct_accepted_net']==0
    assert result['paired_primary_vs_REPLAY_predicted']['star_DURA']['counts']['known_correct_accepted_net']==-1
    assert not result['development_gate_passed'] and result['next_step']=='NO_PROMOTION'
    assert result['A_N'] is result['H_OW'] is None and not result['final_phase3_complete']


def test_actual_complete_training_and_six_column_producer_before_three_truths(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_density_boundary_run import fixture
    from test_pamf_phase3_star_dura_best_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,inputs,backend):
        m.evaluate(c,'cpu');out=Path(c['output_root']);train=m._train();train_root=Path(c['train_config']['output_root'])
        assert events==['load_REPLAY','load_BOUNDARY','V','novel','clean_DURA','star_DURA']
        for key,value in (('COUNTS',counts),('Q_COUNT',m.Q_COUNT),('STEPS',train.STEPS)):monkeypatch.setattr(s,key,value)
        paths=truth_files(tmp_path,s,m,out);reads=[];source=s._source_truth;target=s._target_truth
        def source_truth(*a):reads.append('source');return source(*a)
        def target_truth(path,*a):reads.append(Path(path).name);return target(path,*a)
        monkeypatch.setattr(s,'_source_truth',source_truth);monkeypatch.setattr(s,'_target_truth',target_truth)
        # Teacher rows and checkpoints are not dependencies of independent CPU scoring.
        (train_root/'reference_signal.npz').unlink()
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('No neural backend in scorer'))
        monkeypatch.setattr(train,'load_trained_model',lambda *a:pytest.fail('No checkpoint loading in scorer'))
        monkeypatch.setattr(train.base,'load_role',lambda *a:pytest.fail('No source IQ/features in scorer'))
        monkeypatch.setattr(s.density,'fit_known',lambda *a:pytest.fail('No density fit in scorer'))
        monkeypatch.setattr(s.density,'fit_background',lambda *a:pytest.fail('No background fit in scorer'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No new calibration in scorer'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_truth.npz','star_truth.npz']
        assert result['truth_last'] and result['all_three_predictions_before_truth']
        assert result['training_trace']['total_optimizer_updates']==2*train.STEPS
        for arm in ('REPLAY','BOUNDARY'):
            assert result['training_trace']['arms'][arm]['actual_trace_rows']==train.STEPS
        assert result['source_density_models_fitted']==0 and result['new_source_threshold_sets']==4
        assert result['costs']['real_signal_forward_rows']==2*(counts['V']+counts['novel'])+4*m.Q_COUNT
        assert result['training_cost']['F_head_only_rows']==counts['L']+counts['OE']
        assert result['primary_deployed_neural_parameters']==834665 and result['all_three_models_resident_parameters']==1504107
        assert result['next_step']=='NO_PROMOTION' and not result['final_phase3_complete']
        assert all(len(a['paths']['source_DURA']['groups'])==42 for a in result['methods'].values())
        with pytest.raises(FileExistsError):s.score(out,*paths,tmp_path/'score')
        files=[out/'complete.json',out/'models/complete.json',out/'models/density.npz',out/'thresholds.npz',
            out/'source/source_DURA/novel_predictions.npz',out/'targets/star_DURA/predictions.npz',
            train_root/'complete.json',train_root/'density.npz',train_root/'BOUNDARY/steps.jsonl']
        saved={p:p.read_bytes() for p in files}
        for defect in ('incomplete_train','wrong_source_ancestor','unmatched_trace','training_GMM','prediction_GMM','old_tau','E0_control','new_raw','private_Q','missing_Q','not_frozen'):
            with subtests.test(defect=defect):
                for p,value in saved.items():p.write_bytes(value)
                reads.clear()
                if defect=='missing_Q':(out/'targets/star_DURA/predictions.npz').unlink()
                elif defect in ('incomplete_train','not_frozen'):
                    p=train_root/'complete.json' if defect=='incomplete_train' else out/'complete.json';a=m._read(p)
                    if defect=='incomplete_train':a['rows'].pop('BOUNDARY')
                    else:a['all_thresholds_frozen_before_novel_and_Q']=False
                    p.unlink();m._write(p,a)
                elif defect=='wrong_source_ancestor':
                    for p,key in ((out/'complete.json','model_identities'),(out/'models/complete.json','identities')):
                        a=m._read(p);a[key]['BOUNDARY']['target_U_used']=True;p.unlink();m._write(p,a)
                    p=train_root/'complete.json';a=m._read(p);a['rows']['BOUNDARY']['identity']['target_U_used']=True;a['identities']['BOUNDARY']['target_U_used']=True;p.unlink();m._write(p,a)
                elif defect=='unmatched_trace':
                    p=train_root/'BOUNDARY/steps.jsonl';a=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
                    a[-1]['OE_indices'][:2]=a[-1]['OE_indices'][1::-1]
                    p.write_text(''.join(json.dumps(row)+'\n' for row in a),encoding='utf-8')
                else:
                    p=(train_root/'density.npz' if defect=='training_GMM' else out/'models/density.npz' if defect=='prediction_GMM' else out/'thresholds.npz' if defect=='old_tau'
                        else out/'source/source_DURA/novel_predictions.npz' if defect=='E0_control' else out/'targets/star_DURA/predictions.npz')
                    with np.load(p,allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
                    if defect in ('training_GMM','prediction_GMM'):a['origin'][0]+=.1
                    elif defect=='old_tau':a['conditional'][0,0]+=.1
                    elif defect=='E0_control':a['scores'][0,0]+=.1
                    elif defect=='new_raw':a['raw_scores'][0,5]+=.1
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                with pytest.raises((ValueError,KeyError,FileNotFoundError)):
                    s.score(out,*paths,tmp_path/'invalid')
                assert not reads and not (tmp_path/'invalid').exists()
