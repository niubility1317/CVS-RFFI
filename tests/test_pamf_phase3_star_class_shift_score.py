"""Frozen class shifts, immutable shared controls, and truth-last scoring."""
import importlib
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_class_shift_score')


def test_primary_and_fixed_identity_paired_counts():
    s=api();assert s.METHODS==('DA0_known','DA0_ratio','SHARED_known','SHARED_ratio','CLASS_known','CLASS_ratio') and s.PRIMARY=='CLASS_ratio'
    k=dict(y=np.asarray([0,1,2,3]),closed=np.asarray([0,0,2,3]),scores=np.ones((4,6),np.float64));k['scores'][:,1]=[1,-1,-1,1];k['scores'][:,3]=[-1,1,-1,1];k['scores'][:,5]=[1,1,-1,-1]
    n=dict(scores=np.ones((3,6),np.float64));n['scores'][:,1]=[-1,1,-1];n['scores'][:,3]=[1,-1,-1];n['scores'][:,5]=[-1,-1,1]
    a=s._pair(k,n,5,1);b=s._pair(k,n,5,3)
    assert a['known_correct_accepted_net']==-1 and a['known_accepted_net']==0 and a['unknown_rejected_net']==0
    assert b['known_correct_accepted_net']==b['known_accepted_net']==b['unknown_rejected_net']==0


def test_incomplete_class_shift_or_query_prevents_all_truth_reads(tmp_path,monkeypatch):
    s=api();reads=[];monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source'));monkeypatch.setattr(s,'_target_truth',lambda *a:reads.append('Q'))
    monkeypatch.setattr(s,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('CLASS not frozen')))
    with pytest.raises(ValueError,match='CLASS not frozen'):s.score('prediction','source','clean','star',tmp_path/'score')
    assert reads==[] and not (tmp_path/'score').exists()


def test_class_trace_uses_RMS_not_per_class_max_and_requires_complete_monotonic_fit():
    s=api();basis=np.zeros((1,6,160),np.float64);basis[0,0,0]=np.sqrt(6.)
    prior=dict(schema=np.asarray(s.core.PRIOR_SCHEMA),basis=basis,variances=np.ones(1,np.float64),radius=np.asarray(1.,np.float64))
    coefficients=np.asarray([.75],np.float64);delta=np.einsum('k,kcd->cd',coefficients,basis)
    adapter=dict(schema=np.asarray(s.core.ADAPTER_SCHEMA),delta=delta,background_delta=delta.mean(0),coefficients=coefficients)
    s.core.validate_adapter(adapter,prior)
    trace=[dict(iteration=i,objective=float(i),coefficient_norm=.75*i/20,shift_rms=.75*i/20,class_shift_max=float(np.sqrt(6.)*.75*i/20),known_posterior_mass=.5,constraint_multiplier=0.) for i in range(21)]
    d=dict(iterations=20,U_count=s.U_COUNT,trace=trace,group_prior=[.5,.5],source_rank=1,source_radius=1.,source_prior_frozen_before_U=True,U_only=True,Q_inputs_used=False,true_U_labels_used=False,true_unknown_count_used=False,default_logits_used=False,source_state_unchanged_by_fit=True,fixed_iteration_checkpoint=True)
    assert np.linalg.norm(delta,axis=1).max()>1.;s._trace(d,prior,adapter,True)
    bad=deepcopy(d);bad['trace'].pop()
    with pytest.raises(ValueError,match='iteration0..20'):s._trace(bad,prior,adapter,True)
    bad=deepcopy(d);bad['trace'][20]['objective']=18.
    with pytest.raises(ValueError,match='monotonicity'):s._trace(bad,prior,adapter,True)


def test_success_needs_fixed_main_gain_all_protection_and_scene_coverage(tmp_path,monkeypatch):
    s=api();scenes=s.common.SCENES;kg=[(c,r,sc) for c in range(6) for r in s.common.RECEIVERS for sc in scenes];ng=[(c,r,sc) for c in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for sc in scenes]
    k=dict(ids=np.asarray(['k'+str(i) for i in range(len(kg))]),y=np.asarray([v[0] for v in kg],np.int64),rx_i=np.asarray([v[1] for v in kg],np.int64),day_i=np.arange(len(kg),dtype=np.int64)%3+1,scenario=np.asarray([v[2] for v in kg]),scores=np.ones((len(kg),6),np.float64));k['closed']=k['y'].copy();k['baseline_closed']=k['y'].copy()
    n=dict(ids=np.asarray(['n'+str(i) for i in range(len(ng))]),rx_i=np.asarray([v[1] for v in ng],np.int64),day_i=np.arange(len(ng),dtype=np.int64)%3+1,scores=np.ones((len(ng),6),np.float64));nt=dict(tx=np.asarray([v[0] for v in ng]),scenario=np.asarray([v[2] for v in ng]))
    y=np.repeat(np.arange(9,dtype=np.int64),6);q=dict(ids=np.asarray(['q'+str(i) for i in range(len(y))]),closed=np.minimum(y,5),scores=np.ones((len(y),6),np.float64));q['baseline_closed']=q['closed'].copy();q['scores'][y>=6]=-1.;q['scores'][np.arange(0,36,6),1]=-1.
    ct=dict(y=y,receiver=np.full(len(y),26,np.int64),scenario=np.asarray(['clean']*len(y)));st=dict(ct,scenario=np.tile(np.asarray(scenes),9));rows=dict(source_DURA=dict(known=k,novel=n),clean_DURA=deepcopy(q),star_DURA=deepcopy(q))
    marker=dict(code_commit='e'*40,real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,target_U_reads=384,U_feature_rows_read=384,target_Q_IQ_reads=0,source_IQ_reads=0)
    d=dict(trace=[dict(objective=0.),dict(objective=1.)],group_prior=[.5,.5]);fit=dict(prior=dict(basis=np.zeros((1,6,160)),radius=np.asarray(1.)),adapter=dict(delta=np.zeros((6,160)),background_delta=np.zeros(160),coefficients=np.zeros(1)),diagnostics=d,shared_adapter=dict(delta=np.zeros(160)),shared_diagnostics=dict(arms=dict(DA_M=d)))
    monkeypatch.setattr(s,'_prediction',lambda *a:(marker,{},rows,fit));reads=[];monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source') or nt);monkeypatch.setattr(s,'_target_truth',lambda p,*a:reads.append(p) or (ct if p=='clean' else st))
    r=s.score('prediction','source','clean','star',tmp_path/'pass')
    assert reads==['source','clean','star'] and r['adaptation_success'] and r['star_A_K_gain']>0
    assert not r['methods'][s.PRIMARY]['paths']['source_DURA']['source_unknown_90_diagnostic_passed']
    assert r['new_target_statistical_iterations']==r['inherited_shared_adapter_iterations']==20 and r['target_class_mode_adaptation']
    assert r['A_N'] is r['H_OW'] is None and r['full_goal_status']=='ACTIVE' and not r['final_phase3_complete']
    rows['star_DURA']['scores'][:,1]=rows['star_DURA']['scores'][:,5]
    r=s.score('prediction','source','clean','star',tmp_path/'no_gain');assert r['target_protection_passed'] and not r['adaptation_success']
    rows['star_DURA']['scores'][np.arange(0,36,6),1]=-1.;rows['star_DURA']['scores'][y>=6,5]=1.
    r=s.score('prediction','source','clean','star',tmp_path/'main_fail');assert r['methods']['SHARED_ratio']['paths']['star_DURA']['goals']['passed'] and not r['development_gate_passed']
    rows['star_DURA']['scores'][y>=6,5]=-1.;st['scenario'][(y==6)&(st['scenario']=='practical_high')]='practical_mid'
    r=s.score('prediction','source','clean','star',tmp_path/'missing');assert r['primary_summary']['star_DURA']['R_U']==1. and r['methods'][s.PRIMARY]['paths']['star_DURA']['by_scene']['practical_high']['R_U'] is None and not r['eligible_primary']


def test_cached_U_class_fit_to_independent_CPU_score_and_invalid_external_states(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_class_shift_run import fixture
    from test_pamf_phase3_star_dura_best_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,pm,am,b,u,events,seen):
        m.run(c,'cpu');root=Path(c['output_root']);monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT);monkeypatch.setattr(s,'U_COUNT',m.U_COUNT)
        paths=truth_files(tmp_path,s,m,root);source=s._source_truth;target=s._target_truth;reads=[]
        def source_truth(path,*a):reads.append('source');return source(path,*a)
        def target_truth(path,*a):reads.append(Path(path).name);return target(path,*a)
        monkeypatch.setattr(s,'_source_truth',source_truth);monkeypatch.setattr(s,'_target_truth',target_truth)
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('CPU scoring cannot load checkpoints'))
        monkeypatch.setattr(s.core,'fit_adapter',lambda *a:pytest.fail('No CLASS EM refit'))
        monkeypatch.setattr(s.shared_core,'fit_adapters',lambda *a:pytest.fail('No SHARED EM refit'))
        monkeypatch.setattr(m.best.density,'fit_known',lambda *a:pytest.fail('No source density refit'))
        monkeypatch.setattr(m.best.density,'fit_background',lambda *a:pytest.fail('No source density refit'))
        monkeypatch.setattr(m.best.reference.risk,'calibrate_known',lambda *a:pytest.fail('No threshold refit'))
        r=s.score(root,*paths,tmp_path/'score')
        assert reads==['source','clean_truth.npz','star_truth.npz'] and r['truth_last'] and r['all_three_predictions_before_truth'] and r['U_truth_reads']==0
        assert r['new_target_statistical_iterations']==r['inherited_shared_adapter_iterations']==20 and r['target_class_mode_adaptation']
        assert r['costs']['real_F_forward_rows']==r['costs']['real_signal_forward_rows']==r['costs']['G_forward_rows']==0 and r['costs']['U_feature_rows_read']==16
        assert r['costs']['synthetic_F_forward_rows']==r['costs']['synthetic_signal_forward_rows']==r['costs']['G_synthetic_forward_rows']==2
        assert r['new_neural_updates']==r['source_density_fit_updates']==r['threshold_fit_updates']==0 and not r['development_gate_passed']
        for name in s.METHODS:assert len(r['methods'][name]['paths'])==3 and len(r['methods'][name]['paths']['source_DURA']['groups'])==42
        for key in ('paired_primary_vs_DA0_ratio','paired_primary_vs_SHARED_ratio'):assert r[key]['source_DURA']['counts']['known_correct_accepted_net']==r[key]['source_DURA']['counts']['unknown_rejected_net']==0
        with pytest.raises(FileExistsError):s.score(root,*paths,tmp_path/'score')
        monkeypatch.setattr(s,'_source_truth',lambda *a:pytest.fail('Private source truth before complete state validation'))
        monkeypatch.setattr(s,'_target_truth',lambda *a:pytest.fail('Private Q truth before complete state validation'))
        files=[root/'complete.json',root/'thresholds.npz',root/'adaptation/source_prior.npz',root/'adaptation/CLASS.npz',root/'adaptation/SHARED.npz',root/'adaptation/shared_diagnostics.json',root/'adaptation/U_features.npz',root/'targets/star_DURA/predictions.npz']
        originals={p:p.read_bytes() for p in files}
        for defect in ('unfrozen','tau','prior_not_source','background_not_mean','changed_SHARED','changed_shared_trace','private_U_label','changed_identity','wrong_margin','missing_Q'):
            with subtests.test(defect=defect):
                for p,v in originals.items():p.write_bytes(v)
                if defect=='missing_Q':(root/'targets/star_DURA/predictions.npz').unlink()
                elif defect in ('unfrozen','changed_shared_trace'):
                    p=root/('complete.json' if defect=='unfrozen' else 'adaptation/shared_diagnostics.json');a=m._read(p)
                    if defect=='unfrozen':a['adapters_frozen_before_source_and_Q']=False
                    else:a['arms']['DA_M']['trace'][-1]['objective']+=1.
                    p.unlink();m._write(p,a)
                else:
                    rel={'tau':'thresholds.npz','prior_not_source':'adaptation/source_prior.npz','background_not_mean':'adaptation/CLASS.npz','changed_SHARED':'adaptation/SHARED.npz','private_U_label':'adaptation/U_features.npz','changed_identity':'targets/star_DURA/predictions.npz','wrong_margin':'targets/star_DURA/predictions.npz'}[defect];p=root/rel
                    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='tau':a['conditional'][1,0]+=.1
                    elif defect=='prior_not_source':a['variances'][0]*=2.
                    elif defect=='background_not_mean':a['background_delta'][0]+=.1
                    elif defect=='changed_SHARED':a['delta'][0]+=.1
                    elif defect=='private_U_label':a['y']=np.zeros(m.U_COUNT,np.int64)
                    elif defect=='changed_identity':a['baseline_closed'][0]=(a['closed'][0]+1)%6
                    else:a['scores'][0,5]+=.1
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(root,'source','clean','star',tmp_path/'invalid')
                assert not (tmp_path/'invalid/complete.json').exists()
