"""Two-arm source continuation versus U-SSL, with independent truth joins."""
import importlib
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_signal_ssl_score')


def test_primary_cannot_be_replaced_by_source_or_DA0():
    s=api();assert s.METHODS==('DA0_known','DA0_ratio','SOURCE_known','SOURCE_ratio','SSL_known','SSL_ratio') and s.PRIMARY=='SSL_ratio'
    k=dict(y=np.asarray([0,1,2,3]),closed=np.asarray([0,0,2,3]),scores=np.ones((4,6),np.float64));k['scores'][:,1]=[1,-1,-1,1];k['scores'][:,3]=[-1,1,-1,1];k['scores'][:,5]=[1,1,-1,-1]
    n=dict(scores=np.ones((3,6),np.float64));n['scores'][:,1]=[-1,1,-1];n['scores'][:,3]=[1,-1,-1];n['scores'][:,5]=[-1,-1,1]
    assert s._pair(k,n,5,1)['known_correct_accepted_net']==-1
    assert s._pair(k,n,5,3)['known_correct_accepted_net']==0


def test_no_truth_before_all_models_thresholds_and_three_predictions_complete(tmp_path,monkeypatch):
    s=api();reads=[];monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source'));monkeypatch.setattr(s,'_target_truth',lambda *a:reads.append('Q'))
    monkeypatch.setattr(s,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('incomplete six-column matrix')))
    with pytest.raises(ValueError,match='six-column'):s.score('prediction','source','clean','star',tmp_path/'score')
    assert not reads and not (tmp_path/'score').exists()


def test_three_representations_six_density_columns_and_old_identity_reconstruct(subtests):
    s=api();rng=np.random.default_rng(1);states={}
    for i,name in enumerate(s.MODEL_NAMES):
        states[name]=dict(schema=np.asarray(s.density.SCHEMA),origin=np.zeros(160),whitener=np.eye(160),log_abs_det_whitener=np.asarray(0.),pooled_L_variance=np.asarray(1.),known_means=np.full((90,160),i*.1),known_variances=np.ones(90),known_weights=np.full(90,1/90),background_means=np.full((1,160),.2),background_variances=np.asarray([2.]),background_weights=np.asarray([1.]))
    n=4;z=rng.normal(size=(n,3,160));z=(z/np.linalg.norm(z,axis=2,keepdims=True)).astype(np.float32);closed=np.arange(n,dtype=np.int64);logits=np.eye(6,dtype=np.float32)[:n];tau=np.arange(36,dtype=np.float64).reshape(6,6)/10
    raw=np.column_stack([s.density.score({k:states[name][k] for k in s.density.KNOWN_FIELDS} if j==0 else states[name],z[:,i],'known_only' if j==0 else 'density_ratio') for i,name in enumerate(s.MODEL_NAMES) for j in range(2)])
    rows=dict(ids=np.asarray(['q'+str(i) for i in range(n)]),default_logits=logits,closed=closed,baseline_closed=closed.copy(),signal_features=z,raw_scores=raw,scores=raw-tau[:,closed].T);fields=set(rows)
    s._validate_rows(rows,n,fields,states,tau)
    for defect in ('feature_axis','method_axis','threshold_route','baseline'):
        with subtests.test(defect=defect):
            bad={k:v.copy() for k,v in rows.items()}
            if defect=='feature_axis':bad['signal_features'][:,[1,2]]=bad['signal_features'][:,[2,1]]
            elif defect=='method_axis':bad['raw_scores'][:,[3,5]]=bad['raw_scores'][:,[5,3]]
            elif defect=='threshold_route':bad['scores'][0,5]+=.1
            else:bad['baseline_closed'][0]=5
            with pytest.raises(ValueError):s._validate_rows(bad,n,fields,states,tau)


def test_SSL_must_improve_both_DA0_and_SOURCE_and_preserve_all_required_scenes(tmp_path,monkeypatch):
    s=api();scenes=s.common.SCENES;kg=[(c,r,sc) for c in range(6) for r in s.common.RECEIVERS for sc in scenes];ng=[(c,r,sc) for c in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for sc in scenes]
    k=dict(ids=np.asarray(['k'+str(i) for i in range(len(kg))]),y=np.asarray([v[0] for v in kg],np.int64),rx_i=np.asarray([v[1] for v in kg],np.int64),day_i=np.arange(len(kg),dtype=np.int64)%3+1,scenario=np.asarray([v[2] for v in kg]),scores=np.ones((len(kg),6),np.float64));k['closed']=k['y'].copy();k['baseline_closed']=k['y'].copy()
    n=dict(ids=np.asarray(['n'+str(i) for i in range(len(ng))]),rx_i=np.asarray([v[1] for v in ng],np.int64),day_i=np.arange(len(ng),dtype=np.int64)%3+1,scores=np.ones((len(ng),6),np.float64));nt=dict(tx=np.asarray([v[0] for v in ng]),scenario=np.asarray([v[2] for v in ng]))
    y=np.repeat(np.arange(9,dtype=np.int64),6);q=dict(ids=np.asarray(['q'+str(i) for i in range(len(y))]),closed=np.minimum(y,5),scores=np.ones((len(y),6),np.float64));q['baseline_closed']=q['closed'].copy();q['scores'][y>=6]=-1.;q['scores'][np.arange(0,36,6),1]=-1.
    ct=dict(y=y,receiver=np.full(len(y),26,np.int64),scenario=np.asarray(['clean']*len(y)));st=dict(ct,scenario=np.tile(np.asarray(scenes),9));rows=dict(source_DURA=dict(known=k,novel=n),clean_DURA=deepcopy(q),star_DURA=deepcopy(q))
    marker=dict(code_commit='e'*40,**{k:0 for k in ('source_IQ_reads','target_Q_IQ_reads','target_U_reads','U_ID_metadata_reads','training_target_U_count','real_F_forward_rows','G_forward_rows','real_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows','synthetic_signal_forward_rows')},real_signal_forward_rows_by_arm={name:0 for name in s.MODEL_NAMES});reads=[];monkeypatch.setattr(s,'_prediction',lambda *a:(marker,{},rows));monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source') or nt);monkeypatch.setattr(s,'_target_truth',lambda p,*a:reads.append(p) or (ct if p=='clean' else st))
    r=s.score('prediction','source','clean','star',tmp_path/'continuation_only')
    assert r['star_A_K_gain_vs_DA0']>0 and r['star_A_K_gain_vs_SOURCE']==0 and r['target_protection_passed'] and not r['adaptation_success']
    rows['star_DURA']['scores'][np.arange(0,36,6),3]=-1.
    r=s.score('prediction','source','clean','star',tmp_path/'pass')
    assert r['adaptation_success'] and r['star_A_K_gain_over_both_controls'] and reads==['source','clean','star']*2
    assert r['SOURCE_weights_source_only'] and not r['TARGET_SSL_weights_source_only'] and r['TARGET_SSL_weights_use_target_U']
    assert r['new_signal_optimizer_updates_total']==888 and r['new_source_density_models']==2 and r['new_source_threshold_sets']==4 and r['TARGET_U_augmented_training_exposures']==56832
    assert not r['methods'][s.PRIMARY]['paths']['source_DURA']['source_unknown_90_diagnostic_passed'] and not r['final_phase3_complete'] and r['A_N'] is r['H_OW'] is None
    rows['star_DURA']['scores'][np.arange(0,36,6),3]=1.;rows['star_DURA']['scores'][y>=6,5]=1.
    r=s.score('prediction','source','clean','star',tmp_path/'main_fail');assert r['methods']['SOURCE_ratio']['paths']['star_DURA']['goals']['passed'] and not r['development_gate_passed']
    rows['star_DURA']['scores'][np.arange(0,36,6),3]=-1.;rows['star_DURA']['scores'][y>=6,5]=-1.;st['scenario'][(y==6)&(st['scenario']=='practical_high')]='practical_mid'
    r=s.score('prediction','source','clean','star',tmp_path/'missing');assert r['primary_summary']['star_DURA']['R_U']==1. and r['methods'][s.PRIMARY]['paths']['star_DURA']['by_scene']['practical_high']['R_U'] is None and not r['eligible_primary']


def test_actual_frozen_two_arm_pipeline_to_CPU_score_then_reject_bad_states_before_truth(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_signal_ssl_run import fixture
    from test_pamf_phase3_star_dura_best_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as values:
        m,c=values[:2];m.evaluate(c,'cpu');root=Path(c['output_root']);train=m._train()
        for key,value in (('COUNTS',m.COUNTS),('Q_COUNT',m.Q_COUNT),('U_COUNT',train.U_COUNT),('STEPS',train.STEPS),('U_BATCH',train.U_BATCH)):monkeypatch.setattr(s,key,value)
        paths=truth_files(tmp_path,s,m,root);source=s._source_truth;target=s._target_truth;reads=[]
        def source_truth(path,*a):reads.append('source');return source(path,*a)
        def target_truth(path,*a):reads.append(Path(path).name);return target(path,*a)
        monkeypatch.setattr(s,'_source_truth',source_truth);monkeypatch.setattr(s,'_target_truth',target_truth)
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('CPU scorer cannot load frozen NN checkpoints'))
        monkeypatch.setattr(m,'_load_model',lambda *a:pytest.fail('CPU scorer cannot load adapted models'))
        monkeypatch.setattr(train,'train',lambda *a:pytest.fail('No neural training in scorer'))
        monkeypatch.setattr(train,'load_trained_model',lambda *a:pytest.fail('No checkpoint loading in scorer'))
        monkeypatch.setattr(m.density,'fit_known',lambda *a:pytest.fail('No density fitting in scorer'))
        monkeypatch.setattr(m.density,'fit_background',lambda *a:pytest.fail('No density fitting in scorer'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No threshold fitting in scorer'))
        # This cache also contains restored IQ. The scorer may materialize IDs
        # only, for physical disjointness; it must not reopen IQ or U truth.
        original_load=np.load;u_path=Path(c['train_config']['output_root'])/'U_restored.npz';U_access=[]
        class IDsOnly:
            def __init__(self,z):self.z=z
            def __enter__(self):return self
            def __exit__(self,*a):self.z.close()
            def __getitem__(self,key):
                assert key=='ids';U_access.append(key);return self.z[key]
        def load(path,*a,**kw):
            z=original_load(path,*a,**kw);return IDsOnly(z) if Path(path)==u_path else z
        monkeypatch.setattr(np,'load',load)
        r=s.score(root,*paths,tmp_path/'score')
        assert reads==['source','clean_truth.npz','star_truth.npz'] and U_access==['ids']
        assert r['truth_last'] and r['all_three_predictions_before_truth'] and r['U_truth_reads']==0
        assert r['SOURCE_weights_source_only'] and not r['TARGET_SSL_weights_source_only'] and r['TARGET_SSL_weights_use_target_U']
        assert r['new_signal_optimizer_updates_total']==2*train.STEPS and r['TARGET_U_augmented_training_exposures']==2*train.U_BATCH*train.STEPS
        assert r['new_source_density_models']==2 and r['new_source_threshold_sets']==4 and r['costs']['real_signal_forward_rows']==2*sum(m.COUNTS.values())+6*m.Q_COUNT
        assert all(set(v['paths'])==set(s.CONDITIONS) and len(v['paths']['source_DURA']['groups'])==42 for v in r['methods'].values())
        assert not r['final_phase3_complete'] and not r['development_gate_passed'] and r['next_step']=='NO_PROMOTION'
        with pytest.raises(FileExistsError):s.score(root,*paths,tmp_path/'score')
        monkeypatch.setattr(s,'_source_truth',lambda *a:pytest.fail('Source truth before full matrix validation'))
        monkeypatch.setattr(s,'_target_truth',lambda *a:pytest.fail('Query truth before full matrix validation'))
        train_done=Path(c['train_config']['output_root'])/'complete.json'
        files=[root/'complete.json',root/'models/complete.json',root/'models/DA0.npz',root/'thresholds.npz',root/'targets/star_DURA/predictions.npz',train_done]
        originals={p:p.read_bytes() for p in files}
        for defect in ('unfrozen','incomplete_training','target_sourceonly','DA0_state','old_tau','private_Q_y','U_Q_overlap','missing_Q'):
            with subtests.test(defect=defect):
                for p,v in originals.items():p.write_bytes(v)
                if defect=='missing_Q':(root/'targets/star_DURA/predictions.npz').unlink()
                elif defect in ('unfrozen','incomplete_training','target_sourceonly'):
                    if defect=='target_sourceonly':
                        for p,key in ((root/'complete.json','model_identities'),(root/'models/complete.json','identities')):
                            a=m._read(p);a[key]['TARGET_SSL']['source_only']=True;p.unlink();m._write(p,a)
                        a=m._read(train_done);a['rows']['TARGET_SSL']['identity']['source_only']=True;train_done.unlink();m._write(train_done,a)
                    else:
                        p=train_done if defect=='incomplete_training' else root/'complete.json';a=m._read(p)
                        if defect=='incomplete_training':a['rows'].pop('TARGET_SSL')
                        else:a['all_thresholds_frozen_before_novel_and_Q']=False
                        p.unlink();m._write(p,a)
                else:
                    p=root/('models/DA0.npz' if defect=='DA0_state' else 'thresholds.npz' if defect=='old_tau' else 'targets/star_DURA/predictions.npz')
                    with original_load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='DA0_state':a['origin'][0]+=.1
                    elif defect=='old_tau':a['conditional'][0,0]+=.1
                    elif defect=='private_Q_y':a['y']=np.zeros(m.Q_COUNT,np.int64)
                    else:
                        with original_load(u_path,allow_pickle=False) as z:a['ids'][0]=z['ids'][0]
                    np.savez(p,**a)
                with pytest.raises((ValueError,KeyError,FileNotFoundError)):s.score(root,'source','clean','star',tmp_path/'invalid')
                assert not (tmp_path/'invalid/complete.json').exists()
