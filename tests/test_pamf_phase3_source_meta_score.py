"""One complete source-meta producer/CPU scoring path and focused negatives."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.source_meta_score')


def test_actual_meta_eight_readouts_frozen_before_three_truths(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_source_meta_run import fixture
    with fixture(tmp_path,monkeypatch) as (m,c,pm,flow,training):
        done=m.run(c,'cpu');out=Path(c['output_root']);s=api()
        assert flow==['smoke','L','OE','V','calibrate','calibrate','calibrate','calibrate','novel','clean_DURA','star_DURA']
        assert done['training_costs']['real_student_forward_rows']==426240
        assert done['combined_new_real_signal_forward_rows']==sum((m.COUNTS['L'],m.COUNTS['OE'],426240,done['real_signal_forward_rows']))
        assert done['source_density_models_fitted']==2 and done['new_threshold_sets_fitted']==4 and done['threshold_bytes']==384
        assert done['U_ID_metadata_reads']==done['target_U_reads']==0 and not done['target_DA']
        monkeypatch.setattr(s,'COUNTS',dict(m.COUNTS));monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No refit after known-only thresholds freeze'))
        _,_,rows=s._prediction(out)
        with np.load(out/'thresholds.npz',allow_pickle=False) as z:tau=z['conditional'].copy()
        for condition in m.CONDITIONS:
            for index,arm in enumerate(m.ARMS):
                a=rows[condition][arm]['known'] if condition=='source_DURA' else rows[condition][arm]
                np.testing.assert_array_equal(a['scores'],a['raw_scores']-tau[2*index:2*index+2,a['closed']].T)
                if arm in m.NEW_ARMS:assert np.any(a['closed']!=a['baseline_closed'])
        nv=rows['source_DURA']['E0']['novel'];grid=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for scene in s.common.SCENES]
        paths=[tmp_path/'source_truth.npz'];np.savez(paths[0],ids=nv['ids'],rx_i=nv['rx_i'],day_i=nv['day_i'],tx=np.asarray([v[0] for v in grid]),scenario=np.asarray([v[2] for v in grid]))
        y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);days=np.asarray(['2021_03_15']*10+['2021_03_23']*8)
        for condition in m.CONDITIONS[1:]:
            a=rows[condition]['E0'];p=tmp_path/(condition+'_truth.npz');paths.append(p)
            np.savez(p,ids=a['ids'],y=y,tx=np.asarray(s.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),day=days,
                scenario=np.asarray(['clean' if condition=='clean_DURA' else 'practical_low_urban']*len(y)))
        reads=[];source=s._source_truth;future=s._future_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1])
        monkeypatch.setattr(s,'_future_truth',lambda *a:(reads.append('future'),future(*a))[1])
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('CPU score cannot load G/F0'))
        monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('CPU score cannot load new weights'))
        monkeypatch.setattr(m.fresh,'_training_metadata',lambda *a:pytest.fail('No old discovery U IDs'))
        monkeypatch.setattr(m.density,'fit_known',lambda *a:pytest.fail('No new known fit during score'))
        monkeypatch.setattr(m.density,'fit_background',lambda *a:pytest.fail('No new background fit during score'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','future','future'] and result['adaptation_success'] is None and not result['new_target_DA']
        assert result['primary_method']=='META_NATIVE_predicted_ratio' and len(result['methods'])==8
        assert set(result['paired_primary_vs_controls'])=={'E0_predicted_ratio','OLD_PCD_predicted_ratio','ERM_NATIVE_predicted_ratio'}
        assert all(len(v['paths'])==3 for v in result['methods'].values())
        assert result['methods'][s.PRIMARY]['paths']['clean_DURA']['by_day']['2021_03_23']['A_K'] is None
        assert result['source_meta_success']==result['scientific_success']==result['development_gate_passed']
        assert not result['final_phase3_complete'] and result['full_goal_status']=='ACTIVE' and result['training_costs']['real_student_forward_rows']==426240
        expected_ak=all(result['paired_primary_vs_controls'][name]['star_DURA']['delta']['A_K']>1e-12 for name in ('ERM_NATIVE_predicted_ratio','OLD_PCD_predicted_ratio'))
        expected_ru=all(result['paired_primary_vs_controls'][name]['star_DURA']['delta']['R_U']>=-1e-12 for name in result['paired_primary_vs_controls'])
        assert result['star_A_K_gain_over_both_controls']==expected_ak and result['star_R_U_not_lower_than_all_controls']==expected_ru
        assert result['next_step'] in ('NO_PROMOTION','SOURCE_META_PASSED')
        tr=Path(c['train_config']['output_root']);q=out/'targets/star_DURA/META_NATIVE/predictions.npz'
        files=[q,out/'thresholds.npz',out/'source/E0/known_predictions.npz',tr/'complete.json',tr/'META_NATIVE/complete.json']
        saved={p:p.read_bytes() for p in files}
        for defect in ('meta_alpha','second_order','global_second_order','partial_Q','Q_private','baseline','raw_score','old_control','tau'):
            with subtests.test(defect=defect):
                for p,data in saved.items():p.write_bytes(data)
                reads.clear()
                if defect in ('meta_alpha','second_order','global_second_order'):
                    d=m._read(tr/'complete.json')
                    if defect=='global_second_order':d['full_second_order']=False
                    else:
                        row=m._read(tr/'META_NATIVE/complete.json');row['identity']['meta_alpha' if defect=='meta_alpha' else 'second_order']=0. if defect=='meta_alpha' else False
                        d['rows']['META_NATIVE']=row;d['identities']['META_NATIVE']=row['identity']
                        (tr/'META_NATIVE/complete.json').unlink();m._write(tr/'META_NATIVE/complete.json',row)
                    (tr/'complete.json').unlink();m._write(tr/'complete.json',d)
                elif defect=='partial_Q':q.unlink()
                else:
                    p=out/'thresholds.npz' if defect=='tau' else out/'source/E0/known_predictions.npz' if defect=='old_control' else q
                    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='tau':a['conditional'][7,0]+=.1
                    elif defect=='Q_private':a['y']=y
                    elif defect=='baseline':a['baseline_closed'][0]=(a['baseline_closed'][0]+1)%6
                    elif defect=='old_control':a['default_logits'][0]+=1
                    else:a['raw_scores'][0,1]+=.1
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/('bad_'+defect))
                assert reads==[] and not (tmp_path/('bad_'+defect)).exists()
        for p,data in saved.items():p.write_bytes(data)
        with pytest.raises(FileExistsError):m.run(c,'cpu')
