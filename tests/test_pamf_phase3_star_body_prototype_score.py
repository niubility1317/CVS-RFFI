"""Truth-last validation of all student readouts and fixed primary."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_body_prototype_score')


def test_all_prediction_validation_precedes_three_truths(tmp_path,monkeypatch):
    s=api();reads=[]
    monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source'))
    monkeypatch.setattr(s,'_target_truth',lambda *a:reads.append('Q'))
    monkeypatch.setattr(s,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('partial prediction')))
    with pytest.raises(ValueError):s.score('pred','source','clean','star',tmp_path/'score')
    assert reads==[] and not (tmp_path/'score').exists()


def test_actual_synthetic_models_to_CPU_score_and_reconstruction_failures(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_body_prototype_run import fixture
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        m.run(c,'cpu');out=Path(c['output_root']);monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        proxy=type('Rows',(),dict(_rows=staticmethod(lambda root,role:m._source_rows(root,'source_DURA',role)),_target_rows=staticmethod(m._target_rows)))
        score_proxy=type('Scoring',(),dict(common=s.common,CONDITIONS=s.CONDITIONS[1:]))
        paths=truth_files(tmp_path,score_proxy,proxy,out);reads=[];source=s._source_truth;target=s._target_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1]);monkeypatch.setattr(s,'_target_truth',lambda path,*a:(reads.append(Path(path).name),target(path,*a))[1])
        monkeypatch.setattr(m,'_load_model',lambda *a:pytest.fail('Independent scorer cannot reload student checkpoints'))
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Independent scorer cannot reload F0/G/r03'))
        monkeypatch.setattr(m.density,'fit_known',lambda *a:pytest.fail('Scorer cannot fit density'))
        monkeypatch.setattr(m.density,'fit_background',lambda *a:pytest.fail('Scorer cannot fit density'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('Scorer cannot fit tau'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz'] and result['truth_last']
        assert result['primary_method']=='PCD_predicted_ratio' and result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete']
        assert not result['eligible_primary'] and len(result['methods'])==6
        assert result['training_reused'] and result['training_run_id']==m.TRAIN_RUN_ID and result['this_run_optimizer_updates']==0
        files=[out/'complete.json',out/'thresholds.npz',out/'targets/star_DURA/predictions.npz',out/'models/TARGET_PCD.npz'];saved={p:p.read_bytes() for p in files}
        for defect in ('partial_Q','identity','mixed_eval_identity','new_readout','old_control','old_baseline','tau','density','private_Q'):
            with subtests.test(defect=defect):
                for p,data in saved.items():p.write_bytes(data)
                reads.clear()
                if defect=='partial_Q':(out/'targets/star_DURA/predictions.npz').unlink()
                elif defect in ('identity','mixed_eval_identity'):
                    p=out/'complete.json';a=m._read(p)
                    if defect=='identity':a['model_identities']['TARGET_PCD']['projection_updates']=443
                    else:a['model_identities']['TARGET_PCD']['run_id']=m.RUN_ID;a['model_identities']['TARGET_PCD']['code_commit']=c['code_commit']
                    p.unlink();m._write(p,a)
                else:
                    p=out/'thresholds.npz' if defect=='tau' else out/'models/TARGET_PCD.npz' if defect=='density' else out/'targets/star_DURA/predictions.npz'
                    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='tau':a['conditional'][5,0]+=.1
                    elif defect=='density':a['whitener']*=1.2
                    elif defect=='new_readout':a['raw_scores'][0,5]+=.1
                    elif defect=='old_control':a['raw_scores'][0,1]+=.1
                    elif defect=='old_baseline':a['baseline_closed'][0]=(a['closed'][0]+1)%6
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                bad=tmp_path/('bad_'+defect)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,bad)
                assert reads==[] and not bad.exists()
